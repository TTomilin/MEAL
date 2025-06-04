import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict


@flax.struct.dataclass
class PacknetState:
    '''
    Class to store the state of the Packnet
    '''
    masks: FrozenDict
    current_task: int
    train_mode: bool



class Packnet():
    '''
    Class that implements the Packnet CL-method
    '''
    def __init__(self, 
                 seq_length, 
                 prune_instructions=0.5, 
                 train_finetune_split=(1,1), 
                 prunable_layers=(nn.Conv, nn.Dense)
            ):
        '''
        Initializes the Packnet class
        @param seq_length: the length of the sequence
        @param prune_instructions: the percentage of the network to prune
        @param train_finetune_split: the split between training and finetuning
        @param prunable_layers: the layers that can be pruned
        '''
        self.seq_length = seq_length
        self.prune_instructions = prune_instructions
        self.train_finetune_split = train_finetune_split
        self.prunable_layers = prunable_layers

    def init_mask_tree(self, params):
        '''
        Initializes a pytree with a fixed size and shape to store all masks of previous tasks
        @param params: the parameters of the model, to get the shape of the masks
        Returns a mask Pytree of shape (seq_length, *params.shape) per leaf
        '''
        def make_mask_leaf(leaf):
            '''
            Initializes a mask for a single leaf
            @param leaf: the leaf of the pytree
            returns a mask that mirrors the parameter shape, but with a leading dimension of the number of tasks
            '''
            shape = (self.seq_length,) + leaf.shape
            return jnp.zeros(shape, dtype=bool)
        
        return jax.tree_util.tree_map(make_mask_leaf, params)
    
    def update_mask_tree(self, mask_tree, new_mask, current_task):
        '''
        Updates the mask tree with a new mask
        @param mask_tree: the current mask tree
        @param new_mask: the new mask to add
        returns the updated mask tree
        '''
        def update_mask_leaf(old_leaf, new_leaf):
            '''
            Updates a single leaf (a kernel or bias array of params) with a new mask
            @param mask: the current mask
            @param new_mask: the new mask to add
            returns the updated mask
            '''
            return old_leaf.at[current_task].set(new_leaf)

        return jax.tree_util.tree_map(update_mask_leaf, mask_tree, new_mask)

    def combine_masks(self, mask_tree, last_task):
        '''
        Combines the masks of all old tasks into a single mask to compare the current task against
        @param mask_tree: the mask tree
        returns the combined mask (mask with True for all fixed weights of previous tasks)
        '''
        def combine_masks_leaf(leaf):
            '''
            Combines the masks of all tasks for a single leaf (kernel or bias)
            @param leaf: the leaf of the mask tree
            returns the combined mask
            '''
            max_tasks = self.seq_length
            def combine_for_last_task(last_task):
                indices = jnp.arange(max_tasks)
                
                # Build a boolean mask where each element is True if its index is less than last_task
                prev_tasks = jax.lax.lt(indices, last_task) 
                prev_tasks = jax.lax.convert_element_type(prev_tasks, jnp.bool_)  
                
                # Reshape the prev_tasks mask to match the shape of the leaf
                new_shape = (max_tasks,) + (1,) * (leaf.ndim - 1) # (max_tasks, 1, 1, ...) 
                prev_tasks = jnp.reshape(prev_tasks, new_shape)
                
                # keep only the masks of the previous tasks, set the rest to all False
                masked = jnp.where(prev_tasks, leaf, False)
                
                # Combine the masks over all tasks 
                return jnp.any(masked, axis=0)
            
            return jax.lax.cond(last_task == 0,
                                lambda _: jnp.zeros(leaf.shape[1:], dtype=jnp.bool_),
                                combine_for_last_task,
                                last_task)
        return jax.tree_util.tree_map(combine_masks_leaf, mask_tree)
    
    def get_mask(self, mask_tree, task_id):
        '''
        returns the mask of a given task
        @param mask_tree: the mask tree
        @param task_id: the task id
        returns the mask of the given task
        '''
        def slice_mask_leaf(leaf):
            '''
            Slices the mask of a single leaf
            @param leaf: the leaf of the mask tree
            returns the mask of the given task
            '''
            return leaf[task_id]
        return jax.tree_util.tree_map(slice_mask_leaf, mask_tree)        

    def create_pruning_percentage(self, state: PacknetState):
        '''
        Creates the pruning instructions based on the sequence length
        '''
        assert self.seq_length is not None, "Sequence length not provided"

        num_tasks_left = self.seq_length - state.current_task
        prune_percentage = num_tasks_left/(num_tasks_left + 1)
        return prune_percentage
        
    def layer_is_prunable(self, layer_name):
        '''
        Checks if a layer is prunable
        @param layer_name: the name of the layer
        returns a boolean indicating whether the layer is prunable
        '''
        for prunable_type in self.prunable_layers:
            if prunable_type.__name__ in layer_name:
                return True
        return False
    
    def prune(self, params, state: PacknetState):
        '''
        Prunes the model based on the pruning instructions
        @param model: the model to prune
        @param prune_quantile: the quantile to prune
        @param state: the packnet state
        returns the pruned model
        '''

        masks = jax.lax.cond(
            (state.current_task == 0) & (state.masks is None),
            lambda _: self.init_mask_tree(params),
            lambda _: state.masks,
            operand=None
        )

        state = state.replace(masks=masks)

        # Compute the pruning quantile
        prune_perc = self.create_pruning_percentage(state)

        # Get the combined mask of all previous tasks
        combined_mask = self.combine_masks(state.masks, state.current_task)
        sparsity_mask = self.compute_sparsity(combined_mask)
        jax.debug.print("sparsity_mask: {sparsity_mask}", sparsity_mask=sparsity_mask)
        
        # Create a list for all prunable parameters
        all_prunable = jnp.array([])
        for layer_name, layer_dict in params.items():
            for param_name, param_array in layer_dict.items():
                if (self.layer_is_prunable(layer_name)) and ("bias" not in param_name):
                    # get the combined mask for this layer
                    prev_mask_leaf = combined_mask[layer_name][param_name]

                    # Get parameters not used by previous tasks
                    p = jnp.where(jnp.logical_not(prev_mask_leaf), param_array, jnp.nan)

                    # Concatenate with existing prunable parameters
                    if p.size > 0: 
                        all_prunable = jnp.concatenate([all_prunable.reshape(-1), p.reshape(-1)], axis=0)

        cutoff = jnp.nanquantile(jnp.abs(all_prunable), prune_perc)
        jax.debug.print("cutoff: {cutoff}", cutoff=cutoff)
        # count the number of params under the cutoff
        num_pruned = jnp.sum(jnp.abs(all_prunable) < cutoff)
        jax.debug.print("number of params to be pruned: {num_pruned}", num_pruned=num_pruned)
        mask = {}
        new_params = {}

        for layer_name, layer_dict in params.items():
            new_layer = {}
            mask_layer = {}
            for param_name, param_array in layer_dict.items():
                if (self.layer_is_prunable(layer_name)) and ("bias" not in param_name):
                    # get the params that are used by the previous tasks
                    prev_mask_leaf = combined_mask[layer_name][param_name] 

                    # Create new mask for the current parameter array
                    new_mask_leaf = jnp.logical_and(
                        jnp.abs(param_array) >= cutoff, 
                        jnp.logical_not(prev_mask_leaf)
                    )
                    # keep the fixed parameters and the parameters above the cutoff
                    complete_mask = jnp.logical_or(prev_mask_leaf, new_mask_leaf)

                    # Generate small random values instead of zeros
                    rng_key = jax.random.PRNGKey(state.current_task + 42)
                    rng_key = jax.random.fold_in(rng_key, hash(layer_name + param_name))
                    small_random_values = jax.random.normal(
                        rng_key, param_array.shape) * 0.001  # Small initialization
                
                    # prune the parameters
                    pruned_params = jnp.where(complete_mask, param_array, 0)
                    
                    mask_layer[param_name] = new_mask_leaf
                    new_layer[param_name] = pruned_params
                else:
                    mask_layer[param_name] = jnp.zeros(param_array.shape, dtype=bool)
                    new_layer[param_name] = param_array
            
            new_params[layer_name] = new_layer
            mask[layer_name] = mask_layer

        masks = self.update_mask_tree(state.masks, mask, state.current_task)
        state = state.replace(masks=masks)

        new_param_dict = new_params
        return new_param_dict, state

    def train_mask(self, state: PacknetState, train_state, params_copy): 
        '''
        Zeroes out the gradients of the fixed weights of previous tasks. 
        This mask should be applied after backpropagation and before each optimizer step during training
        '''

        # check if there are any masks to apply
        def first_task():
            # No previous tasks to fix - create a mask with the same process as combine_masks
            # but with all False values
            prev_mask = jax.tree_util.tree_map(
                lambda x: jnp.zeros_like(x, dtype=bool), 
                train_state.params["params"]
            )
            return prev_mask
        
        def other_tasks():
            # get all weights allocated for previous tasks 
            prev_mask = self.combine_masks(state.masks, state.current_task)
            return prev_mask
        
        prev_mask = jax.lax.cond(
            state.current_task == 0,
            first_task,
            other_tasks,
        )

        def reset_params_train(param_leaf, param_copy_leaf, mask_leaf):
            """
            Resets the parameters to the old parameters if the parameter is fixed,
            to counteract the possible momentum that is still present in the update
            """
            # if the parameter is fixed (True), set it to the old parameter
            return jnp.where(mask_leaf, param_copy_leaf, param_leaf)
        
        # Extract the inner parameter dictionaries to match structures
        inner_params = train_state.params["params"]
        inner_params_copy = params_copy["params"]
        
        # apply the reset function to all parameters
        new_params = jax.tree_util.tree_map(reset_params_train, inner_params, inner_params_copy, prev_mask)

        return {"params": new_params}

    def fine_tune_mask(self, state: PacknetState, train_state, params_copy):
        '''
        Zeroes out the gradient of the pruned weights of the current task and previously fixed weights 
        This mask should be applied before each optimizer step during fine-tuning
        '''
        
        current_mask = self.get_mask(state.masks, state.current_task)

        def reset_params_finetune(param_leaf, param_copy_leaf, mask_leaf):
            """
            Resets the parameters to the old parameters if the parameter is fixed,
            to counteract the possible momentum that is still present in the update
            """
            # if the parameter is pruned (False), set it to the old parameter
            return jnp.where(mask_leaf, param_leaf, param_copy_leaf)
        
        # Extract the inner parameter dictionaries to match structures
        inner_params = train_state.params["params"]
        inner_params_copy = params_copy["params"]
        
        # apply the reset function to all parameters
        new_params = jax.tree_util.tree_map(reset_params_finetune, inner_params, inner_params_copy, current_mask)

        return {"params": new_params}

    def fix_biases(self, state: PacknetState):
        '''
        Set all masks for the biases to True after the first task,
        so that the biases will not be updated after the first task
        '''

        masks = state.masks
        def after_first_task(masks):
            # Iterate over all masks and set the biases to True
            for layer_name, layer_dict in masks.items():
                for param_name, mask_array in layer_dict.items():
                    if "bias" in param_name:
                        # Set the mask to True for all tasks
                        mask_array = jnp.ones(mask_array.shape, dtype=bool)
            return masks
        
        def first_task(masks):
            # No previous tasks to fix
            return masks
        
        masks =  jax.lax.cond(state.current_task == 0, first_task, after_first_task, masks)
        state = state.replace(masks=masks)

        return state

    def apply_eval_mask(self, params, task_id, state: PacknetState):
        '''
        Applies the mask of a given task to the model to revert to that network state
        '''
        assert len(state.masks) > task_id, "Current task index exceeds available masks"

        masked_params = {}

        # Iterate over prunable layers and collect the masks of previous tasks
        for layer_name, layer_dict in params.items():
            masked_layer_dict = {}
            for param_name, param_array in layer_dict.items():
                if self.layer_is_prunable and "bias" not in param_name:
                    full_param_name = f"{layer_name}/{param_name}"
                    prev_mask = jnp.zeros(param_array.shape, dtype=bool)
                    for i in range(0, task_id+1):
                        prev_mask = jnp.logical_or(prev_mask, state.masks[i][full_param_name])

                    # Zero out all weights that are not in the mask for this task
                    masked_layer_dict[param_name] = param_array * prev_mask
                else:
                    masked_layer_dict[param_name] = param_array
                
            masked_params[layer_name] = masked_layer_dict

        return masked_params                
        
    def mask_remaining_params(self, params, state: PacknetState):
        '''
        Masks the remaining parameters of the model that are not pruned
        typically called after the last task's initial training phase
        '''
        prev_mask = self.combine_masks(state.masks, state.current_task)

        mask = {}

        for layer_name, layer_dict in params.items():
            mask_layer = {}
            for param_name, param_array in layer_dict.items():
                if self.layer_is_prunable(layer_name) and "bias" not in param_name:

                    prev_mask_leaf = prev_mask[layer_name][param_name]
                    new_mask_leaf = jnp.logical_not(prev_mask_leaf)

                    mask_layer[param_name] = new_mask_leaf
                    
                else:
                    mask_layer[param_name] = jnp.zeros(param_array.shape, dtype=bool)

            mask[layer_name] = mask_layer

        masks = self.update_mask_tree(state.masks, mask, state.current_task)
        state = state.replace(masks=masks)

        # create the parameters to return the same shape as prune
        new_param_dict = params

        return new_param_dict, state

    def on_train_end(self, params, state: PacknetState):
        '''
        Handles the end of the training phase on a task
        '''
        # change the mode to finetuning
        state = state.replace(train_mode=False)

        def last_task(params):
            # if we are on the last task, mask all remaining parameters
            return self.mask_remaining_params(params, state)
             
        def other_tasks(params):
            return self.prune(params, state)
            
        
        new_params, state = jax.lax.cond(
            state.current_task == self.seq_length-1,
            last_task,
            other_tasks,
            params
        )
        # fix the structure of the params:
        new_params = {"params": new_params}
        return new_params, state
    
    def on_finetune_end(self, state: PacknetState):
        '''
        Handles the end of the finetuning phase on a task
        '''
        state = state.replace(current_task=state.current_task+1, train_mode=True)
        
        return state

    def on_backwards_end(self, state: PacknetState, actor_train_state, params_copy):
        '''
        Handles the end of the backwards pass
        '''

        # fix the biases of the gradients
        state = self.fix_biases(state)

        def finetune(state):
            '''
            Revert the masked params to their original values
            '''
            return self.fine_tune_mask(state, actor_train_state, params_copy)
        
        def train(state):
            '''
            Revert the masked params to their original values
            '''
            return self.train_mask(state, actor_train_state, params_copy)   
        
        new_params = jax.lax.cond(
            state.train_mode, 
            train, 
            finetune,
            state
        )

        actor_train_state = actor_train_state.replace(params=new_params)
        return actor_train_state

    def get_total_epochs(self):
        return self.train_finetune_split[0] + self.train_finetune_split[1]
    
    def compute_sparsity(self, params):
        """Calculate percentage of zero weights in model"""
        total_params = 0
        zero_params = 0
        
        for layer_name, layer_dict in params.items():
            for param_name, param_array in layer_dict.items():
                if "kernel" in param_name:  # Only weight parameters
                    total_params += param_array.size
                    zero_params += jnp.sum(jnp.abs(param_array) < 1e-5)
        
        # print(f"Total params: {total_params}, Zero params: {zero_params}")
        
        sparsity = zero_params / total_params if total_params > 0 else 1
        sparsity = jnp.round(sparsity, 4)
        return sparsity
