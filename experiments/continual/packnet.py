import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from experiments.continual.base import CLState, CLMethod
from typing import List


@flax.struct.dataclass
class PacknetState(CLState):
    '''
    Class to store the state of the Packnet
    '''
    masks: FrozenDict
    current_task: int
    train_mode: bool
    weight_memory: List[dict]
    mask_memory: List[dict]



class Packnet(CLMethod):
    '''
    Class that implements the Packnet CL-method
    '''
    def __init__(self, 
                 seq_length, 
                 prune_instructions=0.5, 
                 train_finetune_split=(1,1), 
                 prunable_layers=(nn.Conv, nn.Dense),
                 norm_layer_types=(nn.BatchNorm, nn.LayerNorm),
                 re_init_pruned_weights=False
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

        # variables controlling which weights to freeze
        self.prunable_layer_type_names = [layer_type.__name__ for layer_type in prunable_layers]
        self.normalization_layer_type_names = [layer_type.__name__ for layer_type in norm_layer_types]
        self.forbidden_param_strings = ['bias'] # ignore bias and critic parameters
        self.forbidden_layer_strings = ['critic']
        
        # wether to re-initialize weights to small values after each fine-tuning:
        self.re_init_pruned_weights = re_init_pruned_weights

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

        num_tasks_left = self.seq_length - state.current_task - 1
        prune_percentage = num_tasks_left / (num_tasks_left + 1)
        return prune_percentage
    
    def param_is_prunable(self, param_name):
        '''
        Checks if a parameter is prunable.
        '''
        return not(any([n in param_name for n in self.forbidden_param_strings]))

    def layer_is_for_norm(self, layer_name):
        '''
        Checks if a layer is for normalization
        @param layer_name: the name of the layer
        returns a boolean indicating whether the layer is for normalization
        '''
        if any(n in layer_name for n in self.normalization_layer_type_names):
            return not(any([n in layer_name for n in self.forbidden_layer_strings]))
        
    def layer_is_prunable(self, layer_name):
        '''
        Checks if a layer is prunable
        @param layer_name: the name of the layer
        returns a boolean indicating whether the layer is prunable
        '''
        if any(n in layer_name for n in self.prunable_layer_type_names):
            return not(any([n in layer_name for n in self.forbidden_layer_strings]))
    
    def add_norm_layers_to_mask(self, params, mask):
        '''
        Modifies given mask to also freeze all normalization layers.
        '''
        new_mask = {}

        for layer_name, layer_dict in params.items():
            mask_layer = {}
            if self.layer_is_for_norm(layer_name):
                # if layer for normalization, mask completely
                for param_name, param_array in layer_dict.items():
                    new_mask_leaf = jnp.ones_like(param_array, dtype=bool)
                    mask_layer[param_name] = new_mask_leaf
            else:
                # if layer not for normalization, use previous mask
                for param_name, param_array in layer_dict.items():
                    mask_layer[param_name] = mask[layer_name][param_name]
            new_mask[layer_name] = mask_layer

        return new_mask
    
    def add_biases_to_mask(self, params, mask):
        '''
        Modifies given mask to also freeze all biases.
        '''
        new_mask = {}

        for layer_name, layer_dict in params.items():
            mask_layer = {}
            for param_name, param_array in layer_dict.items():
                if "bias" in param_name:
                    # if bias, mask all:
                    mask_layer[param_name] = jnp.ones_like(param_array, dtype=bool)
                else:
                    # if not, use previous mask:
                    mask_layer[param_name] = mask[layer_name][param_name]
            new_mask[layer_name] = mask_layer

        return new_mask
    
    def fix_biases_and_normalization(self, params, state: PacknetState):
        # add biases an normalization to current mask:
        current_mask = self.get_mask(state.masks, state.current_task)
        new_mask = self.add_norm_layers_to_mask(params["params"], current_mask)
        new_mask = self.add_biases_to_mask(params["params"], new_mask)
        # update the state's mask tree:
        masks = self.update_mask_tree(state.masks, new_mask, state.current_task)
        state = state.replace(masks=masks)
        # return the state:
        return state


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

        mask = {}
        new_params = {}

        for layer_name, layer_dict in params.items():
            new_layer = {}
            mask_layer = {}

            # collect prunable parameters for layer:
            layer_prunable = []
            if self.layer_is_prunable(layer_name):
                for param_name, param_array in layer_dict.items():
                    if self.param_is_prunable(param_name):
                        prev_mask_leaf = combined_mask[layer_name][param_name]
                        p = jnp.where(
                            prev_mask_leaf,
                            jnp.nan,
                            jnp.abs(param_array)
                        ) # keep only parameters not yet reserved

                        if p.size > 0:
                            layer_prunable.append(p.reshape(-1))
            # compute layer cutoff:
            if len(layer_prunable) > 0:
                layer_prunable = jnp.concatenate(layer_prunable)
                cutoff = jnp.nanquantile(layer_prunable, prune_perc)
                num_pruned = jnp.sum(layer_prunable <= cutoff)
                jax.debug.print(
                    "Layer {}, cutoff: {}, num_pruned: {}",
                    layer_name,
                    cutoff,
                    num_pruned
                ) # log info for layer
            elif self.layer_is_prunable(layer_name):
                cutoff = None
                jax.debug.print(
                    "Layer: {}, no more pruning possible",
                    layer_name
                ) # log if pruning complete
            else:
                cutoff = None

            # actually apply the pruning:
            for param_name, param_array in layer_dict.items():
                # in case the layer is prunable and some parameters can still be pruned:
                if (self.layer_is_prunable(layer_name)
                    and self.param_is_prunable(param_name)
                    and cutoff is not None):
                    prev_mask_leaf = combined_mask[layer_name][param_name]
                    new_mask_leaf = jnp.logical_and(
                        jnp.abs(param_array) > cutoff,
                        jnp.logical_not(prev_mask_leaf)
                    )
                    complete_mask = jnp.logical_or(prev_mask_leaf, new_mask_leaf)

                    pruned_params = jnp.where(
                        complete_mask,
                        param_array,
                        0
                    ) # replace all unmasked regions with zero

                    # update the values in mask_layer and new_layer:
                    mask_layer[param_name] = new_mask_leaf
                    new_layer[param_name] = pruned_params
                # in case no pruning is possible:
                else:
                    mask_layer[param_name] = jnp.zeros(
                        param_array.shape, dtype=bool # set mask to all zeroes
                    )
                    new_layer[param_name] = param_array # leave parameters untouched

            new_params[layer_name] = new_layer
            mask[layer_name] = mask_layer

        # update and save mask tree:
        masks = self.update_mask_tree(state.masks, mask, state.current_task)
        state = state.replace(masks=masks)

        # return:
        new_param_dict = new_params
        return new_param_dict, state     

    def _deterministic_leaf_init(self, task_id: int, path, leaf):
        '''
        Create the initial values for a given leaf, depending deterministically
        on the task_id, layer name, and parameter name.
        '''
        layer_name, param_name = str(path[0]), str(path[1])
        rng_key = jax.random.PRNGKey(task_id + 42)
        rng_key = jax.random.fold_in(
            rng_key,
            hash(layer_name + param_name) & 0xFFFFFFFF # ensure positivity
        )
        return (jax.random.normal(rng_key, leaf.shape) * 1e-6)
    
    def _get_deterministic_init(self, task_id: int, param_tree: dict):
        '''
        Create the deterministic initial values for a given parameter tree and task_id.
        '''
        leaf_initiatior = lambda path, leaf: self._deterministic_leaf_init(task_id=task_id, path=path, leaf=leaf)
        return jax.tree.map_with_path(leaf_initiatior, param_tree)
    
    def initialize_pruned_weights(self, params, state: PacknetState):
        '''
        Deterministically sets the pruned weights to small gaussian values based on the current task index,
        the layer names, and the parameter names. Call this method after fine-tuning and after incrementing 
        the PacknetState's task index to prevent overriding tuned parameters.
        '''
        unpacked_params = params["params"]
        prev_tasks_mask = self.combine_masks(state.masks, state.current_task) # mask of all tasks < state.current_task
        small_init = self._get_deterministic_init(state.current_task, unpacked_params)
        
        def init_pruned_weights_leaf(mask, p, init):
            return jnp.where(mask, p, init)

        new_params = jax.tree_util.tree_map(
            init_pruned_weights_leaf,
            prev_tasks_mask,
            unpacked_params,
            small_init
        )

        return {**params, 'params': new_params}

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
                if self.layer_is_prunable(layer_name) and self.param_is_prunable(param_name):

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
    
    def on_train_end(self, train_state, cl_state: PacknetState):
        '''
        Handles pruning and retrieving updated parameters/optimizer after training.
        '''
        # Prune the model and update the parameters:
        new_params, cl_state = self.dispatch_prune(train_state.params, cl_state)
        # compute and log sparsity:
        sparsity = self.compute_sparsity(new_params["params"])
        jax.debug.print(
        "Sparsity after pruning: {sparsity}", sparsity=sparsity)
        # update train_state:        
        train_state = self._update_train_state(train_state, new_params)
        # return train_state and cl_state:
        return train_state, cl_state

    def dispatch_prune(self, params, state: PacknetState):
        '''
        Handles the end of the training phase on a task
        '''
        # change the mode to finetuning
        state = state.replace(train_mode=False)
        unpacked_params = params["params"]

        def last_task(unpacked_params):
            # if we are on the last task, mask all remaining parameters
            return self.mask_remaining_params(unpacked_params, state)

        def other_tasks(unpacked_params):
            return self.prune(unpacked_params, state)


        new_params, state = jax.lax.cond(
            state.current_task == self.seq_length-1,
            last_task,
            other_tasks,
            unpacked_params
        )

        new_params = {**params, "params": new_params}
        return new_params, state

    def on_finetune_end(self, train_state, state: PacknetState):
        '''
        Handles the end of the finetuning phase on a task
        '''
        # compute and report sparsity:
        sparsity = self.compute_sparsity(train_state.params["params"])
        jax.debug.print(
            "Sparsity after finetuning: {sparsity}", sparsity=sparsity)
        # If the first task was just tuned, freeze biases and normalization layers:
        state = jax.lax.cond(
            state.current_task == 0,
            lambda: self.fix_biases_and_normalization(train_state.params, state),
            lambda: state
        )
        # update task id after tuning:
        state = state.replace(current_task=state.current_task+1, train_mode=True)
        if self.re_init_pruned_weights:
            # initialize weights to small values:
            new_params = self.initialize_pruned_weights(train_state.params, state)
            # update train_state:        
            train_state = self._update_train_state(train_state, new_params)
        # return train_state and state:
        return train_state, state

    def _update_train_state(self, train_state, new_params):
        '''
        Updates train state with new parameters and optimizers.
        '''
        train_state = train_state.replace(params=new_params)
        new_opt_state = train_state.tx.init(train_state.params)
        train_state = train_state.replace(opt_state=new_opt_state)
        return train_state
    
    def update_and_verify_weight_memory(self, params, state: PacknetState):
        mask = self.combine_masks(state.masks, state.current_task)
        state.weight_memory.append(params["params"].copy())
        state.mask_memory.append(mask.copy())
        for i in range(len(state.weight_memory)):
            mask_i = state.mask_memory[i]
            weights_i = state.weight_memory[i]
            for j in range(i+1, len(state.weight_memory)):
                weights_j = state.weight_memory[j]
                self.compare_tree_dicts(weights_i, weights_j, mask_i, i, j)

    def compare_tree_dicts(self, dict_a, dict_b, mask_tree, i, j):
        for layer_name, layer_dict in dict_a.items():
            for module_name, module_array in layer_dict.items():
                array_mask = mask_tree[layer_name][module_name]
                array_a = jnp.where(array_mask, module_array, 0)
                array_b = jnp.where(array_mask, dict_b[layer_name][module_name], 0)
                out = jnp.all(array_a == array_b)
                jax.debug.print("{},{}: {}|{} {}", i, j, layer_name, module_name, out)
        return out


    def mask_gradients(self, state: PacknetState, gradients):
        '''
        Masks gradients for frozen weights before the optimizer step.
        This is the proper PackNet approach - zero gradients before optimizer sees them.
        '''

        def first_task():
            # No previous tasks to mask - return gradients unchanged
            return gradients

        def train_mode():
            # Training mode: mask gradients for weights from previous tasks
            prev_mask = self.combine_masks(state.masks, jnp.maximum(state.current_task, 0))

            def mask_gradient_leaf(grad_leaf, mask_leaf):
                """
                Zero out gradients for frozen weights (where mask is True)
                """
                return jnp.where(mask_leaf, jnp.zeros_like(grad_leaf), grad_leaf)
            
            # Apply masking to gradients
            masked_grads = jax.tree_util.tree_map(mask_gradient_leaf, gradients["params"], prev_mask)
            return {**gradients, "params": masked_grads}

        def finetune_mode():
            # Fine-tuning mode: mask gradients for pruned weights of current task
            current_mask = self.get_mask(state.masks, state.current_task)

            def mask_gradient_leaf(grad_leaf, mask_leaf):
                """
                Zero out gradients for pruned weights (where mask is False)
                Keep gradients for active weights (where mask is True)
                """
                return jnp.where(mask_leaf, grad_leaf, jnp.zeros_like(grad_leaf))

            # Apply masking to gradients
            masked_grads = jax.tree_util.tree_map(mask_gradient_leaf, gradients["params"], current_mask)
            return {**gradients, "params": masked_grads}

        def train_mode_dispatch():
            # Dispatch between first task and other tasks in training mode
            return jax.lax.cond(
                state.current_task == 0,
                lambda: first_task(),
                lambda: train_mode()
            )

        # Apply gradient masking based on current task and mode using JAX conditionals
        return jax.lax.cond(
            state.train_mode,
            train_mode_dispatch,
            finetune_mode
        )

    def compute_sparsity(self, params):
        """Calculate percentage of zero weights in model"""
        total_params = 0
        zero_params = 0

        for layer_name, layer_dict in params.items():
            for param_name, param_array in layer_dict.items():
                if "kernel" in param_name:  # Only weight parameters
                    total_params += param_array.size
                    zero_params += jnp.sum(jnp.abs(param_array) < 1e-7)

        # print(f"Total params: {total_params}, Zero params: {zero_params}")

        sparsity = zero_params / total_params if total_params > 0 else 1
        sparsity = jnp.round(sparsity, 4)
        return sparsity

    def apply_mask(self, params, combined_mask):
        """
        Apply a given task mask to the parameters, used at evaluation time (as per original Packnet paper).
        """

        def mask_leaf(p, mask):
            # mask == True → frozen weight → keep parameter
            # mask == False → free weight → discard parameter
            return jnp.where(mask, p, 0)

        masked_params = jax.tree_util.tree_map(
            mask_leaf,
            params["params"],
            combined_mask
        )

        return {**params, "params": masked_params}

def debug_packnet_masks(state: PacknetState, params):
    frozen_counts = {}
    current_task_counts = {}
    free_counts = {}
    total_counts = {}

    # Loop over layers using Python dict iteration
    for layer_name, layer_dict in params.items():
        frozen_counts[layer_name] = {}
        current_task_counts[layer_name] = {}
        free_counts[layer_name] = {}
        total_counts[layer_name] = {}

        for param_name, param_array in layer_dict.items():
            # Mask arrays
            mask_tree = state.masks[layer_name][param_name]

            # Previous tasks mask (combined)
            prev_mask = jnp.any(mask_tree, axis=0)

            # Current task mask
            current_mask = mask_tree[state.current_task]

            # Counts
            frozen_count = jnp.sum(prev_mask)
            current_count = jnp.sum(current_mask)
            free_mask = jnp.logical_not(jnp.logical_or(prev_mask, current_mask))
            free_count = jnp.sum(free_mask)
            total_count = param_array.size

            # Store as JAX arrays (safe for JIT)
            frozen_counts[layer_name][param_name] = frozen_count
            current_task_counts[layer_name][param_name] = current_count
            free_counts[layer_name][param_name] = free_count
            total_counts[layer_name][param_name] = total_count

            # Optional: print via jax.debug.print (works inside jit)
            jax.debug.print(
                "Layer: {}, Param: {} | Total: {} | Frozen: {} | Current: {} | Free: {}",
                layer_name, param_name, total_count, frozen_count, current_count, free_count
            )