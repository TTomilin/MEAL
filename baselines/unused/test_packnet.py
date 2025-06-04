import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import unittest
from typing import List, Tuple, Dict, Any, NamedTuple

# Import your PackNet implementation
from baselines.unused.IPPO_MLP_Packnet import Packnet, PacknetState

class TestPacknet(unittest.TestCase):
    """Tests for the PackNet implementation."""

    def setUp(self):
        """Set up a simple model and PackNet instance for testing."""
        # Define a simple network structure for testing
        self.seq_length = 2
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 5
        
        # Create deterministic parameters for testing
        key = jax.random.PRNGKey(42)
        
        # Create a simple two-layer network params dictionary
        params = {
            "params": {
                "Dense_0": {
                    "kernel": jax.random.normal(key, (self.input_size, self.hidden_size)),
                    "bias": jnp.zeros(self.hidden_size)
                },
                "Dense_1": {
                    "kernel": jax.random.normal(key, (self.hidden_size, self.output_size)),
                    "bias": jnp.zeros(self.output_size)
                }
            }
        }
        
        self.params = params["params"]
        self.packnet = Packnet(
            seq_length=self.seq_length,
            prune_instructions=0.5,  # Prune 50% of weights
            train_finetune_split=(100, 50),
            prunable_layers=[nn.Dense]
        )
        
        # Initialize PackNet state
        self.packnet_state = PacknetState(
            masks=self.packnet.init_mask_tree(self.params),
            current_task=0,
            train_mode=True
        )

   
        





    def test_mask_initialization(self):
        """Test that masks are initialized correctly."""
        masks = self.packnet_state.masks
        
        # Check mask structure matches params structure
        self.assertEqual(set(masks.keys()), set(self.params.keys()))
        
        # Each mask should be initialized to all False (nothing masked yet)
        for layer_name, layer_masks in masks.items():
            for param_name, mask in layer_masks.items():
                if "kernel" in param_name:
                    # The masks have the same shape as the parameters with a leading dimension for tasks
                    expected_shape = (self.seq_length,) + self.params[layer_name][param_name].shape
                    self.assertEqual(mask.shape, expected_shape)

                    # Check that all values in the mask are initialized as False
                    for task_idx in range(self.seq_length):
                        task_mask = mask[task_idx]
                        self.assertTrue(jnp.all(~task_mask), 
                                    f"Initial mask for task {task_idx} should have all values False")

    def test_prune_quantile_computation(self):
        """Test that prune quantile is correctly computed."""
        # Create specific params with known values for testing, including bias
        test_params = {
            "Dense_0": {
                "kernel": jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                "bias": jnp.array([0.01, 0.02, 0.03])
            },
            "Dense_1": {
                "kernel": jnp.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
                "bias": jnp.array([0.04, 0.05])
            }
        }
        
        # Manually flatten all weights (only kernel params are prunable)
        all_weights = jnp.concatenate([
            test_params["Dense_0"]["kernel"].flatten(),
            test_params["Dense_1"]["kernel"].flatten()
        ])

    
        # Calculate expected cutoff at 50% quantile
        expected_cutoff = jnp.quantile(jnp.abs(all_weights), 0.5)
        print(f"Expected cutoff: {expected_cutoff}")
        
        # Create a packnet state with our test parameters
        test_packnet_state = PacknetState(
            masks=self.packnet.init_mask_tree(test_params),
            current_task=0,
            train_mode=True
        )
        
        # Prune the parameters 
        pruned_params, _ = self.packnet.prune(test_params, 0.5, test_packnet_state)
        
        # Verify weights below cutoff are zeroed and above cutoff are preserved
        for layer_name, layer_dict in pruned_params.items():
            for param_name, param in layer_dict.items():
                if "kernel" in param_name:
                    # Original param for comparison
                    orig_param = test_params[layer_name][param_name]
                    
                    # Check if values below cutoff are zeroed
                    should_be_pruned = jnp.abs(orig_param) < expected_cutoff
                    self.assertTrue(jnp.all(param[should_be_pruned] <= 0.00001),
                                f"Values below cutoff {expected_cutoff} should be zeroed")
                    
                    # Check if values above cutoff are preserved
                    should_be_kept = jnp.abs(orig_param) >= expected_cutoff
                    self.assertTrue(jnp.allclose(
                        param[should_be_kept],
                        orig_param[should_be_kept]
                    ), "Values above cutoff should be preserved")
                
                # Verify bias parameters are not pruned
                if "bias" in param_name:
                    self.assertTrue(jnp.allclose(
                        param,
                        test_params[layer_name][param_name]
                    ), "Bias parameters should not be pruned")

    def test_prune_mask(self):
        """
        Test that the new mask is computed correctly
        """
        # Create specific params with known values for testing
        test_params = {
            "Dense_0": {
                "kernel": jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                "bias": jnp.array([0.01, 0.02, 0.03])
            },
            "Dense_1": {
                "kernel": jnp.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
                "bias": jnp.array([0.04, 0.05])
            }
        }
        
        # Create a mask tree with known values
        mask_tree = {
            "Dense_0": {
                "bias": jnp.array([[False, True, True], 
                                   [False, False, False]]),
                "kernel": jnp.array([[[True, True, False], [False, True, True]], 
                                     [[False, False, False], [False, False, False]]])
            },
            "Dense_1": {
                "bias": jnp.array([[False, True], 
                                   [False, False]]),
                "kernel": jnp.array([[[True, False], [True, True], [False, True]], 
                                     [[False, False], [False, False], [False, False]]])
            }
        }

        # Create a packnet state with our test parameters
        test_packnet_state = PacknetState(
            masks=mask_tree,
            current_task=1,
            train_mode=True
        )

        # Prune the parameters
        new_params, state = self.packnet.prune(test_params, 0.5, test_packnet_state)
        # print(f"new mask tree: {state.masks}")

        # create expected mask
        expected_mask = {
            "Dense_0": {
                "bias": jnp.array([[False, True, True], 
                                   [False, False, False]]),
                "kernel": jnp.array([[[True, True, False], [False, True, True]], 
                                     [[True, True, False], [False, True, True]]])
            },
            "Dense_1": {
                "bias": jnp.array([[False, True], 
                                   [False, False]]),
                "kernel": jnp.array([[[True, False], [True, True], [False, True]], 
                                     [[True, True], [True, True], [True, True]]])
            }
        }

        # create expected new params
        expected_new_params = {
            "Dense_0": {
                "kernel": jnp.array([[0.1, 0.2, 0.0], [0.0, 0.5, 0.6]]),
                "bias": jnp.array([0.01, 0.02, 0.03])
            },
            "Dense_1": {
                "kernel": jnp.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
                "bias": jnp.array([0.04, 0.05])
            }
        }

        # Check that the kernels match
        for layer_name, layer_dict in new_params.items():
            for param_name, param in layer_dict.items():
                if "kernel" in param_name:
                    # print(f"param: {param}")
                    # print(f"expected: {expected_new_params[layer_name][param_name]}")
                    self.assertTrue(jnp.array_equal(
                        param,
                        expected_new_params[layer_name][param_name]
                    ), f"Kernel parameters for {layer_name} do not match expected values")
                
        # Check that the masks match
        for layer_name, layer_masks in state.masks.items():
            for param_name, mask in layer_masks.items():
                if "kernel" in param_name:
                    self.assertTrue(jnp.array_equal(
                        mask,
                        expected_mask[layer_name][param_name]
                    ), f"Mask for {layer_name} {param_name} does not match expected values")

    def test_get_mask(self):
        """Test that get_mask retrieves the correct mask for a given layer."""
        # create a mask tree with known values
        mask_tree = {
            "Dense_0": {
                "bias": jnp.array([[False, True, True], [False, False, False]]),
                "kernel": jnp.array([[[True, True, False], [False, True, True]],
                                    [[True, False, False], [False, False, True]]])
            },
            "Dense_1": {
                "bias": jnp.array([[False, True], [False, False]]),
                "kernel": jnp.array([[[True, False], [True, True], [False, True]],
                                        [[True, False], [False, True], [True, False]]])
            }
        }

        mask_task_0 = self.packnet.get_mask(mask_tree, 0)
        expected_mask_task_0 = {
            "Dense_0": {
                "bias": jnp.array([False, True, True]),
                "kernel": jnp.array([[True, True, False], [False, True, True]])
            },
            "Dense_1": {
                "bias": jnp.array([False, True]),
                "kernel": jnp.array([[True, False], [True, True], [False, True]])
            }
        }
        self.assertTrue(jnp.array_equal(
            mask_task_0["Dense_0"]["kernel"],
            expected_mask_task_0["Dense_0"]["kernel"]
        ), "Dense_0 kernel masks do not match")
        self.assertTrue(jnp.array_equal(
            mask_task_0["Dense_1"]["kernel"],
            expected_mask_task_0["Dense_1"]["kernel"]
        ), "Dense_1 kernel masks do not match")
        # Check that the bias mask is correctly retrieved
        self.assertTrue(jnp.array_equal(
            mask_task_0["Dense_0"]["bias"],
            expected_mask_task_0["Dense_0"]["bias"]
        ), "Dense_0 bias masks do not match")
        self.assertTrue(jnp.array_equal(
            mask_task_0["Dense_1"]["bias"],
            expected_mask_task_0["Dense_1"]["bias"]
        ), "Dense_1 bias masks do not match")
        # Test for task 1
        mask_task_1 = self.packnet.get_mask(mask_tree, 1)
        expected_mask_task_1 = {
            "Dense_0": {
                "bias": jnp.array([False, False, False]),
                "kernel": jnp.array([[True, False, False], [False, False, True]])
            },
            "Dense_1": {
                "bias": jnp.array([False, False]),
                "kernel": jnp.array([[True, False], [False, True], [True, False]])
            }
        }
        self.assertTrue(jnp.array_equal(
            mask_task_1["Dense_0"]["kernel"],
            expected_mask_task_1["Dense_0"]["kernel"]
        ), "Dense_0 kernel masks do not match")
        self.assertTrue(jnp.array_equal(
            mask_task_1["Dense_1"]["kernel"],
            expected_mask_task_1["Dense_1"]["kernel"]
        ), "Dense_1 kernel masks do not match")
        # Check that the bias mask is correctly retrieved
        self.assertTrue(jnp.array_equal(
            mask_task_1["Dense_0"]["bias"],
            expected_mask_task_1["Dense_0"]["bias"]
        ), "Dense_0 bias masks do not match")
        self.assertTrue(jnp.array_equal(
            mask_task_1["Dense_1"]["bias"],
            expected_mask_task_1["Dense_1"]["bias"]
        ), "Dense_1 bias masks do not match")
        
    def test_mask_combination(self):
        """Test that masks are correctly combined across tasks."""
        
        # Create a mask tree that mirrors the parameters
        mask_tree = {
            "Dense_0": {
                "bias": jnp.array([[False, True, True], [False, False, False]]),
                "kernel": jnp.array([[[True, True, False], [False, True, True]], 
                                     [[True, False, False], [False, False, True]]])
            },
            "Dense_1": {
                "bias": jnp.array([[False, True], [False, False]]),
                "kernel": jnp.array([[[True, False], [True, True], [False, True]], 
                                     [[True, False], [False, True], [True, False]]])
            }
        }



        # Combine the masks
        combined_mask = self.packnet.combine_masks(mask_tree, 2)
        expected = {
            "Dense_0": {
                "bias": jnp.array([False, True, True]),
                "kernel": jnp.array([[True, True, False], [False, True, True]])
            },
            "Dense_1": {
                "bias": jnp.array([False, True]),
                "kernel": jnp.array([[True, False], [True, True], [True, True]])
            }
        }
        
        self.assertTrue(jnp.array_equal(
            combined_mask["Dense_0"]["kernel"],
            expected["Dense_0"]["kernel"]
        ), "Dense_0 kernel masks do not match")

        self.assertTrue(jnp.array_equal(
            combined_mask["Dense_1"]["kernel"],
            expected["Dense_1"]["kernel"]
        ), "Dense_1 kernel masks do not match")

        # Create a new packnet instance with more tasks
        packnet_more_tasks = Packnet(
            seq_length=3,
            prune_instructions=0.5,  # Prune 50% of weights
            train_finetune_split=(100, 50),
            prunable_layers=[nn.Dense]
        )


        mask_tree_more_tasks = {
            "Dense_0": {
                "bias": jnp.array([
                    [False, True, True], 
                    [False, False, False], 
                    [True, True, True]
                ]),
                "kernel": jnp.array([
                    [[True, True, False], [False, True, True]], 
                    [[True, False, False], [False, False, True]],
                    [[True, True, True], [True, True, True]]
                ])
            },
            "Dense_1": {
                "bias": jnp.array([
                    [False, True], 
                    [False, False], 
                    [True, False]
                ]),
                "kernel": jnp.array([
                    [[True, False], [True, True], [False, True]], 
                    [[True, False], [False, True], [False, False]],
                    [[True, True], [True, False], [True, False]]
                    ])
            }
        }
        # Combine the masks
        combined_mask_more_tasks = packnet_more_tasks.combine_masks(mask_tree_more_tasks, 2)
        expected_more_tasks = {
            "Dense_0": {
                "bias": jnp.array([False, True, True]),
                "kernel": jnp.array([[True, True, False], [False, True, True]])
            },
            "Dense_1": {
                "bias": jnp.array([False, True]),
                "kernel": jnp.array([[True, False], [True, True], [False, True]])
            }
        }
        self.assertTrue(jnp.array_equal(
            combined_mask_more_tasks["Dense_0"]["kernel"],
            expected_more_tasks["Dense_0"]["kernel"]
        ), "Dense_0 kernel masks do not match")
        self.assertTrue(jnp.array_equal(
            combined_mask_more_tasks["Dense_1"]["kernel"],
            expected_more_tasks["Dense_1"]["kernel"]
        ), "Dense_1 kernel masks do not match")

    def test_bias_fixing(self):
        """Test that bias fixing correctly zeros gradients."""

        self.packnet_state = self.packnet_state.replace(
            current_task=0,
            train_mode=True
        )

        # Create test gradients
        test_grads = {
            "Dense_0": {
                "kernel": jnp.ones((self.input_size, self.hidden_size)),
                "bias": jnp.ones(self.hidden_size)
            },
            "Dense_1": {
                "kernel": jnp.ones((self.hidden_size, self.output_size)),
                "bias": jnp.ones(self.output_size)
            }
        }
        
        # Apply fix_biases
        fixed_grads = self.packnet.fix_biases(test_grads, self.packnet_state)
        
        # Check that bias gradients are zeroed while kernel gradients are preserved
        for layer_name, layer_dict in fixed_grads.items():
            if self.packnet.layer_is_prunable(layer_name):
                self.assertTrue(jnp.all(layer_dict["bias"] == 0),
                              f"Bias gradients for {layer_name} should be zeroed")
                self.assertTrue(jnp.all(layer_dict["kernel"] == 1),
                              f"Kernel gradients for {layer_name} should be preserved")

    def test_train_mask(self):
        """Test that training mask correctly protects previous task parameters."""
        # Set up a scenario with one completed task
        self.packnet_state = self.packnet_state.replace(current_task=1)
        
        # Create task 0 mask with some parameters assigned to task 0
        mask_tree = {
            "Dense_0": {
                "bias": jnp.array([[False, True, True], [False, False, False]]),
                "kernel": jnp.array([[[True, True, False], [False, True, True]], 
                                     [[False, False, False], [False, False, False]]])
            },
            "Dense_1": {
                "bias": jnp.array([[False, True], [False, False]]),
                "kernel": jnp.array([[[True, False], [True, True], [False, True]], 
                                     [[False, False], [False, False], [False, False]]])
            }
        }
        self.packnet_state = self.packnet_state.replace(masks=mask_tree)
        
        # Create test gradients that match the mask structure
        test_grads = {
            "Dense_0": {
                "kernel": jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                "bias": jnp.array([0.01, 0.02, 0.03])
            },
            "Dense_1": {
                "kernel": jnp.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
                "bias": jnp.array([0.04, 0.05])
            }
        }

        # Apply the training mask
        masked_grads = self.packnet.train_mask(test_grads, self.packnet_state)

        # Check that gradients for task 0 are zeroed (parameters cannot be updated)
        expected_masked_grads = {
            "Dense_0": {
                "kernel": jnp.array([[0.0, 0.0, 0.3], [0.4, 0.0, 0.0]]),
                "bias": jnp.array([0.01, 0.02, 0.03])
            },
            "Dense_1": {
                "kernel": jnp.array([[0.0, 0.8], [0.0, 0.0], [1.1, 0.0]]),
                "bias": jnp.array([0.04, 0.05])
            }
        }

        for layer_name, layer_dict in masked_grads.items():
            for param_name, param in layer_dict.items():
                if "kernel" in param_name:
                    # Check that masked gradients are zeroed
                    self.assertTrue(jnp.all(param == expected_masked_grads[layer_name][param_name]),
                                  f"Masked gradients for {layer_name} {param_name} do not match expected values")
                else:
                    # Bias gradients should remain unchanged
                    self.assertTrue(jnp.all(param == test_grads[layer_name][param_name]),
                                  f"Bias gradients for {layer_name} {param_name} should remain unchanged")
                    
    def test_finetune_mask(self):

        """Test that fine-tune mask correctly protects previous task parameters."""
        # Set up a scenario with one completed task
        self.packnet_state = self.packnet_state.replace(current_task=1)
        
        # Create task 0 mask with some parameters assigned to task 0
        mask_tree = {
            "Dense_0": {
                "bias": jnp.array([[False, True, True], [False, False, False]]),
                "kernel": jnp.array([[[True, False, False], [False, True, False]], 
                                     [[True, False, True], [False, True, False]]])
            },
            "Dense_1": {
                "bias": jnp.array([[False, True], [False, False]]),
                "kernel": jnp.array([[[True, False], [True, False], [False, True]], 
                                     [[False, True], [True, False], [False, False]]])
            }
        }
        self.packnet_state = self.packnet_state.replace(masks=mask_tree)
        
        # Create test gradients that match the mask structure
        test_grads = {
            "Dense_0": {
                "kernel": jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                "bias": jnp.array([0.01, 0.02, 0.03])
            },
            "Dense_1": {
                "kernel": jnp.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
                "bias": jnp.array([0.04, 0.05])
            }
        }

        # Apply the fine-tune mask
        masked_grads = self.packnet.fine_tune_mask(test_grads, self.packnet_state)
        
        # True, True, False,  True, True, True
        # True, False, True, True, True, True

        # Check that gradients for task 0 are zeroed (parameters cannot be updated)
        expected_masked_grads = {
            "Dense_0": {
                "kernel": jnp.array([[0.0, 0.0, 0.3], [0.0, 0.0, 0.0]]),
                "bias": jnp.array([0.01, 0.02, 0.03])
            },
            "Dense_1":
            {
                "kernel": jnp.array([[0.0, 0.8], [0.0, 0.0], [0.0, 0.0]]),
                "bias": jnp.array([0.04, 0.05])
            }
        }

        for layer_name, layer_dict in masked_grads.items():
            for param_name, param in layer_dict.items():
                # Check if the parameter is a kernel
                if "kernel" in param_name:
                    # Check that masked gradients are equal to the expected masked grads
                    self.assertTrue(jnp.array_equal(
                        param,
                        expected_masked_grads[layer_name][param_name]
                    ), f"Masked gradients for {layer_name} {param_name} do not match expected values")
                else:
                    # Bias gradients should remain unchanged
                    self.assertTrue(jnp.array_equal(
                        param,
                        test_grads[layer_name][param_name]
                    ), f"Bias gradients for {layer_name} {param_name} should remain unchanged")

    def test_mask_remaining_params(self):
        """Test that remaining parameters are correctly masked."""

        mask_tree = {
            "Dense_0": {
                "bias": jnp.array([[False, True, True], 
                                   [False, False, False],
                                   [False, False, True]]),
                "kernel": jnp.array([[[True, True, False], [False, True, True]], 
                                     [[True, False, False], [False, False, True]],
                                     [[False, False, False], [False, False, False]]])
            },
            "Dense_1": {
                "bias": jnp.array([[False, True], 
                                   [False, False], 
                                   [False, True]]),
                "kernel": jnp.array([[[True, False], [True, True], [False, True]], 
                                     [[True, False], [False, True], [True, False]], 
                                     [[False, False], [False, False], [False, False]]])
            }
        }

        test_grads = {
            "Dense_0": {
                "bias": jnp.array([0.01, 0.02, 0.03]),
                "kernel": jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            },
            "Dense_1": {
                "bias": jnp.array([0.04, 0.05]),
                "kernel": jnp.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
            }
        }

        # Create a new packnet instance
        packnet = Packnet(
            seq_length=3,
            prune_instructions=0.5,  # Prune 50% of weights
            train_finetune_split=(100, 50),
            prunable_layers=[nn.Dense]
        )

        # Create a packnet state with our test parameters
        test_packnet_state = PacknetState(
            masks=mask_tree,
            current_task=2,
            train_mode=True
        )

        # Apply the remaining mask
        params, state = packnet.mask_remaining_params(test_grads, test_packnet_state)

        expected_mask_tree = {
            "Dense_0": {
                "bias": jnp.array([[False, True, True], 
                                   [False, False, False],
                                   [False, False, True]]),
                "kernel": jnp.array([[[True, True, False], [False, True, True]], 
                                     [[True, False, False], [False, False, True]],
                                     [[False, False, True], [True, False, False]]])
            },
            "Dense_1": {
                "bias": jnp.array([[False, True], 
                                   [False, False], 
                                   [False, True]]),
                "kernel": jnp.array([[[True, False], [True, True], [False, True]], 
                                     [[True, False], [False, True], [True, False]], 
                                     [[False, True], [False, False], [False, False]]])
            }
        }
        
        # Check that the masks match
        for layer_name, layer_masks in state.masks.items():
            for param_name, mask in layer_masks.items():
                if "kernel" in param_name:
                    self.assertTrue(jnp.array_equal(
                        mask,
                        expected_mask_tree[layer_name][param_name]
                    ), f"Mask for {layer_name} {param_name} does not match expected values")

    def test_on_backwards_end(self):
        """Tests that on_backwards_end applies the correct method for masking, given the mode"""

        # Create test gradients that match the mask structure
        test_grads = {
            "Dense_0": {
                "kernel": jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                "bias": jnp.array([0.01, 0.02, 0.03])
            },
            "Dense_1": {
                "kernel": jnp.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
                "bias": jnp.array([0.04, 0.05])
            }
        }
        
        # Create mask structure similar to previous tests
        mask_tree = {
            "Dense_0": {
                "bias": jnp.array([[False, True, True], [False, False, False]]),
                "kernel": jnp.array([[[True, True, False], [False, True, True]], 
                                     [[False, False, False], [False, False, False]]])
            },
            "Dense_1": {
                "bias": jnp.array([[False, True], [False, False]]),
                "kernel": jnp.array([[[True, False], [True, True], [False, True]], 
                                     [[False, False], [False, False], [False, False]]])
            }
        }

        # create a new state
        self.packnet_state = self.packnet_state.replace(
            masks=mask_tree,
            current_task=1,
            train_mode=True
        )

        # Since current_task is 0, the training mask should be applied and fix_gradients should not be applied
        masked_grads = self.packnet.on_backwards_end(test_grads, self.packnet_state)

        # Check that gradients for task 0 are zeroed (parameters cannot be updated)
        expected_masked_grads = {
            "Dense_0": {
                "bias": jnp.array([0.01, 0.02, 0.03]),
                "kernel": jnp.array([[0., 0., 0.3], [0.4, 0., 0.]])
            },
            "Dense_1": {
                "bias": jnp.array([0.04, 0.05]),
                "kernel": jnp.array([[0., 0.8], [0., 0.], [1.1, 0.]])
            }
        }

        for layer_name, layer_dict in masked_grads["params"].items():
            for param_name, param in layer_dict.items():
                if "kernel" in param_name:
                    # Check that masked gradients are equal to the expected masked grads
                    self.assertTrue(jnp.array_equal(
                        param,
                        expected_masked_grads[layer_name][param_name]
                    ), f"Masked gradients for {layer_name} {param_name} do not match expected values")

    def test_on_train_end(self):
        """ Tests if the on_train_end method correctly prunes the parameters and changes the train mode"""
        # Create test parameters
        test_params = {
            "Dense_0": {
                "kernel": jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                "bias": jnp.array([0.01, 0.02, 0.03])
            },
            "Dense_1": {
                "kernel": jnp.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
                "bias": jnp.array([0.04, 0.05])
            }
        }

        test_mask_tree = {
            "Dense_0": {
                "bias": jnp.array([[False, True, True], [False, False, False]]),
                "kernel": jnp.array([[[True, True, False], [False, True, True]], 
                                     [[False, False, False], [False, False, False]]])
            },
            "Dense_1": {
                "bias": jnp.array([[False, True], [False, False]]),
                "kernel": jnp.array([[[True, False], [True, True], [False, True]], 
                                     [[False, False], [False, False], [False, False]]])
            }
        }

        # Set the current task to 1 and train mode to True
        self.packnet_state = self.packnet_state.replace(
            masks=test_mask_tree,
            current_task=0,
            train_mode=True
        )

        # Call on_train_end
        new_params, new_state = self.packnet.on_train_end(test_params, self.packnet_state)

        # Check that the new state is in fine-tuning mode
        self.assertFalse(new_state.train_mode, "State should be in fine-tuning mode")

        print(f"New state: {new_state}")
        print(f"New params: {new_params}")
        print(f"New masks: {new_state.masks}")
        

        # Check that some parameters are pruned (exact number depends on implementation)
        pruned_count = jnp.sum(jnp.abs(new_params["params"]["Dense_0"]["kernel"]) == 0)
        self.assertTrue(pruned_count > 0, "Some parameters should be pruned")

    def test_sparsity_computation(self):
        """ Method that tests if the sparsity of the model is computed correctly"""

        test_params_empty = {
            "Dense_0": {
                "kernel": jnp.array([]),
                "bias": jnp.array([])
            },
            "Dense_1": {
                "kernel": jnp.array([]),
                "bias": jnp.array([])
            }
        }

        test_params_low_sparsity = {
            "Dense_0": {
                "kernel": jnp.array([[0.1, 0.2], [0.3, 0.4]]),
                "bias": jnp.array([0.01, 0.02])
            },
            "Dense_1": {
                "kernel": jnp.array([[0.0, 0.6], [0.7, 0.8]]),
                "bias": jnp.array([0.03, 0.04])
            }
        }

        test_params_high_sparsity = {
            "Dense_0": {
                "kernel": jnp.array([[0.0, 0.0], [0.3, 0.0]]),
                "bias": jnp.array([0.01, 0.02])
            },
            "Dense_1": {
                "kernel": jnp.array([[0.0, 0.6], [0.0, 0.0]]),
                "bias": jnp.array([0.03, 0.04])
            }
        }

        expected_sparsity_empty = 0.0
        expected_sparsity_low = (1/8)
        expected_sparsity_high = (6/8)

        sparsity_empty = self.packnet.compute_sparsity(test_params_empty)
        sparsity_low = self.packnet.compute_sparsity(test_params_low_sparsity)
        sparsity_high = self.packnet.compute_sparsity(test_params_high_sparsity)

        self.assertEqual(sparsity_empty, expected_sparsity_empty,
                         "Sparsity for empty parameters should be 0.0")
        self.assertEqual(sparsity_low, expected_sparsity_low,
                         "Sparsity for low sparsity parameters should be 0.125")
        self.assertEqual(sparsity_high, expected_sparsity_high,
                         "Sparsity for high sparsity parameters should be 0.75")

        

    def test_whole_training_process(self):
        """ 
        Test the whole process of training, finetuning and pruning on a sequence of 3 tasks
        """

        # 1: Create a new packnet instance
        packnet = Packnet(
            seq_length=3,
            prune_instructions=0.5,  # Prune 50% of weights
            train_finetune_split=(100, 50),
            prunable_layers=[nn.Dense]
        )

        # Create parameters
        params = {
            "Dense_0": {
                "kernel": jnp.array([[5, 8, 2, 3], [1, 5, 9, 4]]),
                "bias": jnp.array([0.1, 0.2, 0.3, 0.4])
            }
        }

        grads = {
            "Dense_0": {
                "kernel": jnp.array([[0.5, 0.8, 0.2, 0.3], [0.1, 0.5, 0.9, 0.4]]),
                "bias": jnp.array([0.01, 0.02, 0.03, 0.04])
            }
        }

        # Create a mask tree with everything false, for 3 tasks
        mask_tree = {
            "Dense_0": {
                "kernel": jnp.array([[[False, False, False, False], [False, False, False, False]],
                                     [[False, False, False, False], [False, False, False, False]], 
                                     [[False, False, False, False], [False, False, False, False]]]),
                "bias": jnp.array([[False, False, False, False],
                                    [False, False, False, False], 
                                    [False, False, False, False]])
            }

        }
        # Create a packnet state with our test parameters
        packnet_state = PacknetState(
            masks=mask_tree,
            current_task=0,
            train_mode=True
        )

        # 2: Train on task 0, 
        for i in range(2):
            new_grads = packnet.on_backwards_end(grads, packnet_state)
        
        # None of the gradients should be masked out: 
        sparsity = packnet.compute_sparsity(new_grads["params"])
        assert sparsity == 0.0, "Sparsity should be 0.0 during the first training"
        
        # check if gradients are unchanged 
        expected_grads_task0 = {
            "Dense_0": {
                "kernel": jnp.array([[0.5, 0.8, 0.2, 0.3], [0.1, 0.5, 0.9, 0.4]]),
                "bias": jnp.array([0.01, 0.02, 0.03, 0.04])
            }
        }

        # Check if the gradients are unchanged
        assert jnp.array_equal(new_grads["params"]["Dense_0"]["kernel"], 
                               expected_grads_task0["Dense_0"]["kernel"]), "kernel Gradients should be unchanged in task 0"
        assert jnp.array_equal(new_grads["params"]["Dense_0"]["bias"],
                               expected_grads_task0["Dense_0"]["bias"]), "bias Gradients should be unchanged in task 0"
        
        
        # 3. Finish training on task 0, call on_train_end
        new_params, packnet_state = packnet.on_train_end(params, packnet_state)

        sparsity = packnet.compute_sparsity(new_params["params"])
        print(sparsity)
        assert sparsity == 0.5, "Sparsity should be 0.5 after first pruning"

        # check if the training mode is false
        self.assertFalse(packnet_state.train_mode, "State should be in fine-tuning mode")

        expected_params_after_task0 = {
            "Dense_0": {
                "kernel": jnp.array([[5, 8, 0, 0], [0, 5, 9, 0]]),
                "bias": jnp.array([0.1, 0.2, 0.3, 0.4])
            }
        }

        expected_mask_after_task0 = {
            "Dense_0": {
                "kernel": jnp.array([[[True, True, False, False], [False, True, True, False]],
                                     [[False, False, False, False], [False, False, False, False]], 
                                     [[False, False, False, False], [False, False, False, False]]]),
                "bias": jnp.array([[False, False, False, False],
                                    [False, False, False, False], 
                                    [False, False, False, False]])
            }

        }

        # check if the parameters are pruned correctly
        assert jnp.array_equal(new_params["params"]["Dense_0"]["kernel"],
                               expected_params_after_task0["Dense_0"]["kernel"]), "kernel parameters should be pruned in task 0"
        assert jnp.array_equal(new_params["params"]["Dense_0"]["bias"],
                               expected_params_after_task0["Dense_0"]["bias"]), "bias parameters should not be pruned in task 0"
        # check if the masks are correct
        assert jnp.array_equal(packnet_state.masks["Dense_0"]["kernel"],
                               expected_mask_after_task0["Dense_0"]["kernel"]), "kernel masks should be correct in task 0"
        assert jnp.array_equal(packnet_state.masks["Dense_0"]["bias"],
                               expected_mask_after_task0["Dense_0"]["bias"]), "bias masks should be correct in task 0"
    

        # finetune on task 0: 
        for i in range(2):
            # new gradients are calculated: 
            grads_finetune = {
                "Dense_0": {
                    "kernel": jnp.array([[0.2, 0.9, 0.1, 0.3], [0.4, 0.4, 0.7, 0.4]]),
                    "bias": jnp.array([0.03, 0.02, 0.07, 0.03])
                }
            }
            new_grads_finetune_task0 = packnet.on_backwards_end(grads_finetune, packnet_state)
        
        sparsity_grads = packnet.compute_sparsity(new_grads_finetune_task0["params"])
        assert sparsity_grads == 0.5, "Sparsity of gradients should be 0.5 during finetuning task 0"
        
        expected_grads_finetune_task0 = {
                "Dense_0": {
                    "kernel": jnp.array([[0.2, 0.9, 0.0, 0.0], [0.0, 0.4, 0.7, 0.0]]),
                    "bias": jnp.array([0.03, 0.02, 0.07, 0.03])
                }
            }
        
        # print("new grads finetune task 0", new_grads_finetune_task0)
        # Check if the gradients are unchanged
        assert jnp.array_equal(new_grads_finetune_task0["params"]["Dense_0"]["kernel"], 
                               expected_grads_finetune_task0["Dense_0"]["kernel"]), "kernel Gradients should be unchanged in task 0"
        assert jnp.array_equal(new_grads_finetune_task0["params"]["Dense_0"]["bias"],
                               expected_grads_finetune_task0["Dense_0"]["bias"]), "bias Gradients should be unchanged in task 0"

        
        # on finetune end:
        packnet_state = packnet.on_finetune_end(packnet_state)

        self.assertTrue(packnet_state.train_mode, "State should be in training mode")
        self.assertTrue(packnet_state.current_task == 1, "State should be in task 1")

        ###### TASK 1 ########
        print("#################################################################")
        # expected_mask = {
        #     "Dense_0": {
        #         "kernel": jnp.array([[[False, False, False], [True, True, True]], 
        #                              [[False, False, False], [False, False, False]], 
        #                              [[False, False, False], [False, False, False]]]),
        #         "bias": jnp.array([[False, False, False], 
        #                            [False, False, False], 
        #                            [False, False, False]]),
        #     }
        # }

        # packnet_state = packnet_state.replace(masks=expected_mask)


        # train the model on task 1
        for i in range(2):
            grads_task1 = {
                "Dense_0": {
                    "kernel": jnp.array([[0.5, 0.9, 0.7, 0.2], [0.4, 0.5, 0.6, 0.1]]),
                    "bias": jnp.array([0.1, 0.2, 0.3, 0.8])
                }
            }
            new_grads_task1 = packnet.on_backwards_end(grads_task1, packnet_state)


            expected_grads_task1 = {
                "Dense_0": {
                    "kernel": jnp.array([[0.0, 0.0, 0.7, 0.2], [0.4, 0.0, 0.0, 0.1]]),
                    "bias": jnp.array([0, 0, 0, 0])
                }
            }
            
            sparsity_grads = packnet.compute_sparsity(new_grads_task1["params"])
            assert sparsity_grads == 0.5, "Sparsity of gradients should be 0.5 during training task 1"

            # check if the bias parameters are masked out
            assert jnp.array_equal(new_grads_task1["params"]["Dense_0"]["bias"], 
                                   expected_grads_task1["Dense_0"]["bias"]), "bias Gradients should be masked out in task 1"
            assert jnp.array_equal(new_grads_task1["params"]["Dense_0"]["kernel"],
                                   expected_grads_task1["Dense_0"]["kernel"]), "fixed parameters of task 1 should be masked out"
            
        # after training, the pruned parameters are filled again:
        refilled_params = {
            "Dense_0": {
                "kernel": jnp.array([[5, 8, 3, 4], [2, 5, 9, 7]]),
                "bias": jnp.array([0.1, 0.2, 0.3, 0.4])
            }
        }
            
        # finish  training on task 1
        new_params_task1, packnet_state = packnet.on_train_end(refilled_params, packnet_state)
        sparsity = packnet.compute_sparsity(new_params_task1["params"])
        assert sparsity == 0.25, "Sparsity should be 0.25 after second pruning"

        expected_params_after_task1 = {
            "Dense_0": {
                "kernel": jnp.array([[5, 8, 0, 4], [0, 5, 9, 7]]),
                "bias": jnp.array([0.1, 0.2, 0.3, 0.4])
            }
        }

        expected_mask_after_task1 = {
            "Dense_0": {
                "kernel": jnp.array([[[True, True, False, False], [False, True, True, False]],
                                     [[False, False, False, True], [False, False, False, True]], 
                                     [[False, False, False, False], [False, False, False, False]]]),
                "bias": jnp.array([[False, False, False, False],
                                    [False, False, False, False], 
                                    [False, False, False, False]])
            }

        }

        assert jnp.array_equal(new_params_task1["params"]["Dense_0"]["kernel"],
                        expected_params_after_task1["Dense_0"]["kernel"]), "pruning is not correct in task 1"
        assert jnp.array_equal(new_params_task1["params"]["Dense_0"]["bias"],
                               expected_params_after_task1["Dense_0"]["bias"]), "bias parameters should not be pruned in task 1"
        assert jnp.array_equal(packnet_state.masks["Dense_0"]["kernel"],
                                expected_mask_after_task1["Dense_0"]["kernel"]), "kernel masks should be correct in task 1"
        assert jnp.array_equal(packnet_state.masks["Dense_0"]["bias"],
                                expected_mask_after_task1["Dense_0"]["bias"]), "bias masks should be correct in task 1"
        
        # finetune on task 1:
        for i in range(1):
            # new gradients are calculated: 
            grads_finetune = {
                "Dense_0": {
                    "kernel": jnp.array([[0.8, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
                    "bias": jnp.array([0.04, 0.02, 0.056, 0.02])
                }
            }
            new_grads_finetune_task1 = packnet.on_backwards_end(grads_finetune, packnet_state)

            sparsity_grads = packnet.compute_sparsity(new_grads_finetune_task1["params"])
            assert sparsity_grads == 0.75, "Sparsity of gradients should be 0.75 during finetuning task 1"
        
        expected_grads_finetune_task1 = {
                "Dense_0": {
                    "kernel": jnp.array([[0.0, 0.0, 0.0, 0.1], [0.0, 0.0, 0.0, 0.1]]),
                    "bias": jnp.array([0, 0, 0, 0])
                }
            }

        # Check if the gradients are correctly masked
        assert jnp.array_equal(new_grads_finetune_task1["params"]["Dense_0"]["kernel"], 
                               expected_grads_finetune_task1["Dense_0"]["kernel"]), "kernel gradients not masked correctly task 1"
        assert jnp.array_equal(new_grads_finetune_task1["params"]["Dense_0"]["bias"],
                               expected_grads_finetune_task1["Dense_0"]["bias"]), "bias gradients should be zeroed after finetuning task 1"

        
        # on finetune end:
        packnet_state = packnet.on_finetune_end(packnet_state)

        self.assertTrue(packnet_state.train_mode, "State should be in training mode")
        self.assertTrue(packnet_state.current_task == 2, "State should be in task 1")



        # ###### TASK 2 ########
        # print("#################################################################")
        # expected_mask = {
        #     "Dense_0": {
        #         "kernel": jnp.array([[[False, False, False], [True, True, True]], 
        #                              [[True, False, True], [False, False, False]], 
        #                              [[False, False, False], [False, False, False]]]),
        #         "bias": jnp.array([[False, False, False], 
        #                            [False, False, False], 
        #                            [False, False, False]]),
        #     }
        # }

        # packnet_state = packnet_state.replace(masks=expected_mask)

        # # train the model on task 2
        # for i in range(5):
        #     grads_task2 = {
        #         "Dense_0": {
        #             "kernel": jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        #             "bias": jnp.array([0.1, 0.2, 0.3])
        #         }
        #     }
        #     new_grads_task2 = packnet.on_backwards_end(grads_task2, packnet_state)

        #     expected_grads_task2 = {
        #         "Dense_0": {
        #             "kernel": jnp.array([[0, 0.2, 0], [0, 0, 0]]),
        #             "bias": jnp.array([0, 0, 0])
        #         }
        #     }

        #     # check if the bias parameters are masked out
        #     assert jnp.array_equal(new_grads_task2["params"]["Dense_0"]["bias"], 
        #                            expected_grads_task2["Dense_0"]["bias"]), "bias Gradients should be masked out in task 2"
        #     assert jnp.array_equal(new_grads_task2["params"]["Dense_0"]["kernel"],
        #                            expected_grads_task2["Dense_0"]["kernel"]), "fixed parameters of task 2 should be masked out"
            
        # # after training, the pruned parameters are filled again:
        # refilled_params2 = {
        #     "Dense_0": {
        #         "kernel": jnp.array([[6, 5, 8], [4, 5, 6]]),
        #         "bias": jnp.array([0.1, 0.2, 0.3])
        #     }
        # }
            
        # # finish  training on task 1
        # new_params_task2, packnet_state = packnet.on_train_end(refilled_params2, packnet_state)

        # # THIS SHOULD CALL MASK_REMAINING_PARAMS, NOT PRUNE

        # expected_params_after_task2 = {
        #     "Dense_0": {
        #         "kernel": jnp.array([[6, 5, 8], [4, 5, 6]]),
        #         "bias": jnp.array([0.1, 0.2, 0.3])
        #     }
        # }

        # expected_mask_after_task2 = {
        #     "Dense_0": {
        #         "kernel": jnp.array([[[False, False, False], [True, True, True]], 
        #                              [[True, False, True], [False, False, False]], 
        #                              [[False, True, False], [False, False, False]]]),
        #         "bias": jnp.array([[False, False, False], 
        #                            [False, False, False], 
        #                            [False, False, False]]),
        #     }
        # }

        # assert jnp.array_equal(new_params_task2["params"]["Dense_0"]["kernel"],
        #                 expected_params_after_task2["Dense_0"]["kernel"]), "pruning is not correct in task 2"
        # assert jnp.array_equal(new_params_task2["params"]["Dense_0"]["bias"],
        #                        expected_params_after_task2["Dense_0"]["bias"]), "bias parameters should not be pruned in task 2"
        # assert jnp.array_equal(packnet_state.masks["Dense_0"]["kernel"],
        #                         expected_mask_after_task2["Dense_0"]["kernel"]), "kernel masks should be correct in task 2"
        # assert jnp.array_equal(packnet_state.masks["Dense_0"]["bias"],
        #                         expected_mask_after_task2["Dense_0"]["bias"]), "bias masks should be correct in task 2"
        
        # # finetune on task 2:
        # for i in range(5):
        #     # new gradients are calculated: 
        #     grads_finetune = {
        #         "Dense_0": {
        #             "kernel": jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        #             "bias": jnp.array([0.1, 0.2, 0.3])
        #         }
        #     }
        #     new_grads_finetune_task2 = packnet.on_backwards_end(grads_finetune, packnet_state)
        
        # expected_grads_finetune_task2 = {
        #     "Dense_0": {
        #         "kernel": jnp.array([[0, 0.2, 0], [0, 0, 0]]),
        #         "bias": jnp.array([0, 0, 0])
        #     }
        # }
        # # Check if the gradients are correctly masked
        # assert jnp.array_equal(new_grads_finetune_task2["params"]["Dense_0"]["kernel"], 
        #                        expected_grads_finetune_task2["Dense_0"]["kernel"]), "kernel gradients not masked correctly task 2"
        # assert jnp.array_equal(new_grads_finetune_task2["params"]["Dense_0"]["bias"],
        #                        expected_grads_finetune_task2["Dense_0"]["bias"]), "bias gradients should be zeroed after finetuning task 2"
        
        # # on finetune end:
        # packnet_state = packnet.on_finetune_end(packnet_state)


if __name__ == "__main__":
    unittest.main()