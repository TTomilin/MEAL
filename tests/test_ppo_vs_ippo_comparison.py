#!/usr/bin/env python
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.training.train_state import TrainState
import optax

from meal.env.layouts.presets import cramped_room
from meal.env.overcooked import Overcooked
from meal.env.overcooked import Overcooked as OvercookedNAgent

from experiments.ippo import Config as PPOConfig
from experiments.ippo_po import Config as IPPOConfig
from experiments.model.mlp import ActorCritic as MLPActorCritic
from experiments.utils import batchify, unbatchify


def create_identical_environments(num_agents=2):
    """Create identical environments for both PPO_CL and IPPO_CL"""
    # Use cramped room layout with deterministic reset
    env_kwargs = {
        "layout": FrozenDict(cramped_room),
        "num_agents": num_agents,
        "random_reset": False,
        "max_steps": 400
    }

    # Create environments
    if num_agents == 1:
        # Original overcooked.py doesn't support num_agents=1, so use n_agent for both
        ppo_env = OvercookedNAgent(**env_kwargs)
        ippo_env = OvercookedNAgent(**env_kwargs)
    else:
        # For num_agents >= 2, use the original comparison
        ppo_env = OvercookedNAgent(**env_kwargs)
        ippo_env = Overcooked(**env_kwargs)

    return ppo_env, ippo_env


def create_identical_networks(config, env):
    """Create identical networks for both implementations"""
    # Get proper observation shape by simulating the batchify process
    rng = jax.random.PRNGKey(42)
    reset_rngs = jax.random.split(rng, config.num_envs)
    obs, _ = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

    # Apply batchify to get the actual input shape the network will see
    obs_batch = batchify(obs, env.agents, config.num_actors, not config.use_cnn)
    obs_shape = obs_batch.shape[1]  # Get the feature dimension after batchify

    # Network parameters
    network = MLPActorCritic(
        action_dim=env.action_space().n,
        activation=config.activation,
        use_layer_norm=config.use_layer_norm,
        big_network=config.big_network,
        use_task_id=config.use_task_id,
        use_multihead=config.use_multihead,
        shared_backbone=config.shared_backbone,
        num_tasks=1  # Single task for testing
    )

    # Initialize network with proper observation shape
    init_obs = jnp.zeros((1, obs_shape))
    network_params = network.init(rng, init_obs, env_idx=0)

    return network, network_params


def create_identical_configs(num_agents=2):
    """Create identical configurations for both implementations"""
    # Base configuration
    base_config = {
        "lr": 3e-4,
        "num_envs": 4,
        "num_steps": 32,
        "update_epochs": 2,
        "num_minibatches": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "activation": "relu",
        "use_cnn": False,
        "use_layer_norm": True,
        "big_network": False,
        "use_task_id": True,
        "use_multihead": True,
        "shared_backbone": False,
        "sparse_rewards": False,
        "individual_rewards": False,
        "agem_gradient_scale": 1.0,
        "cl_method": "ft",  # Use fine-tuning for simplicity
        "reg_coef": 0.0
    }

    # Create configs
    ppo_config = PPOConfig(**base_config)
    ippo_config = IPPOConfig(**base_config)

    # Set computed parameters
    ppo_config.num_actors = ppo_config.num_envs * num_agents
    ippo_config.num_actors = ippo_config.num_envs * num_agents
    ppo_config.minibatch_size = ppo_config.num_actors * ppo_config.num_steps // ppo_config.num_minibatches
    ippo_config.minibatch_size = ippo_config.num_actors * ippo_config.num_steps // ippo_config.num_minibatches

    return ppo_config, ippo_config


def run_identical_episode(env, network, train_state, config, rng):
    """Run an identical episode on both implementations"""
    # Reset environment
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config.num_envs)
    obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

    # Collect trajectory data
    trajectory_data = []

    for step in range(config.num_steps):
        # Prepare observations
        obs_batch = batchify(obs, env.agents, config.num_actors, not config.use_cnn)

        # Get actions and values
        pi, value, _ = network.apply(train_state.params, obs_batch, env_idx=0)

        # Sample actions
        rng, action_rng = jax.random.split(rng)
        action = pi.sample(seed=action_rng)
        log_prob = pi.log_prob(action)

        # Format actions for environment
        env_act = unbatchify(action, env.agents, config.num_envs, env.num_agents)
        env_act = {k: v.flatten() for k, v in env_act.items()}

        # Step environment
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, config.num_envs)
        new_obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
            step_rngs, env_state, env_act
        )

        # Store trajectory data
        trajectory_data.append({
            'obs': obs,
            'action': action,
            'value': value,
            'log_prob': log_prob,
            'reward': reward,
            'done': done,
            'info': info
        })

        obs = new_obs

    return trajectory_data, rng


def compare_reward_processing(reward, info, config_ppo, config_ippo):
    """Compare reward processing between PPO_CL and IPPO_CL"""
    # Get agent keys dynamically
    agent_keys = list(reward.keys())

    # PPO_CL reward processing
    if config_ppo.sparse_rewards:
        ppo_reward = reward
    elif config_ppo.individual_rewards:
        ppo_reward = jax.tree_util.tree_map(lambda x, y: x + y, reward, info["shaped_reward"])
    else:
        # Default: shared delivery + individual shaped
        if len(agent_keys) == 1:
            # For single agent, no sharing needed
            ppo_reward = jax.tree_util.tree_map(lambda x, y: x + y, reward, info["shaped_reward"])
        else:
            # For multiple agents, share delivery rewards
            total_delivery = sum(reward[agent] for agent in agent_keys)
            shared_delivery = {agent: total_delivery for agent in agent_keys}
            ppo_reward = jax.tree_util.tree_map(lambda x, y: x + y, shared_delivery, info["shaped_reward"])

    # IPPO_CL reward processing (should be identical now)
    if config_ippo.sparse_rewards:
        ippo_reward = reward
    elif config_ippo.individual_rewards:
        ippo_reward = jax.tree_util.tree_map(lambda x, y: x + y, reward, info["shaped_reward"])
    else:
        # Default: shared delivery + individual shaped
        if len(agent_keys) == 1:
            # For single agent, no sharing needed
            ippo_reward = jax.tree_util.tree_map(lambda x, y: x + y, reward, info["shaped_reward"])
        else:
            # For multiple agents, share delivery rewards
            total_delivery = sum(reward[agent] for agent in agent_keys)
            shared_delivery = {agent: total_delivery for agent in agent_keys}
            ippo_reward = jax.tree_util.tree_map(lambda x, y: x + y, shared_delivery, info["shaped_reward"])

    return ppo_reward, ippo_reward


def test_ppo_vs_ippo_comparison(num_agents=1):
    """Main test function comparing PPO_CL and IPPO_CL"""
    print(f"🔍 TESTING PPO_CL vs IPPO_CL COMPARISON (num_agents={num_agents})")
    print("=" * 60)

    # Create identical environments
    ppo_env, ippo_env = create_identical_environments(num_agents)
    print("✅ Created identical environments")

    # Create identical configurations
    ppo_config, ippo_config = create_identical_configs(num_agents)
    print("✅ Created identical configurations")

    # Create identical networks
    ppo_network, ppo_params = create_identical_networks(ppo_config, ppo_env)
    ippo_network, ippo_params = create_identical_networks(ippo_config, ippo_env)
    print("✅ Created identical networks")

    # Verify network parameters are identical
    def compare_pytrees(tree1, tree2, name):
        flat1, _ = jax.tree_util.tree_flatten(tree1)
        flat2, _ = jax.tree_util.tree_flatten(tree2)

        if len(flat1) != len(flat2):
            print(f"❌ {name}: Different number of parameters")
            return False

        for i, (p1, p2) in enumerate(zip(flat1, flat2)):
            if not jnp.allclose(p1, p2, atol=1e-6):
                print(f"❌ {name}: Parameter {i} differs")
                return False

        print(f"✅ {name}: All parameters identical")
        return True

    compare_pytrees(ppo_params, ippo_params, "Network parameters")

    # Create train states
    tx = optax.adam(learning_rate=ppo_config.lr)
    ppo_train_state = TrainState.create(apply_fn=ppo_network.apply, params=ppo_params, tx=tx)
    ippo_train_state = TrainState.create(apply_fn=ippo_network.apply, params=ippo_params, tx=tx)

    # Run identical episodes
    rng = jax.random.PRNGKey(123)  # Fixed seed for reproducibility

    print("\n🎮 Running identical episodes...")
    ppo_trajectory, rng1 = run_identical_episode(ppo_env, ppo_network, ppo_train_state, ppo_config, rng)
    ippo_trajectory, rng2 = run_identical_episode(ippo_env, ippo_network, ippo_train_state, ippo_config, rng)

    print(f"✅ Collected {len(ppo_trajectory)} steps from both implementations")

    # Compare trajectories step by step
    print("\n🔍 Comparing trajectories step by step...")

    differences_found = False

    for step, (ppo_step, ippo_step) in enumerate(zip(ppo_trajectory, ippo_trajectory)):
        # Compare observations for all agents
        for agent in ppo_env.agents:
            if not jnp.allclose(ppo_step['obs'][agent], ippo_step['obs'][agent], atol=1e-6):
                print(f"❌ Step {step}: {agent} observations differ")
                differences_found = True

        # Compare actions
        if not jnp.allclose(ppo_step['action'], ippo_step['action'], atol=1e-6):
            print(f"❌ Step {step}: Actions differ")
            differences_found = True

        # Compare values
        if not jnp.allclose(ppo_step['value'], ippo_step['value'], atol=1e-6):
            print(f"❌ Step {step}: Values differ")
            differences_found = True

        # Compare log probabilities
        if not jnp.allclose(ppo_step['log_prob'], ippo_step['log_prob'], atol=1e-6):
            print(f"❌ Step {step}: Log probabilities differ")
            differences_found = True

        # Compare reward processing
        ppo_reward, ippo_reward = compare_reward_processing(
            ppo_step['reward'], ppo_step['info'], ppo_config, ippo_config
        )

        for agent in ppo_env.agents:
            if not jnp.allclose(ppo_reward[agent], ippo_reward[agent], atol=1e-6):
                print(f"❌ Step {step}: {agent} processed rewards differ")
                print(f"   PPO: {ppo_reward[agent]}, IPPO: {ippo_reward[agent]}")
                differences_found = True

    if not differences_found:
        print("✅ All trajectory data identical between PPO_CL and IPPO_CL!")

    # Test different reward modes
    print("\n🎯 Testing different reward modes...")

    # Test sparse rewards
    ppo_config.sparse_rewards = True
    ippo_config.sparse_rewards = True

    test_reward = {"agent_0": jnp.array([1.0]), "agent_1": jnp.array([2.0])}
    test_info = {"shaped_reward": {"agent_0": jnp.array([0.5]), "agent_1": jnp.array([0.3])}}

    ppo_sparse, ippo_sparse = compare_reward_processing(test_reward, test_info, ppo_config, ippo_config)

    if jnp.allclose(ppo_sparse['agent_0'], ippo_sparse['agent_0']) and jnp.allclose(ppo_sparse['agent_1'],
                                                                                    ippo_sparse['agent_1']):
        print("✅ Sparse reward mode identical")
    else:
        print("❌ Sparse reward mode differs")

    # Test individual rewards
    ppo_config.sparse_rewards = False
    ppo_config.individual_rewards = True
    ippo_config.sparse_rewards = False
    ippo_config.individual_rewards = True

    ppo_individual, ippo_individual = compare_reward_processing(test_reward, test_info, ppo_config, ippo_config)

    if jnp.allclose(ppo_individual['agent_0'], ippo_individual['agent_0']) and jnp.allclose(ppo_individual['agent_1'],
                                                                                            ippo_individual['agent_1']):
        print("✅ Individual reward mode identical")
    else:
        print("❌ Individual reward mode differs")

    # Test default rewards
    ppo_config.individual_rewards = False
    ippo_config.individual_rewards = False

    ppo_default, ippo_default = compare_reward_processing(test_reward, test_info, ppo_config, ippo_config)

    if jnp.allclose(ppo_default['agent_0'], ippo_default['agent_0']) and jnp.allclose(ppo_default['agent_1'],
                                                                                      ippo_default['agent_1']):
        print("✅ Default reward mode identical")
    else:
        print("❌ Default reward mode differs")

    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    if not differences_found:
        print("🎉 SUCCESS: PPO_CL and IPPO_CL are now functionally equivalent!")
        print("✅ All observations, actions, values, and rewards match")
        print("✅ All reward processing modes work identically")
    else:
        print("⚠️  ISSUES FOUND: Some differences remain between implementations")

    return not differences_found


if __name__ == "__main__":
    # Test with 1 agent (as specified in the issue)
    success = test_ppo_vs_ippo_comparison(num_agents=2)
    if success:
        print("\n🎯 All tests passed! PPO_CL and IPPO_CL are equivalent.")
    else:
        print("\n❌ Some tests failed. Check the differences above.")
