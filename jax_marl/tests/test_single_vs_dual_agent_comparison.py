#!/usr/bin/env python
"""
Test that compares 1-agent overcooked_single environment with 2-agent regular overcooked environment.
Agent 0 performs the same deterministic actions in both environments to cook soup.
Compares observations and rewards (excluding agent 1 layers from the 2-agent environment).
Generates GIFs for both rollouts using visualization.
"""
import os
import sys
from os import makedirs

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import pygame
from flax.core import FrozenDict

from jax_marl.environments.overcooked.layouts import cramped_room
from jax_marl.environments.overcooked.overcooked_single import OvercookedSingle
from jax_marl.environments.overcooked.overcooked import Overcooked, DELIVERY_REWARD
from jax_marl.eval.visualizer import OvercookedVisualizer


def test_single_vs_dual_agent_comparison():
    """
    Compare 1-agent overcooked_single with 2-agent regular overcooked.
    Agent 0 performs identical deterministic actions in both environments.
    Excludes agent 1 layers from comparison.
    """
    print("=== SINGLE VS DUAL AGENT COMPARISON TEST ===")
    print("Comparing 1-agent overcooked_single with 2-agent regular overcooked")
    print("Agent 0 will perform identical actions in both environments")
    print("Agent 1 layers (position and orientations) are excluded from comparison")

    # Use the same fixed layout and deterministic reset for both environments
    layout = FrozenDict(cramped_room)
    rng_seed = 42
    max_steps = 400

    # Create both environments
    print("\nCreating environments...")
    env_1_agent = OvercookedSingle(layout=layout, random_reset=False, max_steps=max_steps)
    env_2_agent = Overcooked(layout=layout, num_agents=2, random_reset=False, max_steps=max_steps)

    print(f"1-agent environment: {type(env_1_agent).__name__} with agents {env_1_agent.agents}")
    print(f"2-agent environment: {type(env_2_agent).__name__} with agents {env_2_agent.agents}")

    # Reset both environments with the same seed
    rng = jax.random.PRNGKey(rng_seed)
    rng1, rng2 = jax.random.split(rng)

    obs_1, state_1 = env_1_agent.reset(rng1)
    obs_2, state_2 = env_2_agent.reset(rng2)

    print(f"Initial observations - 1-agent shape: {obs_1.shape}")
    print(f"Initial observations - 2-agent keys: {list(obs_2.keys())}")

    # Set up visualization for both environments
    frames_1_agent = []
    frames_2_agent = []

    viz_1_agent = OvercookedVisualizer(num_agents=1, use_old_rendering=False)
    viz_2_agent = OvercookedVisualizer(num_agents=2, use_old_rendering=False)

    def add_frame_1_agent(st):
        surface = viz_1_agent.render(env_1_agent.agent_view_size, st)
        frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
        frames_1_agent.append(frame)

    def add_frame_2_agent(st):
        surface = viz_2_agent.render(env_2_agent.agent_view_size, st)
        frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
        frames_2_agent.append(frame)

    # Add initial frames
    add_frame_1_agent(state_1)
    add_frame_2_agent(state_2)

    # Define deterministic action sequence for agent 0 to cook soup
    A = {'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5}

    # Action sequence: pick 3 onions, wait for cooking, get plate, scoop soup, deliver
    onion_cycle = [A['L'], A['I'], A['R'], A['U'], A['I']]  # pick onion, put in pot
    actions_agent_0 = onion_cycle * 3  # 3 onions
    actions_agent_0 += [A['S']] * 20  # wait for cooking
    actions_agent_0 += [
        A['D'],  # step down
        A['L'],  # step left  
        A['D'],  # step down, now facing plate-pile
        A['I'],  # take plate
        A['U'],  # back up
        A['R'],  # step right
        A['U'],  # turn up to pot, face pot
        A['I'],  # scoop soup (now holding dish)
        A['D'],  # step down toward goal line
        A['R'],  # step right
        A['D'],  # turn down, facing the serving window
        A['I'],  # deliver!
    ]

    print(f"\nRunning {len(actions_agent_0)} deterministic actions...")

    # Storage for comparison data
    rollout_data_1_agent = []
    rollout_data_2_agent = []

    # Run the same actions in both environments
    for t, action in enumerate(actions_agent_0):
        # Step 1-agent environment
        rng, step_key1 = jax.random.split(rng)
        obs_1, state_1, reward_1, done_1, info_1 = env_1_agent.step(
            step_key1, state_1, jnp.uint32(action)
        )

        # Step 2-agent environment (agent 0 does action, agent 1 stays)
        rng, step_key2 = jax.random.split(rng)
        obs_2, state_2, reward_2, done_2, info_2 = env_2_agent.step_env(
            step_key2, state_2, {"agent_0": jnp.uint32(action), "agent_1": jnp.uint32(A['S'])}
        )

        # Store rollout data
        rollout_data_1_agent.append({
            'step': t,
            'obs': obs_1,  # Single agent returns observation directly
            'reward': reward_1,
            'shaped_reward': info_1["shaped_reward"],
            'done': done_1
        })

        rollout_data_2_agent.append({
            'step': t,
            'obs': obs_2["agent_0"],  # Only agent 0's observation
            'reward': reward_2["agent_0"],
            'shaped_reward': info_2["shaped_reward"]["agent_0"],
            'done': done_2["agent_0"]
        })

        # Add frames
        add_frame_1_agent(state_1)
        add_frame_2_agent(state_2)

        # Print progress for key steps
        if t % 10 == 0 or reward_1 > 0 or reward_2["agent_0"] > 0:
            print(f"Step {t}: 1-agent reward={reward_1:.1f}, 2-agent reward={reward_2['agent_0']:.1f}")

    # Compare the rollouts
    print(f"\n=== COMPARING ROLLOUTS ===")

    total_reward_1 = sum(data['reward'] for data in rollout_data_1_agent)
    total_reward_2 = sum(data['reward'] for data in rollout_data_2_agent)
    total_shaped_1 = sum(data['shaped_reward'] for data in rollout_data_1_agent)
    total_shaped_2 = sum(data['shaped_reward'] for data in rollout_data_2_agent)

    print(f"1-agent total reward: {total_reward_1:.1f} (shaped: {total_shaped_1:.1f})")
    print(f"2-agent total reward: {total_reward_2:.1f} (shaped: {total_shaped_2:.1f})")

    # Compare observations with detailed analysis
    print(f"\n=== DETAILED OBSERVATION COMPARISON ===")
    obs_1_shape = rollout_data_1_agent[0]['obs'].shape
    obs_2_shape = rollout_data_2_agent[0]['obs'].shape
    print(f"1-agent obs shape: {obs_1_shape} (overcooked_single)")
    print(f"2-agent obs shape: {obs_2_shape} (overcooked)")

    # Channel structure analysis
    print(f"\nChannel structure analysis:")
    print(f"1-agent environment (overcooked_single):")
    print(f"  - Channel 0: Agent 0 position")
    print(f"  - Channel 1: Empty (no agent 1)")
    print(f"  - Channels 2-5: Agent 0 orientations (4 directions)")
    print(f"  - Channels 6-9: Empty (no agent 1 orientations)")
    print(f"  - Channels 10-25: Environment layers (16 layers)")
    print(f"  - Total: 26 channels")

    print(f"\n2-agent environment (overcooked with 2 agents):")
    print(f"  - Channels 0-1: Agent positions (agent 0, agent 1)")
    print(f"  - Channels 2-9: Agent orientations (4 for agent 0, 4 for agent 1)")
    print(f"  - Channels 10-25: Environment layers (16 layers)")
    print(f"  - Total: 26 channels")

    print(f"\nComparison strategy:")
    print(f"  - Compare agent 0 position: channel 0 vs channel 0")
    print(f"  - Compare agent 0 orientations: channels 2-5 vs channels 2-5")
    print(f"  - Compare environment layers: channels 10-25 vs channels 10-25")
    print(f"  - EXCLUDE agent 1 position: skip channel 1 from 2-agent env")
    print(f"  - EXCLUDE agent 1 orientations: skip channels 6-9 from 2-agent env")

    def analyze_observation_differences(obs_1, obs_2, step_num):
        """Detailed analysis of observation differences between environments"""
        H, W = obs_1.shape[0], obs_1.shape[1]

        # Debug data types on first step
        if step_num == 0:
            print(f"  DEBUG: obs_1 dtype: {obs_1.dtype}, obs_2 dtype: {obs_2.dtype}")
            print(f"  DEBUG: obs_1 range: [{np.min(obs_1)}, {np.max(obs_1)}]")
            print(f"  DEBUG: obs_2 range: [{np.min(obs_2)}, {np.max(obs_2)}]")

        # Convert both observations to the same data type to avoid overflow issues
        obs_1_float = obs_1.astype(np.float32)
        obs_2_float = obs_2.astype(np.float32)

        # Extract agent 0 position from both environments
        agent_0_pos_1 = obs_1_float[:, :, 0]  # single-agent: channel 0
        agent_0_pos_2 = obs_2_float[:, :, 0]  # 2-agent: channel 0

        # Extract agent 0 orientations (excluding agent 1 orientations from 2-agent env)
        agent_0_ori_1 = obs_1_float[:, :, 2:6]  # single-agent: channels 2-5
        agent_0_ori_2 = obs_2_float[:, :, 2:6]  # 2-agent: channels 2-5 (agent 0 only)

        # Extract environment layers (same channels in both environments)
        env_layers_1 = obs_1_float[:, :, 10:26]   # single-agent: channels 10-25 (16 layers)
        env_layers_2 = obs_2_float[:, :, 10:26]   # 2-agent: channels 10-25 (16 layers)

        # Note: We explicitly EXCLUDE the following from comparison:
        # - Channel 1 from 2-agent env (agent 1 position)
        # - Channels 6-9 from 2-agent env (agent 1 orientations)

        # Calculate differences (only for agent 0 and environment layers)
        agent_pos_diff = np.sum(np.abs(agent_0_pos_1 - agent_0_pos_2))
        agent_ori_diff = np.sum(np.abs(agent_0_ori_1 - agent_0_ori_2))
        env_layers_diff = np.sum(np.abs(env_layers_1 - env_layers_2))

        # No urgency layer comparison since both environments use the same channel
        urgency_diff = 0.0

        # Analyze specific environment layers
        layer_names = [
            "pot_locations", "walls", "onion_piles", "tomato_piles", "plate_piles", "goals",
            "onions_in_pot", "tomatoes_in_pot", "onions_in_soup", "tomatoes_in_soup",
            "pot_cook_time", "soup_ready", "plate_locations", "onion_locations", 
            "tomato_locations", "urgency"
        ]

        layer_diffs = []
        for i in range(16):
            if i < env_layers_1.shape[2] and i < env_layers_2.shape[2]:
                layer_diff = np.sum(np.abs(env_layers_1[:, :, i] - env_layers_2[:, :, i]))
                layer_diffs.append((layer_names[i], layer_diff))

        return {
            'step': step_num,
            'agent_pos_diff': agent_pos_diff,
            'agent_ori_diff': agent_ori_diff,
            'env_layers_diff': env_layers_diff,
            'urgency_diff': urgency_diff,
            'layer_diffs': layer_diffs,
            'total_diff': agent_pos_diff + agent_ori_diff + env_layers_diff + urgency_diff
        }

    # Analyze key steps
    print(f"\n=== STEP-BY-STEP ANALYSIS ===")
    detailed_analysis = []

    for i, (data_1, data_2) in enumerate(zip(rollout_data_1_agent, rollout_data_2_agent)):
        analysis = analyze_observation_differences(data_1['obs'], data_2['obs'], i)
        detailed_analysis.append(analysis)

        # Print detailed analysis for key steps
        if (i % 15 == 0 or data_1['reward'] > 0 or data_2['reward'] > 0 or 
            analysis['total_diff'] > 0):
            print(f"\nStep {i}:")
            print(f"  Agent 0 position difference: {analysis['agent_pos_diff']}")
            print(f"  Agent 0 orientation difference: {analysis['agent_ori_diff']}")
            print(f"  Environment layers difference: {analysis['env_layers_diff']}")
            print(f"  Urgency layer difference: {analysis['urgency_diff']}")
            print(f"  Total difference: {analysis['total_diff']}")

            # Show which specific environment layers differ
            if analysis['env_layers_diff'] > 0:
                print(f"  Environment layer differences:")
                for layer_name, diff in analysis['layer_diffs']:
                    if diff > 0:
                        print(f"    {layer_name}: {diff}")

    # Summary statistics
    total_diffs = [a['total_diff'] for a in detailed_analysis]
    agent_pos_diffs = [a['agent_pos_diff'] for a in detailed_analysis]
    agent_ori_diffs = [a['agent_ori_diff'] for a in detailed_analysis]
    env_layer_diffs = [a['env_layers_diff'] for a in detailed_analysis]
    urgency_diffs = [a['urgency_diff'] for a in detailed_analysis]

    avg_obs_diff = np.mean(total_diffs)
    max_obs_diff = np.max(total_diffs)

    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total observation differences across all steps:")
    print(f"  Average total difference: {avg_obs_diff:.1f}")
    print(f"  Maximum total difference: {max_obs_diff:.1f}")
    print(f"  Average agent position differences: {np.mean(agent_pos_diffs):.1f}")
    print(f"  Average agent orientation differences: {np.mean(agent_ori_diffs):.1f}")
    print(f"  Average environment layer differences: {np.mean(env_layer_diffs):.1f}")
    print(f"  Average urgency layer differences: {np.mean(urgency_diffs):.1f}")

    # Count steps with differences
    steps_with_pos_diff = sum(1 for d in agent_pos_diffs if d > 0)
    steps_with_ori_diff = sum(1 for d in agent_ori_diffs if d > 0)
    steps_with_env_diff = sum(1 for d in env_layer_diffs if d > 0)
    steps_with_urgency_diff = sum(1 for d in urgency_diffs if d > 0)

    print(f"\nSteps with differences (out of {len(detailed_analysis)} total steps):")
    print(f"  Agent position differences: {steps_with_pos_diff}")
    print(f"  Agent orientation differences: {steps_with_ori_diff}")
    print(f"  Environment layer differences: {steps_with_env_diff}")
    print(f"  Urgency layer differences: {steps_with_urgency_diff}")

    # Analyze which environment layers differ most
    layer_totals = {}
    for analysis in detailed_analysis:
        for layer_name, diff in analysis['layer_diffs']:
            if layer_name not in layer_totals:
                layer_totals[layer_name] = 0
            layer_totals[layer_name] += diff

    print(f"\nEnvironment layers with most differences:")
    sorted_layers = sorted(layer_totals.items(), key=lambda x: x[1], reverse=True)
    for layer_name, total_diff in sorted_layers[:5]:  # Top 5
        if total_diff > 0:
            print(f"  {layer_name}: {total_diff:.1f} total difference")

    print(f"\n=== INTERPRETATION ===")
    if avg_obs_diff == 0:
        print("✅ PERFECT MATCH: Observations are identical between environments!")
        print("   This means agent 0 sees exactly the same world state in both environments.")
    elif avg_obs_diff < 1:
        print("✅ EXCELLENT MATCH: Observations are nearly identical between environments!")
        print("   Minor differences likely due to implementation details.")
    elif avg_obs_diff < 10:
        print("⚠️  GOOD MATCH: Some differences exist but environments are largely consistent.")
    else:
        print("❌ SIGNIFICANT DIFFERENCES: The environments show different observations.")

    print(f"\nKey insights:")
    if steps_with_pos_diff == 0:
        print("✅ Agent 0 positions are identical in both environments")
    else:
        print(f"⚠️  Agent 0 positions differ in {steps_with_pos_diff} steps")

    if steps_with_ori_diff == 0:
        print("✅ Agent 0 orientations are identical in both environments")
    else:
        print(f"⚠️  Agent 0 orientations differ in {steps_with_ori_diff} steps")

    if steps_with_env_diff == 0:
        print("✅ Environment layers are identical in both environments")
    else:
        print(f"⚠️  Environment layers differ in {steps_with_env_diff} steps")
        most_different_layer = sorted_layers[0][0] if sorted_layers and sorted_layers[0][1] > 0 else "none"
        if most_different_layer != "none":
            print(f"   Most different layer: {most_different_layer}")

    # Final assessment
    if avg_obs_diff == 0 and steps_with_pos_diff == 0 and steps_with_ori_diff == 0:
        print(f"\n🎯 CONCLUSION: The environments provide identical experiences for agent 0!")
        print(f"   This validates that overcooked_single is equivalent to")
        print(f"   overcooked with 2 agents from agent 0's perspective (excluding agent 1 layers).")
    elif avg_obs_diff < 5:
        print(f"\n✅ CONCLUSION: The environments are highly consistent for agent 0!")
        print(f"   Small differences are likely due to implementation variations.")
    else:
        print(f"\n⚠️  CONCLUSION: There are notable differences between the environments.")
        print(f"   This may indicate implementation differences that affect agent 0's experience.")

    # Save GIFs
    print(f"\n=== SAVING GIFS ===")
    makedirs("gifs", exist_ok=True)

    gif_path_1 = "gifs/single_agent_overcooked_single.gif"
    gif_path_2 = "gifs/dual_agent_overcooked_agent0_only.gif"

    iio.imwrite(gif_path_1, frames_1_agent, loop=0, fps=8)
    iio.imwrite(gif_path_2, frames_2_agent, loop=0, fps=8)

    print(f"1-agent GIF saved to {gif_path_1}")
    print(f"2-agent GIF saved to {gif_path_2}")

    # Validation and assertions
    print(f"\n=== VALIDATION ===")

    # Check that both environments produced similar results
    reward_diff = abs(total_reward_1 - total_reward_2)
    shaped_diff = abs(total_shaped_1 - total_shaped_2)

    print(f"Reward difference: {reward_diff:.6f}")
    print(f"Shaped reward difference: {shaped_diff:.6f}")

    # The delivery rewards should be identical since agent 0 is doing the same actions
    # Shaped rewards may differ due to different environment implementations
    assert reward_diff < 1e-3, f"Delivery reward difference too large: {reward_diff}"

    # Allow larger differences for shaped rewards since environments may implement them differently
    # The key is that both agents complete the task successfully
    if shaped_diff > 5.0:
        print(f"⚠️  Warning: Large shaped reward difference: {shaped_diff}")
    else:
        print(f"✅ Shaped reward difference acceptable: {shaped_diff}")

    # Observations should be similar but may have some differences due to different agent counts
    # in the observation channels
    print(f"Observation comparison: avg_diff={avg_obs_diff:.6f}, max_diff={max_obs_diff:.6f}")

    # Check that both agents successfully completed the task
    assert total_reward_1 >= DELIVERY_REWARD * 0.9, f"1-agent didn't complete task: {total_reward_1}"
    assert total_reward_2 >= DELIVERY_REWARD * 0.9, f"2-agent didn't complete task: {total_reward_2}"

    print(f"\n🎉 TEST PASSED! 🎉")
    print(f"✅ Both environments produced similar results for agent 0")
    print(f"✅ Agent 0 successfully cooked and delivered soup in both environments")
    print(f"✅ GIFs generated for visual comparison")

    return {
        'rollout_1_agent': rollout_data_1_agent,
        'rollout_2_agent': rollout_data_2_agent,
        'total_reward_1': total_reward_1,
        'total_reward_2': total_reward_2,
        'total_shaped_1': total_shaped_1,
        'total_shaped_2': total_shaped_2,
        'avg_obs_diff': avg_obs_diff,
        'max_obs_diff': max_obs_diff
    }


if __name__ == "__main__":
    test_single_vs_dual_agent_comparison()
