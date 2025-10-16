#!/usr/bin/env python3
"""
Example usage of the Partially Observable Overcooked environment.
This demonstrates how to create and use the environment with different view settings.
"""

import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import argparse
import jax
import jax.numpy as jnp
import numpy as np
from meal.env.overcooked_po import OvercookedPO
from meal.visualization.visualizer_po import OvercookedVisualizerPO
from meal.env.generation.layout_generator import generate_random_layout
from meal.env.utils.difficulty_config import get_difficulty_params

# For GIF creation
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL (Pillow) not available. GIF creation will be disabled.")

import os

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")

    # Create partially observable environment with default settings
    env = OvercookedPO(
        layout_name="cramped_room",
        view_ahead=3,      # 3 tiles ahead
        view_behind=1,     # 1 tile behind  
        view_sides=1,      # 1 tile to each side
        max_steps=400
    )

    print(f"Environment: {env.name}")
    print(f"View settings: ahead={env.view_ahead}, behind={env.view_behind}, sides={env.view_sides}")

    # Reset environment
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    print(f"Observation shape: {obs['agent_0'].shape}")

    # Get view masks to see what each agent can observe
    view_masks = env.get_agent_view_masks(state)
    for agent_key, mask in view_masks.items():
        visible_tiles = jnp.sum(mask)
        print(f"{agent_key} can see {visible_tiles} tiles")

    # Run a few steps
    for step in range(5):
        # Random actions
        actions = {
            "agent_0": jax.random.randint(key, (), 0, env.num_actions),
            "agent_1": jax.random.randint(key, (), 0, env.num_actions)
        }
        key, subkey = jax.random.split(key)

        obs, state, rewards, dones, info = env.step(subkey, state, actions)

        if step == 0:  # Print view masks after first step
            view_masks = env.get_agent_view_masks(state)
            for agent_key, mask in view_masks.items():
                visible_tiles = jnp.sum(mask)
                print(f"After step {step + 1}: {agent_key} can see {visible_tiles} tiles")

def example_different_view_settings():
    """Example with different view settings"""
    print("\n=== Different View Settings Example ===")

    view_configs = [
        {"name": "Very Limited", "ahead": 1, "behind": 0, "sides": 0},
        {"name": "Moderate", "ahead": 2, "behind": 1, "sides": 1},
        {"name": "Wide View", "ahead": 4, "behind": 2, "sides": 2},
    ]

    for config in view_configs:
        env = OvercookedPO(
            layout_name="cramped_room",
            view_ahead=config["ahead"],
            view_behind=config["behind"],
            view_sides=config["sides"]
        )

        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        view_masks = env.get_agent_view_masks(state)
        total_visible = sum(jnp.sum(mask) for mask in view_masks.values())

        print(f"{config['name']} (ahead={config['ahead']}, behind={config['behind']}, sides={config['sides']}): "
              f"Total visible tiles = {total_visible}")

def example_visualization():
    """Example with visualization"""
    print("\n=== Visualization Example ===")

    # Create environment and visualizer
    env = OvercookedPO(
        layout_name="cramped_room",
        view_ahead=3,
        view_behind=1,
        view_sides=1
    )

    visualizer = OvercookedVisualizerPO(use_old_rendering=False)

    # Reset environment
    key = jax.random.PRNGKey(123)
    obs, state = env.reset(key)

    try:
        # Render with view area highlighting
        img = visualizer.render(
            agent_view_size=5,  # Not used in PO version
            state=state,
            env=env,           # Pass env to get view masks
            highlight_views=True,
            tile_size=32
        )

        print(f"Successfully rendered image with shape: {img.shape}")
        print("View areas are highlighted with:")
        print("- Light red for agent 0")
        print("- Light blue for agent 1")
        print("- Purple where both agents can see")

        # You could save the image here if needed:
        # import matplotlib.pyplot as plt
        # plt.imsave('overcooked_po_example.png', img)

    except Exception as e:
        print(f"Visualization failed: {e}")
        print("This might be due to missing dependencies or display issues")

def example_observation_analysis():
    """Analyze the partial observations"""
    print("\n=== Observation Analysis Example ===")

    # Create environment with limited view
    env = OvercookedPO(
        layout_name="cramped_room",
        view_ahead=2,
        view_behind=1,
        view_sides=1
    )

    key = jax.random.PRNGKey(456)
    obs, state = env.reset(key)

    # Analyze observations
    for agent_key, agent_obs in obs.items():
        print(f"\n{agent_key} observation analysis:")

        # Count unseen tiles (object type 0)
        unseen_tiles = jnp.sum(agent_obs[:, :, 0] == 0)
        total_tiles = agent_obs.shape[0] * agent_obs.shape[1]
        visible_tiles = total_tiles - unseen_tiles

        print(f"  Total tiles: {total_tiles}")
        print(f"  Visible tiles: {visible_tiles}")
        print(f"  Unseen tiles: {unseen_tiles}")
        print(f"  Visibility ratio: {visible_tiles/total_tiles:.2%}")

        # Check what objects are visible
        visible_objects = set()
        for i in range(agent_obs.shape[0]):
            for j in range(agent_obs.shape[1]):
                if agent_obs[i, j, 0] != 0:  # Not unseen
                    visible_objects.add(int(agent_obs[i, j, 0]))

        print(f"  Visible object types: {sorted(visible_objects)}")

def create_simple_visualization(state, env, width=200, height=160):
    """Create a simple visualization when full rendering fails"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Create figure
    fig, ax = plt.subplots(figsize=(width/80, height/80), dpi=80)
    ax.set_xlim(0, env.obs_shape[0])
    ax.set_ylim(0, env.obs_shape[1])
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match grid coordinates

    # Draw grid
    for i in range(env.obs_shape[0] + 1):
        ax.axvline(i, color='lightgray', linewidth=0.5)
    for i in range(env.obs_shape[1] + 1):
        ax.axhline(i, color='lightgray', linewidth=0.5)

    # Get view masks
    view_masks = env.get_agent_view_masks(state)

    # Draw view areas
    colors = ['red', 'blue']
    alphas = [0.3, 0.3]

    for agent_idx, (agent_key, mask) in enumerate(view_masks.items()):
        if agent_idx < len(colors):
            mask_array = np.array(mask)
            for i in range(mask_array.shape[0]):
                for j in range(mask_array.shape[1]):
                    if mask_array[i, j]:
                        rect = patches.Rectangle((j, i), 1, 1, 
                                               facecolor=colors[agent_idx], 
                                               alpha=alphas[agent_idx])
                        ax.add_patch(rect)

    # Draw agents
    for agent_idx in range(env.num_agents):
        agent_pos = state.agent_pos[agent_idx]
        x, y = agent_pos[0], agent_pos[1]
        circle = patches.Circle((x + 0.5, y + 0.5), 0.3, 
                               facecolor=colors[agent_idx], 
                               edgecolor='black', linewidth=2)
        ax.add_patch(circle)

    ax.set_title(f'Partially Observable Overcooked\nAgent Views', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # Convert to image
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return buf

def create_episode_gif(num_steps=50, gif_filename="overcooked_po_episode.gif", tile_size=32, seed=None, difficulty="hard"):
    """Create a GIF showing a short episode with agents moving and their field of view
    Also saves individual frame images to the images directory.

    Args:
        num_steps: Number of steps to run the episode
        gif_filename: Name of the output GIF file
        tile_size: Size of each tile in pixels for rendering
        seed: Random seed for environment generation (if None, uses random seed)
    """
    print(f"\n=== Creating Episode GIF ({num_steps} steps) ===")

    if not PIL_AVAILABLE:
        print("Cannot create GIF: PIL (Pillow) not available.")
        print("Install with: pip install Pillow")
        return

    # Check if matplotlib is available for fallback rendering
    try:
        import matplotlib.pyplot as plt
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        print("Warning: matplotlib not available for fallback rendering")

    # Create images directory for saving individual frames
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    print(f"Individual frame images will be saved to: {images_dir}")

    # Generate a random seed if not provided
    if seed is None:
        seed = np.random.randint(0, 10000)

    print(f"Using seed: {seed}")

    # Generate random layout with specified difficulty
    print(f"Generating {difficulty} difficulty environment...")
    try:
        grid_str, layout = generate_random_layout(
            num_agents=2,
            difficulty=difficulty,
            seed=seed,
            max_attempts=100
        )
        print(f"Generated layout: {layout['width']}x{layout['height']} with {params['wall_density']} wall density")
    except Exception as e:
        print(f"Failed to generate {difficulty} environment: {e}")
        print("Falling back to cramped_room layout")
        layout = None

    # Create environment with difficulty-based view settings
    env = OvercookedPO(
        layout=layout,
        layout_name=f"generated_{difficulty}",
        view_ahead=params["view_ahead"],
        view_behind=params["view_behind"],
        view_sides=params["view_sides"],
        max_steps=num_steps + 10  # Add buffer
    )

    print(f"Using {difficulty} difficulty view settings:")
    print(f"  view_ahead={params['view_ahead']}, view_behind={params['view_behind']}, view_sides={params['view_sides']}")

    visualizer = OvercookedVisualizerPO(use_old_rendering=False)

    # Reset environment
    key = jax.random.PRNGKey(42)  # Fixed seed for reproducible results
    obs, state = env.reset(key)

    frames = []

    print(f"Running episode for {num_steps} steps...")

    try:
        # Capture initial frame
        img = visualizer.render(
            agent_view_size=5,  # Not used in PO version
            state=state,
            env=env,
            highlight_views=True,
            tile_size=tile_size
        )
        if img is not None:
            frames.append(Image.fromarray(img))
            # Save initial frame as individual image
            try:
                filename = f"{images_dir}/gif_frame_{0:03d}.png"
                plt.imsave(filename, img)
                print(f"Saved initial frame: {filename}")
            except Exception as e:
                print(f"Failed to save initial frame image: {e}")
        elif MATPLOTLIB_AVAILABLE:
            print("Using fallback rendering for initial frame")
            img = create_simple_visualization(state, env)
            frames.append(Image.fromarray(img))
            # Save fallback initial frame as individual image
            try:
                filename = f"{images_dir}/gif_frame_{0:03d}.png"
                plt.imsave(filename, img)
                print(f"Saved initial fallback frame: {filename}")
            except Exception as e:
                print(f"Failed to save initial fallback frame image: {e}")
        else:
            print("Warning: Initial frame rendering failed and no fallback available")

        # Run episode and capture frames
        for step in range(num_steps):
            # Use a mix of random actions and some directed movement
            # This creates more interesting movement patterns
            # Note: Overcooked has 6 actions: [stay, north, south, east, west, interact]
            if step < 10:
                # First 10 steps: mostly move actions to see agents moving
                action_probs = [0.1, 0.225, 0.225, 0.225, 0.225, 0.0]  # [stay, north, south, east, west, interact]
            elif step < 20:
                # Next 10 steps: include some interactions
                action_probs = [0.1, 0.18, 0.18, 0.18, 0.18, 0.18]
            else:
                # Remaining steps: more random behavior
                action_probs = [0.2, 0.16, 0.16, 0.16, 0.16, 0.16]

            # Sample actions based on probabilities
            key, key_a0, key_a1 = jax.random.split(key, 3)

            action_0 = jax.random.choice(key_a0, env.num_actions, p=jnp.array(action_probs))
            action_1 = jax.random.choice(key_a1, env.num_actions, p=jnp.array(action_probs))

            actions = {
                "agent_0": action_0,
                "agent_1": action_1
            }

            # Step environment
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(step_key, state, actions)

            # Render frame with view highlighting
            img = visualizer.render(
                agent_view_size=5,
                state=state,
                env=env,
                highlight_views=True,
                tile_size=tile_size
            )
            if img is not None:
                frames.append(Image.fromarray(img))
                # Save frame as individual image
                try:
                    filename = f"{images_dir}/gif_frame_{step + 1:03d}.png"
                    plt.imsave(filename, img)
                    if step < 5 or (step + 1) % 10 == 0:  # Print progress for first few frames and every 10th frame
                        print(f"Saved frame {step + 1}: {filename}")
                except Exception as e:
                    if step < 5:  # Only print error for first few frames to avoid spam
                        print(f"Failed to save frame {step + 1} image: {e}")
            elif MATPLOTLIB_AVAILABLE:
                # Use fallback rendering only occasionally to avoid spam
                if step < 5 or step % 10 == 0:
                    print(f"Using fallback rendering for frame {step + 1}")
                img = create_simple_visualization(state, env)
                frames.append(Image.fromarray(img))
                # Save fallback frame as individual image
                try:
                    filename = f"{images_dir}/gif_frame_{step + 1:03d}.png"
                    plt.imsave(filename, img)
                    print(f"Saved fallback frame {step + 1}: {filename}")
                except Exception as e:
                    print(f"Failed to save fallback frame {step + 1} image: {e}")
            else:
                if step < 5:  # Only print warning for first few frames
                    print(f"Warning: Frame {step + 1} rendering failed and no fallback available")

            # Print progress every 10 steps
            if (step + 1) % 10 == 0:
                print(f"  Captured frame {step + 1}/{num_steps}")

            # Break if episode is done
            if dones["__all__"]:
                print(f"  Episode finished early at step {step + 1}")
                break

        # Save GIF
        if frames:
            print(f"Saving GIF with {len(frames)} frames to 'gifs/{gif_filename}'...")

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(gif_filename) if os.path.dirname(gif_filename) else ".", exist_ok=True)

            # Save as GIF with appropriate duration
            frames[0].save(
                gif_filename,
                save_all=True,
                append_images=frames[1:],
                duration=200,  # 200ms per frame (5 FPS)
                loop=0  # Loop forever
            )

            print(f"Successfully created GIF: {gif_filename}")
            print(f"GIF contains {len(frames)} frames showing agents' field of view")
            print(f"Individual frame images also saved to: {images_dir}")
            print("View areas are highlighted with:")
            print("- Light red for agent 0's field of view")
            print("- Light blue for agent 1's field of view")
            print("- Purple where both agents can see")
        else:
            print("No frames captured - GIF creation failed")

    except Exception as e:
        print(f"GIF creation failed: {e}")
        print("This might be due to rendering issues or missing dependencies")
        import traceback
        traceback.print_exc()

def main():
    """Run all examples"""
    print("Partially Observable Overcooked Environment Examples")
    print("=" * 60)

    try:
        example_basic_usage()
        example_different_view_settings()
        example_visualization()
        example_observation_analysis()

        # Create episode GIF showing agents moving and their field of view
        # Use a random seed for environment generation
        import random
        random_seed = random.randint(1, 10000)
        create_episode_gif(
            num_steps=30,  # Shorter episode for faster generation
            gif_filename="gifs/overcooked_po_episode.gif",
            tile_size=32,
            seed=random_seed,
            difficulty="medium"
        )

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("\nKey Features:")
        print("- Configurable view distances (ahead, behind, sides)")
        print("- Direction-aware partial observability")
        print("- View area visualization with highlighting")
        print("- JAX-compatible implementation")
        print("- Backward compatible with base Overcooked environment")
        print("- Episode GIF generation showing field of view changes")

    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test Partially Observable Overcooked environment with configurable difficulty"
    )

    parser.add_argument(
        "--difficulty", 
        type=str, 
        choices=["easy", "medium", "hard"], 
        default="medium",
        help="Difficulty level that determines both environment complexity and partial view size (default: medium)"
    )

    parser.add_argument(
        "--num-steps", 
        type=int, 
        default=30,
        help="Number of steps to run in the episode (default: 30)"
    )

    parser.add_argument(
        "--gif-filename", 
        type=str, 
        default="gifs/overcooked_po_episode.gif",
        help="Output filename for the GIF (default: gifs/overcooked_po_episode.gif)"
    )

    parser.add_argument(
        "--tile-size", 
        type=int, 
        default=32,
        help="Size of each tile in pixels for rendering (default: 32)"
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for environment generation (default: random)"
    )

    parser.add_argument(
        "--run-examples", 
        action="store_true",
        help="Run all example functions before creating the GIF"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("Partially Observable Overcooked Environment")
    print("=" * 60)
    print(f"Difficulty: {args.difficulty}")

    # Show difficulty parameters
    try:
        params = get_difficulty_params(args.difficulty)
        print(f"Environment size: {params['height_rng'][0]}-{params['height_rng'][1]} x {params['width_rng'][0]}-{params['width_rng'][1]}")
        print(f"Wall density: {params['wall_density']}")
        print(f"View settings: ahead={params['view_ahead']}, behind={params['view_behind']}, sides={params['view_sides']}")
    except Exception as e:
        print(f"Error getting difficulty parameters: {e}")
        exit(1)

    print("=" * 60)

    try:
        # Run examples if requested
        if args.run_examples:
            example_basic_usage()
            example_different_view_settings()
            example_visualization()
            example_observation_analysis()

        # Create episode GIF with specified difficulty
        create_episode_gif(
            num_steps=args.num_steps,
            gif_filename=args.gif_filename,
            tile_size=args.tile_size,
            seed=args.seed,
            difficulty=args.difficulty
        )

        print("\n" + "=" * 60)
        print("GIF creation completed successfully!")
        print(f"Difficulty: {args.difficulty}")
        print(f"Output: {args.gif_filename}")
        print("\nKey Features:")
        print("- Difficulty-based environment complexity and partial view size")
        print("- Direction-aware partial observability")
        print("- View area visualization with highlighting")
        print("- JAX-compatible implementation")
        print("- Episode GIF generation showing field of view changes")

    except Exception as e:
        print(f"\nScript failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
