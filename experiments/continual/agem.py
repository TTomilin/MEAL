import jax
import jax.numpy as jnp
from flax import struct
from jax._src.flatten_util import ravel_pytree


@struct.dataclass
class AGEMMemory:
    obs: jnp.ndarray  # [max_tasks, max_size_per_task, *obs_shape]
    actions: jnp.ndarray  # [max_tasks, max_size_per_task]
    log_probs: jnp.ndarray  # [max_tasks, max_size_per_task]
    advantages: jnp.ndarray  # [max_tasks, max_size_per_task]
    targets: jnp.ndarray  # [max_tasks, max_size_per_task]
    values: jnp.ndarray  # [max_tasks, max_size_per_task]
    ptrs: jnp.ndarray  # [max_tasks]   write pointer for each task
    sizes: jnp.ndarray  # [max_tasks]   how many valid samples per task
    max_tasks: int
    max_size_per_task: int


class AGEM:
    """
    AGEM (Averaged Gradient Episodic Memory) continual learning method.

    This class implements the AGEM method for IPPO_CL.py.
    """

    def __init__(self, memory_size=1000, sample_size=128):
        """
        Initialize the AGEM method.

        Args:
            memory_size: Size of the memory buffer
            sample_size: Number of samples to use for gradient projection
        """
        self.memory_size = memory_size
        self.sample_size = sample_size

    def init_state(self, params, regularize_critic=True, regularize_heads=True):
        """
        Initialize the AGEM state.

        Args:
            params: Initial parameters
            regularize_critic: Whether to regularize the critic (not used in AGEM)
            regularize_heads: Whether to regularize the heads (not used in AGEM)

        Returns:
            AGEM state (memory buffer)
        """
        # For AGEM, we'll initialize an empty memory buffer
        # The actual initialization will happen in the first call to update_state
        # We return None for now, and handle it in update_state
        return None

    def compute_importance(self, params, env, network, env_idx, rng, use_cnn=False,
                           num_episodes=5, max_steps=100, normalize=False):
        """
        Compute the importance of parameters for a task.

        For AGEM, this is not used, but we need to implement it for compatibility.

        Args:
            params: Parameters to compute importance for
            env: Environment
            network: Network
            env_idx: Environment index
            rng: Random number generator
            use_cnn: Whether to use CNN
            num_episodes: Number of episodes to run
            max_steps: Maximum number of steps per episode
            normalize: Whether to normalize importance

        Returns:
            Importance (not used in AGEM, so we return None)
        """
        # AGEM doesn't use importance, so we return None
        return None

    def update_state(self, state, params, importance):
        """
        Update the AGEM state with new parameters and importance.

        For AGEM, this would update the memory buffer, but we'll handle that
        separately in the training loop.

        Args:
            state: Current AGEM state
            params: New parameters
            importance: Importance (not used in AGEM)

        Returns:
            Updated AGEM state
        """
        # AGEM doesn't update state based on importance, so we just return the current state
        return state

    def penalty(self, params, state, reg_coef):
        """
        Calculate the regularization penalty.

        For AGEM, there is no regularization penalty, so we return 0.

        Args:
            params: Parameters to calculate penalty for
            state: AGEM state
            reg_coef: Regularization coefficient

        Returns:
            Regularization penalty (0 for AGEM)
        """
        # AGEM doesn't use a regularization penalty, so we return 0
        return 0.0


def init_agem_memory(max_size: int, obs_shape: tuple, max_tasks: int = 20):
    """Create an *empty* per-task buffer."""
    max_size_per_task = max_size // max_tasks  # Divide total memory among tasks
    zeros = lambda *shape: jnp.zeros(shape, dtype=jnp.float32)
    zeros_int = lambda *shape: jnp.zeros(shape, dtype=jnp.int32)
    return AGEMMemory(
        obs=zeros(max_tasks, max_size_per_task, *obs_shape),
        actions=zeros(max_tasks, max_size_per_task),
        log_probs=zeros(max_tasks, max_size_per_task),
        advantages=zeros(max_tasks, max_size_per_task),
        targets=zeros(max_tasks, max_size_per_task),
        values=zeros(max_tasks, max_size_per_task),
        ptrs=zeros_int(max_tasks),
        sizes=zeros_int(max_tasks),
        max_tasks=max_tasks,
        max_size_per_task=max_size_per_task,
    )


def agem_project(grads_ppo, grads_mem, max_norm=40.):
    """
    Implements the AGEM projection:
      If g_new^T g_mem < 0:
         g_new := g_new - (g_new^T g_mem / ||g_mem||^2) * g_mem
    """
    g_new, unravel = ravel_pytree(grads_ppo)
    g_mem, _ = ravel_pytree(grads_mem)

    # Compute gradient norms for logging
    ppo_grad_norm = jnp.linalg.norm(g_new)
    mem_grad_norm = jnp.linalg.norm(g_mem)

    dot_g = jnp.vdot(g_new, g_mem)
    dot_mem = jnp.vdot(g_mem, g_mem) + 1e-12
    alpha = dot_g / dot_mem
    projected = jax.lax.cond(
        dot_g < 0,
        lambda _: g_new - alpha * g_mem,
        lambda _: g_new,
        operand=None
    )

    # second global-norm clip (recommended in AGEM paper)
    projected_grad_norm = jnp.linalg.norm(projected)
    projected = jnp.where(projected_grad_norm > max_norm, projected * (max_norm / projected_grad_norm), projected)

    # Final gradient norm after clipping
    final_grad_norm = jnp.linalg.norm(projected)

    stats = dict(
        agem_dot_g=dot_g, 
        agem_alpha=alpha, 
        agem_is_proj=(dot_g < 0),
        agem_ppo_grad_norm=ppo_grad_norm,
        agem_mem_grad_norm=mem_grad_norm,
        agem_projected_grad_norm=projected_grad_norm,
        agem_final_grad_norm=final_grad_norm
    )

    # Prefix all stats with "agem/" for consistent wandb logging
    stats = {f"agem/{k}": v for k, v in stats.items()}
    return unravel(projected), stats


def sample_memory(mem: AGEMMemory, sample_size: int, rng):
    """Sample uniformly from data of all past tasks."""
    # Find which tasks have data
    total_samples = jnp.sum(mem.sizes)

    # If no tasks have data, return zeros
    obs_shape = mem.obs.shape[2:]  # Remove task and per-task dimensions

    def no_data_case():
        return (
            jnp.zeros((sample_size, *obs_shape)),
            jnp.zeros((sample_size,)),
            jnp.zeros((sample_size,)),
            jnp.zeros((sample_size,)),
            jnp.zeros((sample_size,)),
            jnp.zeros((sample_size,))
        )

    def sample_from_tasks():
        # Simple approach: randomly select task indices and sample indices
        rng_task, rng_sample = jax.random.split(rng)

        # For each sample, choose a random task weighted by its size
        cumulative_sizes = jnp.cumsum(mem.sizes)
        random_positions = jax.random.randint(rng_task, (sample_size,), 0, total_samples)

        # Find which task each random position belongs to
        task_indices = jnp.searchsorted(cumulative_sizes, random_positions, side='right')
        task_indices = jnp.clip(task_indices, 0, mem.max_tasks - 1)

        # For each selected task, choose a random sample index
        sample_indices = jax.random.randint(rng_sample, (sample_size,), 0, jnp.maximum(jnp.max(mem.sizes), 1))
        sample_indices = jnp.minimum(sample_indices, mem.sizes[task_indices] - 1)
        sample_indices = jnp.maximum(sample_indices, 0)  # Ensure non-negative

        # Extract the samples
        sampled_obs = mem.obs[task_indices, sample_indices]
        sampled_actions = mem.actions[task_indices, sample_indices]
        sampled_logp = mem.log_probs[task_indices, sample_indices]
        sampled_advs = mem.advantages[task_indices, sample_indices]
        sampled_targets = mem.targets[task_indices, sample_indices]
        sampled_values = mem.values[task_indices, sample_indices]

        return sampled_obs, sampled_actions, sampled_logp, sampled_advs, sampled_targets, sampled_values

    return jax.lax.cond(
        total_samples > 0,
        lambda: sample_from_tasks(),
        lambda: no_data_case()
    )


def compute_memory_gradient(network, params,
                            clip_eps, vf_coef, ent_coef,
                            mem_obs, mem_actions,
                            mem_advs, mem_log_probs,
                            mem_targets, mem_values,
                            env_idx=0):
    """Compute the same clipped PPO loss on the memory data."""

    # Stop gradient flow through memory data
    mem_obs = jax.lax.stop_gradient(mem_obs)
    mem_actions = jax.lax.stop_gradient(mem_actions)
    mem_advs = jax.lax.stop_gradient(mem_advs)
    mem_log_probs = jax.lax.stop_gradient(mem_log_probs)
    mem_targets = jax.lax.stop_gradient(mem_targets)
    mem_values = jax.lax.stop_gradient(mem_values)

    def loss_fn(params):
        pi, value, _ = network.apply(params, mem_obs, env_idx=env_idx)  # shapes: [B]
        log_prob = pi.log_prob(mem_actions)

        ratio = jnp.exp(log_prob - mem_log_probs)
        # standard advantage normalization
        adv_std = jnp.std(mem_advs) + 1e-8
        adv_mean = jnp.mean(mem_advs)
        normalized_adv = (mem_advs - adv_mean) / adv_std

        unclipped = ratio * normalized_adv
        clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * normalized_adv
        actor_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

        # Critic Loss (same as normal PPO)
        value_pred_clipped = mem_values + (value - mem_values).clip(-clip_eps, clip_eps)
        value_losses = (value - mem_targets) ** 2
        value_losses_clipped = (value_pred_clipped - mem_targets) ** 2
        critic_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))

        entropy = jnp.mean(pi.entropy())

        total_loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy
        return total_loss, (critic_loss, actor_loss, entropy)

    (total_loss, (v_loss, a_loss, ent)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    stats = {
        "agem/ppo_total_loss": total_loss,
        "agem/ppo_value_loss": v_loss,
        "agem/ppo_actor_loss": a_loss,
        "agem/ppo_entropy": ent
    }
    return grads, stats


def update_agem_memory(mem: AGEMMemory, task_idx: int,
                       new_obs, new_actions, new_log_probs,
                       new_adv, new_tgt, new_val):
    """
    Update AGEM memory using simple circular buffer per task.

    Args:
        mem: Current memory state
        task_idx: Index of the current task (0-indexed)
        new_obs, new_actions, etc.: New data to add to the task's buffer
    """
    b = new_obs.shape[0]  # batch size to insert

    # Ensure task_idx is within bounds
    task_idx = jnp.clip(task_idx, 0, mem.max_tasks - 1)

    # Get current pointer and size for this task
    current_ptr = mem.ptrs[task_idx]
    current_size = mem.sizes[task_idx]

    # Calculate indices where to insert new data (circular buffer)
    indices = (current_ptr + jnp.arange(b)) % mem.max_size_per_task

    # Update the memory for this specific task
    updated_obs = mem.obs.at[task_idx, indices].set(new_obs)
    updated_actions = mem.actions.at[task_idx, indices].set(new_actions)
    updated_log_probs = mem.log_probs.at[task_idx, indices].set(new_log_probs)
    updated_advantages = mem.advantages.at[task_idx, indices].set(new_adv)
    updated_targets = mem.targets.at[task_idx, indices].set(new_tgt)
    updated_values = mem.values.at[task_idx, indices].set(new_val)

    # Update pointer and size for this task
    new_ptr = (current_ptr + b) % mem.max_size_per_task
    new_size = jnp.minimum(current_size + b, mem.max_size_per_task)

    updated_ptrs = mem.ptrs.at[task_idx].set(new_ptr)
    updated_sizes = mem.sizes.at[task_idx].set(new_size)

    return mem.replace(
        obs=updated_obs,
        actions=updated_actions,
        log_probs=updated_log_probs,
        advantages=updated_advantages,
        targets=updated_targets,
        values=updated_values,
        ptrs=updated_ptrs,
        sizes=updated_sizes
    )
