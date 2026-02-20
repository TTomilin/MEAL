import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from jax._src.flatten_util import ravel_pytree

from experiments.continual.base import RegCLMethod


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


class AGEM(RegCLMethod):
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

    def update_state(self, state, params, importance):
        # AGEM doesn't update state based on importance, so we just return the current state
        return state

    def penalty(self, params, state, reg_coef):
        # AGEM doesn't use a regularization penalty, so we return 0
        return 0.0

    # ── importance function factory (to satisfy unified interface) ───────────
    def make_importance_fn(self, reset_switch, step_switch, network, agents, use_cnn: bool, max_episodes: int,
                           max_steps: int, norm_importance: bool, stride: int) -> callable:
        # Returns a jitted function with the same call signature but producing zeros.
        @jax.jit
        def importance_fn(params: FrozenDict, env_idx: jnp.int32, rng):
            return jax.tree.map(jnp.zeros_like, params)

        return importance_fn


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


@struct.dataclass
class VDNAGEMMemory:
    obs: jnp.ndarray       # [max_tasks, max_size_per_task, num_agents, obs_dim]
    actions: jnp.ndarray   # [max_tasks, max_size_per_task, num_agents]
    rewards: jnp.ndarray   # [max_tasks, max_size_per_task]  (joint __all__ reward)
    next_obs: jnp.ndarray  # [max_tasks, max_size_per_task, num_agents, obs_dim]
    dones: jnp.ndarray     # [max_tasks, max_size_per_task]  (__all__ done)
    ptrs: jnp.ndarray      # [max_tasks]
    sizes: jnp.ndarray     # [max_tasks]
    max_tasks: int
    max_size_per_task: int


def init_vdn_agem_memory(max_size: int, num_agents: int, obs_dim: int, max_tasks: int = 20):
    """Create an empty per-task VDN AGEM memory buffer."""
    max_size_per_task = max_size // max_tasks
    zeros = lambda *shape: jnp.zeros(shape, dtype=jnp.float32)
    zeros_int = lambda *shape: jnp.zeros(shape, dtype=jnp.int32)
    return VDNAGEMMemory(
        obs=zeros(max_tasks, max_size_per_task, num_agents, obs_dim),
        actions=zeros_int(max_tasks, max_size_per_task, num_agents),
        rewards=zeros(max_tasks, max_size_per_task),
        next_obs=zeros(max_tasks, max_size_per_task, num_agents, obs_dim),
        dones=zeros(max_tasks, max_size_per_task),
        ptrs=zeros_int(max_tasks),
        sizes=zeros_int(max_tasks),
        max_tasks=max_tasks,
        max_size_per_task=max_size_per_task,
    )


def sample_vdn_task_slot(mem: VDNAGEMMemory, task_idx: int, sample_size: int, rng):
    """Sample uniformly from a single task's VDN memory slot."""
    task_size = mem.sizes[task_idx]
    idx = jax.random.randint(rng, (sample_size,), 0, jnp.maximum(task_size, 1))
    idx = jnp.minimum(idx, jnp.maximum(task_size - 1, 0))
    return (
        mem.obs[task_idx, idx],       # (sample_size, num_agents, obs_dim)
        mem.actions[task_idx, idx],   # (sample_size, num_agents)
        mem.rewards[task_idx, idx],   # (sample_size,)
        mem.next_obs[task_idx, idx],  # (sample_size, num_agents, obs_dim)
        mem.dones[task_idx, idx],     # (sample_size,)
    )


def compute_vdn_memory_gradient(network, params, target_params, gamma,
                                mem_obs, mem_actions, mem_rewards, mem_next_obs, mem_dones,
                                env_idx=0):
    """Compute the VDN TD-loss gradient on memory data using the current target network."""
    mem_obs = jax.lax.stop_gradient(mem_obs)           # (B, A, D)
    mem_actions = jax.lax.stop_gradient(mem_actions)   # (B, A)
    mem_rewards = jax.lax.stop_gradient(mem_rewards)   # (B,)
    mem_next_obs = jax.lax.stop_gradient(mem_next_obs) # (B, A, D)
    mem_dones = jax.lax.stop_gradient(mem_dones)       # (B,)

    # Transpose to (A, B, D) for agent-wise vmap
    mem_obs_T = mem_obs.transpose(1, 0, 2)
    mem_next_obs_T = mem_next_obs.transpose(1, 0, 2)
    mem_actions_T = mem_actions.T  # (A, B)

    # Fresh TD targets from current target network
    q_next = jax.vmap(
        lambda o: network.apply(target_params, o, env_idx=env_idx), in_axes=0
    )(mem_next_obs_T)  # (A, B, action_dim)
    q_next_max_sum = q_next.max(-1).sum(0)  # (B,) — max per agent, sum over agents

    vdn_target = jax.lax.stop_gradient(
        mem_rewards + gamma * (1.0 - mem_dones) * q_next_max_sum
    )

    def loss_fn(p):
        q_vals = jax.vmap(
            lambda o: network.apply(p, o, env_idx=env_idx), in_axes=0
        )(mem_obs_T)  # (A, B, action_dim)
        chosen_q = jnp.take_along_axis(
            q_vals, mem_actions_T[..., jnp.newaxis], axis=-1
        ).squeeze(-1)  # (A, B)
        return jnp.mean((chosen_q.sum(0) - vdn_target) ** 2)

    return jax.grad(loss_fn)(params)


def update_vdn_agem_memory(mem: VDNAGEMMemory, env_idx: int,
                           obs_b, actions_b, rewards_all, next_obs_b, dones_all):
    """Store experience into the VDN AGEM circular buffer for the given task slot."""
    task_idx = jnp.clip(env_idx, 0, mem.max_tasks - 1)
    b = obs_b.shape[0]
    current_ptr = mem.ptrs[task_idx]
    current_size = mem.sizes[task_idx]
    indices = (current_ptr + jnp.arange(b)) % mem.max_size_per_task

    return mem.replace(
        obs=mem.obs.at[task_idx, indices].set(obs_b),
        actions=mem.actions.at[task_idx, indices].set(actions_b),
        rewards=mem.rewards.at[task_idx, indices].set(rewards_all),
        next_obs=mem.next_obs.at[task_idx, indices].set(next_obs_b),
        dones=mem.dones.at[task_idx, indices].set(dones_all),
        ptrs=mem.ptrs.at[task_idx].set((current_ptr + b) % mem.max_size_per_task),
        sizes=mem.sizes.at[task_idx].set(jnp.minimum(current_size + b, mem.max_size_per_task)),
    )


def sample_task_slot(mem: AGEMMemory, task_idx: int, sample_size: int, rng):
    """Sample uniformly from a single task's circular buffer slot."""
    task_size = mem.sizes[task_idx]
    idx = jax.random.randint(rng, (sample_size,), 0, jnp.maximum(task_size, 1))
    idx = jnp.minimum(idx, jnp.maximum(task_size - 1, 0))
    return (
        mem.obs[task_idx, idx],
        mem.actions[task_idx, idx],
        mem.log_probs[task_idx, idx],
        mem.advantages[task_idx, idx],
        mem.targets[task_idx, idx],
        mem.values[task_idx, idx],
    )


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


def update_agem_memory(agem_sample_size, env_idx, advantages, mem, rng, targets, traj_batch):
    """
    Update AGEM memory using simple circular buffer per task.

    Args:
        mem: Current memory state
        task_idx: Index of the current task (0-indexed)
        new_obs, new_actions, etc.: New data to add to the task's buffer
    """

    rng, mem_rng = jax.random.split(rng)
    perm = jax.random.permutation(mem_rng, advantages.shape[0])  # length = traj_len
    idx = perm[: agem_sample_size]
    new_obs = traj_batch.obs[idx].reshape(-1, traj_batch.obs.shape[-1])
    new_actions = traj_batch.action[idx].reshape(-1)
    new_log_probs = traj_batch.log_prob[idx].reshape(-1)
    new_adv = advantages[idx].reshape(-1)
    new_tgt = targets[idx].reshape(-1)
    new_val = traj_batch.value[idx].reshape(-1)

    b = new_obs.shape[0]  # batch size to insert

    # Ensure task_idx is within bounds
    task_idx = jnp.clip(env_idx, 0, mem.max_tasks - 1)

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

    mem = mem.replace(
        obs=updated_obs,
        actions=updated_actions,
        log_probs=updated_log_probs,
        advantages=updated_advantages,
        targets=updated_targets,
        values=updated_values,
        ptrs=updated_ptrs,
        sizes=updated_sizes
    )
    return mem, rng
