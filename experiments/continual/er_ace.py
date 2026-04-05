import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from experiments.continual.agem import AGEMMemory, sample_task_slot
from experiments.continual.base import RegCLMethod


class ERACE(RegCLMethod):
    """
    ER-ACE (Experience Replay with Asymmetric Cross-Entropy) continual learning method.

    Adds a per-task behavioural-cloning (BC) loss on stored past-task samples directly
    to the PPO loss, without gradient projection.  Simpler than AGEM and often more
    stable.

    Memory management reuses the AGEMMemory buffer and update functions.
    """

    def __init__(self, memory_size: int = 100_000, sample_size: int = 1024):
        self.memory_size = memory_size
        self.sample_size = sample_size

    def update_state(self, state, params, importance):
        return state

    def penalty(self, params, state, reg_coef):
        return 0.0

    def make_importance_fn(self, reset_switch, step_switch, network, agents, use_cnn: bool,
                           max_episodes: int, max_steps: int, norm_importance: bool,
                           stride: int) -> callable:
        @jax.jit
        def importance_fn(params: FrozenDict, env_idx: jnp.int32, rng):
            return jax.tree.map(jnp.zeros_like, params)
        return importance_fn


def compute_er_ace_gradient(network, params, mem: AGEMMemory, sample_size: int,
                             rng, past_sizes):
    """
    Loop over all task slots (like AGEM), sample from each past task, and compute
    a BC gradient -E[log π_t(a|s)] using the CORRECT task head (env_idx=t).

    Summed gradients are masked so only tasks with stored data contribute.
    Returns zero gradients when no past data exists (first task).
    """
    max_tasks = mem.obs.shape[0]       # static Python int (shapes are concrete at trace time)
    samples_per_task = max(sample_size // max_tasks, 1)

    total_grads = None
    for t in range(max_tasks):   # unrolled at JAX trace time
        rng, task_rng = jax.random.split(rng)
        t_obs, t_actions, _, _, _, _ = sample_task_slot(mem, t, samples_per_task, task_rng)
        t_obs     = jax.lax.stop_gradient(t_obs)
        t_actions = jax.lax.stop_gradient(t_actions)

        def bc_loss(p, obs=t_obs, acts=t_actions, task=t):
            pi, _, _ = network.apply(p, obs, env_idx=task)
            return -jnp.mean(pi.log_prob(acts))

        t_grads = jax.grad(bc_loss)(params)
        mask = (past_sizes[t] > 0).astype(jnp.float32)
        t_grads = jax.tree_util.tree_map(lambda g: g * mask, t_grads)
        total_grads = (
            t_grads if total_grads is None
            else jax.tree_util.tree_map(lambda a, b: a + b, total_grads, t_grads)
        )

    # Zero out gradients when there is no past data (first task).
    total_grads = jax.lax.cond(
        jnp.sum(past_sizes) > 0,
        lambda: total_grads,
        lambda: jax.tree_util.tree_map(jnp.zeros_like, total_grads),
    )

    stats = {"er_ace/total_past_samples": jnp.sum(past_sizes).astype(jnp.float32)}
    return total_grads, stats
