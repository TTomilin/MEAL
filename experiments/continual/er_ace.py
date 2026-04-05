import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from experiments.continual.agem import AGEMMemory, sample_memory
from experiments.continual.base import RegCLMethod


class ERACE(RegCLMethod):
    """
    ER-ACE (Experience Replay with Asymmetric Cross-Entropy) continual learning method.

    Adds a behavioural-cloning loss on stored past-task samples directly to the PPO
    loss, without gradient projection.  Simpler than AGEM and often more stable.

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
                             rng, past_sizes, env_idx: int = 0):
    """
    Sample from all past tasks (current task masked out) and compute a
    behavioural-cloning (BC) gradient: -E[log π(a_mem | s_mem)].

    Returns the raw gradient (not yet scaled by er_ace_coef) and a stats dict.
    If no past data exists, returns zero gradients.
    """
    # Build a temporary memory view that zeroes the current task's size so
    # sample_memory only draws from previously seen tasks.
    mem_past = mem.replace(sizes=past_sizes)

    total_past = jnp.sum(past_sizes)

    mem_obs, mem_actions, _, _, _, _ = sample_memory(mem_past, sample_size, rng)
    mem_obs = jax.lax.stop_gradient(mem_obs)
    mem_actions = jax.lax.stop_gradient(mem_actions)

    def bc_loss_fn(p):
        pi, _, _ = network.apply(p, mem_obs, env_idx=env_idx)
        return -jnp.mean(pi.log_prob(mem_actions))

    grads = jax.grad(bc_loss_fn)(params)

    # Zero out the gradient when there is no past data so the first task is
    # unaffected (avoids learning from the zeroed-out dummy samples).
    grads = jax.lax.cond(
        total_past > 0,
        lambda: grads,
        lambda: jax.tree_util.tree_map(jnp.zeros_like, grads),
    )

    stats = {
        "er_ace/total_past_samples": total_past,
    }
    return grads, stats
