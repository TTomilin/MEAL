# ewc.py  ──────────────────────────────────────────────────────────────

import functools

import jax
import jax.numpy as jnp
from flax import struct
from flax.core.frozen_dict import FrozenDict

from baselines.utils import copy_params


# ──────────────────────────────────────────────────────────────────────
# CONFIG CONVENTION
#   config.ewc_mode   : "last", "multi", or "online"      (default "last")
#   config.online_lam : decay λ for online mode (float)   (default 0.9)
#   config.regularize_critic, config.regularize_heads     (bools)
# ──────────────────────────────────────────────────────────────────────


@struct.dataclass
class EWCState:
    """
    A unified state that works for all three modes.
      • 'last'   → tuples of length 1, replaced each task
      • 'multi'  → tuples grow by one per task
      • 'online' → tuples length 1, fisher merged with λ
    """
    old_params: tuple[FrozenDict, ...]
    fishers: tuple[FrozenDict, ...]
    reg_weights: FrozenDict


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _zeros_like_tree(t):
    return jax.tree_map(jnp.zeros_like, t)


def _ones_like_tree(t):
    return jax.tree_map(jnp.ones_like, t)


def build_reg_weights(params: FrozenDict,
                      regularize_critic: bool,
                      regularize_heads: bool) -> FrozenDict:
    """
    1s where we want EWC, 0s where we do not (e.g. heads, critic trunk, …).
    """

    def _assign(path, x):
        p = "/".join(str(k) for k in path).lower()
        if not regularize_heads and ("actor_head" in p or "critic_head" in p):
            return jnp.zeros_like(x)
        if not regularize_critic and "critic" in p:
            return jnp.zeros_like(x)
        return jnp.ones_like(x)

    return jax.tree_util.tree_map_with_path(_assign, params)


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------
def init_cl_state(params: FrozenDict, regularize_critic, regularize_heads) -> EWCState:
    """Call once, right after first task finishes."""
    reg_w = build_reg_weights(params, regularize_critic, regularize_heads)
    zero_fisher = _zeros_like_tree(params)

    return EWCState(
        old_params=(copy_params(params),),  # tuple length 1
        fishers=(zero_fisher,),  # idem
        reg_weights=reg_w
    )


def update_ewc_state(state: EWCState, new_params: FrozenDict, new_fisher: FrozenDict, mode: str,
                     ewc_lambda: float) -> EWCState:
    """
    Call after finishing a task *before* starting training on the next.
    """

    if mode == "multi":
        # append a fresh snapshot
        return EWCState(
            old_params=state.old_params + (copy_params(new_params),),
            fishers=state.fishers + (new_fisher,),
            reg_weights=state.reg_weights,
        )

    if mode == "online":
        merged_fisher = jax.tree_util.tree_map(
            lambda old_f, new_f: ewc_lambda * old_f + new_f,
            state.fishers[0], new_fisher,
        )
        return EWCState(
            old_params=(copy_params(new_params),),  # length 1
            fishers=(merged_fisher,),  # length 1
            reg_weights=state.reg_weights,
        )

    # default: "last"
    return EWCState(
        old_params=(copy_params(new_params),),
        fishers=(new_fisher,),
        reg_weights=state.reg_weights,
    )


@functools.partial(jax.jit, static_argnums=2)
def _ewc_single(params, snapshot, coef):
    """Penalty for one (paramŝ, Fisher̂) snapshot."""
    old_p, fisher = snapshot
    diff2 = jax.tree_util.tree_map(lambda p, op: (p - op) ** 2, params, old_p)
    term = jax.tree_util.tree_map(lambda d2, f: d2 * f, diff2, fisher)
    return 0.5 * coef * jax.tree_util.tree_reduce(lambda a, x: a + x.sum(),
                                                  term, 0.0)


def compute_ewc_loss(params: FrozenDict,
                     state: EWCState,
                     ewc_coef: float) -> float:
    """
    0.5 * λ * Σ_i F_i (θ − θ̂_i)²   summed over *masked* tasks i.
    Extremely cheap now: one JIT call per snapshot (1 for last/online).
    """
    snapshots = tuple(zip(state.old_params, state.fishers))
    penalties = tuple(_ewc_single(params, snap, ewc_coef) for snap in snapshots)
    return jnp.sum(jnp.stack(penalties))


# -------------------------------------------------------------------
# helper: pad (or crop) an (H,W,C) image to `target_shape`
#         – no tracers in pad_width, 100 % JIT-safe
# -------------------------------------------------------------------
def _pad_to(img: jnp.ndarray, target_shape):
    th, tw, tc = target_shape  # target (height, width, channels)
    h, w, c = img.shape  # current shape – *Python* ints
    assert c == tc, "channel mismatch"

    dh = th - h  # + ⇒ need pad, − ⇒ need crop
    dw = tw - w

    # amounts have to be Python ints so jnp.pad sees concrete values
    pad_top = max(dh // 2, 0)
    pad_bottom = max(dh - pad_top, 0)
    pad_left = max(dw // 2, 0)
    pad_right = max(dw - pad_left, 0)

    img = jnp.pad(
        img,
        ((pad_top, pad_bottom),
         (pad_left, pad_right),
         (0, 0)),  # no channel padding
        mode="constant",
    )

    # If the image was *larger* than the target we crop back
    return img[:th, :tw, :]


# ---------------------------------------------------------------
# util: build a (2, …) batch without Python branches
# ---------------------------------------------------------------
def _prep_obs(raw_obs, *, expected_shape, use_cnn,
              use_task_id, env_idx, seq_len):
    def _single(img):  # (H,W,C)
        if use_cnn:
            img = _pad_to(img, expected_shape)  # (H_pad,W_pad,C)
            return img[None]  # add batch dim
        else:
            vec = img.reshape(-1)  # flatten
            vec = jax.lax.cond(
                use_task_id,
                lambda z: jnp.concatenate(
                    [z, jax.nn.one_hot(env_idx, seq_len)], axis=-1),
                lambda z: z,
                operand=vec,
            )
            # pad to expected length (if needed)
            full_len = jnp.prod(expected_shape)
            pad_len = full_len - vec.shape[0]
            vec = jnp.pad(vec, (0, jnp.maximum(pad_len, 0)))
            return vec[None]  # (1, D)

    return jnp.concatenate(
        [_single(raw_obs["agent_0"]),
         _single(raw_obs["agent_1"])],
        axis=0)  # (2, …)


# ---------------------------------------------------------------


# ---------------------------------------------------------------------
# JAX Fisher estimator
# ---------------------------------------------------------------------
@functools.partial(
    jax.jit,
    static_argnums=(1, 2, 3, 4),
    static_argnames=(
            "expected_shape",
            "use_cnn",
            "use_task_id",
            "max_episodes",
            "max_steps",
            "normalize_fisher",
    ),
)
def compute_fisher(params: FrozenDict,
                   env,
                   network,
                   env_idx: int,
                   seq_length: int,
                   key: jax.random.PRNGKey,
                   *,
                   expected_shape: tuple,
                   use_cnn: bool = True,
                   use_task_id: bool = False,
                   max_episodes: int = 5,
                   max_steps: int = 500,
                   normalize_fisher: bool = False):
    # -----------------------------------------------------------------
    # grad(log π) for a batch of agents, vectorised
    # -----------------------------------------------------------------
    def _logp(p, obs, act):
        obs = jnp.expand_dims(obs, 0)
        dists, _ = network.apply(p, obs, env_idx=env_idx)  # (A, …)
        return jnp.sum(dists.log_prob(act))  # scalar

    batched_grad = jax.vmap(jax.grad(_logp), in_axes=(None, 0, 0))  # map over steps

    # -----------------------------------------------------------------
    # one environment step inside lax.scan
    # -----------------------------------------------------------------
    def step_fn(carry, _):
        rng, env_state, obs, fisher_acc = carry
        rng, key_sample, key_step = jax.random.split(rng, 3)

        obs_batch = _prep_obs(obs,
                              expected_shape=expected_shape,
                              use_cnn=use_cnn,
                              use_task_id=use_task_id,
                              env_idx=env_idx,
                              seq_len=seq_length)  # (A, obs_dim)

        # sample & log-prob in one forward pass (batched)
        dists, _ = network.apply(params, obs_batch, env_idx=env_idx)
        actions = dists.sample(seed=key_sample)  # (A,)
        actions = jax.lax.stop_gradient(actions)  # no grad through sample

        # grad(log π)²
        g = batched_grad(params, obs_batch, actions)  # pytree with leading dim = A
        g2 = jax.tree_util.tree_map(lambda x: jnp.mean(x ** 2, axis=0), g)

        fisher_acc = jax.tree_util.tree_map(jnp.add, fisher_acc, g2)

        # env step (batched over agents internally by env)
        act_dict = {"agent_0": actions[0], "agent_1": actions[1]}
        obs_next, env_state, _, done_info, _ = env.step(key_step, env_state, act_dict)
        done = done_info["__all__"]

        return (rng, env_state, obs_next, fisher_acc), done

    # -----------------------------------------------------------------
    # run one episode under lax.while_loop
    # -----------------------------------------------------------------
    def run_episode(carry, key_ep):
        params, fisher_global = carry
        key_ep, key_reset = jax.random.split(key_ep)
        obs0, env_state0 = env.reset(key_reset)

        fisher_zero = jax.tree_util.tree_map(jnp.zeros_like, fisher_global)

        ep_carry = (key_ep, env_state0, obs0, fisher_zero)
        ep_carry, _ = jax.lax.scan(step_fn,
                                   ep_carry,
                                   xs=None,
                                   length=max_steps)

        fisher_ep = ep_carry[-1]
        fisher_global = jax.tree_util.tree_map(jnp.add, fisher_global, fisher_ep)
        return (params, fisher_global), None

    # -----------------------------------------------------------------
    # run many episodes under lax.scan
    # -----------------------------------------------------------------
    fisher_global_init = jax.tree_util.tree_map(jnp.zeros_like, params)
    (_, fisher_tot), _ = jax.lax.scan(run_episode,
                                      (params, fisher_global_init),
                                      jax.random.split(key, max_episodes))

    # -----------------------------------------------------------------
    # average & optional normalisation
    # -----------------------------------------------------------------
    fisher_tot = jax.tree_util.tree_map(
        lambda x: x / (max_episodes * max_steps), fisher_tot)

    if normalize_fisher:
        total_abs = jax.tree_util.tree_reduce(lambda a, x: a + jnp.sum(jnp.abs(x)),
                                              fisher_tot, 0.)
        denom = total_abs / (jax.tree_util.tree_reduce(
            lambda a, x: a + x.size, fisher_tot, 0) + 1e-8)
        fisher_tot = jax.tree_util.tree_map(lambda x: x / (denom + 1e-8),
                                            fisher_tot)

    return fisher_tot  # FrozenDict
