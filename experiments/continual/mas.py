import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from experiments.continual.base import RegCLMethod, RegCLState
from experiments.utils import build_reg_weights, unbatchify, batchify


def _tree_mean_abs(t):
    total = jax.tree_util.tree_reduce(lambda a, x: a + jnp.sum(jnp.abs(x)), t, 0.0)
    count = jax.tree_util.tree_reduce(lambda a, x: a + x.size, t, 0)
    return total / (count + 1e-8)

class MAS(RegCLMethod):
    """
    Memory-Aware Synapses with online (EMA) accumulation to avoid ω blow-up.
    mode: "online" (EMA), "multi" (sum), or "last" (replace).
    """
    name = "mas"

    def __init__(self, mode: str = "online", decay: float = 0.9, normalize_task: bool = True):
        assert mode in {"online", "multi", "last"}
        self.mode = mode
        self.decay = decay
        self.normalize_task = normalize_task

    def init_state(self, params: FrozenDict, regularize_critic: bool, regularize_heads: bool) -> RegCLState:
        mask = build_reg_weights(params, regularize_critic, regularize_heads)
        zeros = jax.tree.map(jnp.zeros_like, params)
        return RegCLState(old_params=jax.tree.map(lambda x: x.copy(), params),
                       importance=zeros, mask=mask)

    def update_state(self, cl_state: RegCLState, new_params: FrozenDict, new_importance: FrozenDict) -> RegCLState:
        ω_old = cl_state.importance
        ω_new = new_importance
        if self.normalize_task:
            m = _tree_mean_abs(ω_new)
            ω_new = jax.tree_util.tree_map(lambda x: x / (m + 1e-8), ω_new)

        if self.mode == "online":
            ω = jax.tree_util.tree_map(lambda a, b: self.decay * a + (1.0 - self.decay) * b, ω_old, ω_new)
        elif self.mode == "multi":
            ω = jax.tree_util.tree_map(jnp.add, ω_old, ω_new)
        else:  # "last"
            ω = ω_new

        return RegCLState(old_params=new_params, importance=ω, mask=cl_state.mask)

    def penalty(self, params: FrozenDict, cl_state: RegCLState, coef: float) -> jnp.ndarray:
        def _term(p, o, w, m): return m * w * (p - o) ** 2
        tot = jax.tree_util.tree_map(_term, params, cl_state.old_params, cl_state.importance, cl_state.mask)
        tot = jax.tree_util.tree_reduce(lambda a, b: a + b.sum(), tot, 0.0)
        denom = jax.tree_util.tree_reduce(lambda a, b: a + b.sum(), cl_state.mask, 0.0) + 1e-8
        return 0.5 * coef * tot / denom

    def make_importance_fn(self, reset_switch, step_switch, network, agents, use_cnn: bool,
                           max_episodes: int, max_steps: int, norm_importance: bool, stride: int):
        num_agents = len(agents)

        @jax.jit
        def mas_importance(params, env_idx: jnp.int32, rng):
            importance0 = jax.tree_util.tree_map(jnp.zeros_like, params)
            def one_episode(carry, _):
                imp_acc, total_steps, rng = carry
                rng, r = jax.random.split(rng)
                obs, state = reset_switch(r, env_idx)
                done = jnp.array(0.0, jnp.float32)

                def one_step(carry, t):
                    imp_acc, state, obs, rng, done, steps = carry
                    # batchify once
                    obs_b = batchify(obs, agents, num_agents, not use_cnn)

                    def l2_loss(p):
                        pi, v, _ = network.apply(p, obs_b, env_idx=env_idx)
                        v = v.reshape(num_agents, 1)
                        vec = jnp.concatenate([pi.logits, v], axis=-1)
                        return 0.5 * jnp.sum(vec * vec) / vec.shape[0]

                    grads = jax.grad(l2_loss)(params)
                    alpha = (t % stride == 0).astype(jnp.float32)
                    factor = (1.0 - done) * alpha
                    grads2 = jax.tree_util.tree_map(lambda g: (g * g) * factor, grads)
                    imp_acc = jax.tree_util.tree_map(jnp.add, imp_acc, grads2)

                    rng, s1, s2 = jax.random.split(rng, 3)
                    pi, _, _ = network.apply(params, obs_b, env_idx=env_idx)
                    action = pi.sample(seed=s1)
                    env_act = unbatchify(action, agents, 1, num_agents)
                    env_act = {k: v.flatten() for k, v in env_act.items()}

                    obs2, state2, _r, done_d, _info = step_switch(s2, state, env_act, env_idx)
                    done2 = done_d["__all__"].astype(jnp.float32)

                    steps = steps + (1.0 - done)
                    done = jnp.maximum(done, done2)
                    return (imp_acc, state2, obs2, rng, done, steps), None

                (imp_acc, state, obs, rng, done, steps), _ = jax.lax.scan(
                    one_step,
                    (imp_acc, state, obs, rng, done, jnp.array(0.0, jnp.float32)),
                    xs=jnp.arange(max_steps)
                )
                return (imp_acc, total_steps + steps, rng), None

            (ω_acc, total_steps, rng), _ = jax.lax.scan(
                one_episode,
                (importance0, jnp.array(0.0, jnp.float32), rng),
                xs=jnp.arange(max_episodes)
            )

            ω_acc = jax.tree_util.tree_map(lambda x: x / (total_steps + 1e-8), ω_acc)
            if norm_importance:
                mean_abs = _tree_mean_abs(ω_acc)
                ω_acc = jax.tree_util.tree_map(lambda x: x / (mean_abs + 1e-8), ω_acc)
            return ω_acc
        return mas_importance
