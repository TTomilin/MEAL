import functools

import jax
import jax.numpy as jnp
import distrax
from flax.core.frozen_dict import FrozenDict

from experiments.utils import build_reg_weights, _prep_obs
from experiments.continual.base import RegCLMethod, CLState


class EWC(RegCLMethod):
    """
    Diagonal Elastic-Weight-Consolidation (Kirkpatrick 2017).

    Modes
    -----
    • "last"   – keep only the Fisher from the previous task
    • "online" – running exponential average with decay λ
    • "multi"  – accumulate *sum* of Fishers (standard EWC)
    """
    name = "ewc"

    def __init__(self, mode="last", decay: float = 0.9):
        assert mode in {"last", "online", "multi"}
        self.mode = mode
        self.decay = decay

    def init_state(self, params, regularize_critic, regularize_heads) -> CLState:
        mask = build_reg_weights(params, regularize_critic=regularize_critic, regularize_heads=regularize_heads)
        return CLState(
            old_params=jax.tree.map(lambda x: x.copy(), params),
            importance=jax.tree.map(jnp.zeros_like, params),
            mask=mask
        )

    def update_state(self, cl_state: CLState, new_params: FrozenDict, new_fisher: FrozenDict) -> CLState:

        if self.mode == "last":
            fish = new_fisher

        elif self.mode == "multi":
            fish = jax.tree_util.tree_map(jnp.add, cl_state.importance, new_fisher)

        else:  # "online"
            fish = jax.tree_util.tree_map(
                lambda old_f, f_new: self.decay * old_f + (1. - self.decay) * f_new,
                cl_state.importance, new_fisher)

        return CLState(old_params=new_params, importance=fish, mask=cl_state.mask)

    def penalty(self,
                params: FrozenDict,
                cl_state: CLState,
                coef: float) -> jnp.ndarray:

        def _term(p, o, f, m):
            return m * f * (p - o) ** 2

        tot = jax.tree_util.tree_map(_term,
                                     params, cl_state.old_params,
                                     cl_state.importance, cl_state.mask)
        tot = jax.tree_util.tree_reduce(lambda a, b: a + b.sum(), tot, 0.)
        denom = jax.tree_util.tree_reduce(lambda a, b: a + b.sum(),
                                          cl_state.mask, 0.) + 1e-8
        return 0.5 * coef * tot / denom

    def compute_importance(self,
                           params: FrozenDict,
                           env,
                           net,
                           env_idx: int,
                           key: jax.random.PRNGKey,
                           use_cnn: bool = True,
                           max_episodes: int = 5,
                           max_steps: int = 500,
                           normalize_importance: bool = False):
        return compute_fisher(params, env, net, env_idx, key, use_cnn=use_cnn, max_episodes=max_episodes,
                              max_steps=max_steps, normalize_importance=normalize_importance)


@functools.partial(
    jax.jit,
    static_argnums=(1, 2, 3),
    static_argnames=(
            "use_cnn",
            "max_episodes",
            "max_steps",
            "normalize_importance",
    ),
)
def compute_fisher(params: FrozenDict,
                   env,
                   net,
                   env_idx: int,
                   key: jax.random.PRNGKey,
                   *,
                   use_cnn: bool = True,
                   max_episodes: int = 5,
                   max_steps: int = 500,
                   normalize_importance: bool = False):
    # -----------------------------------------------------------------
    # grad(log π) for a batch of agents, vectorised
    # -----------------------------------------------------------------
    def _logp(p, obs, act):
        # Ensure obs has the right shape for the network
        # If obs is from _prep_obs, it has shape (num_agents, obs_dim)
        # We need to take the first agent's observation and ensure it has batch dim 1
        if obs.ndim > 1 and obs.shape[0] > 1:
            obs = obs[0:1]  # Take first agent, keep batch dimension
        else:
            obs = jnp.expand_dims(obs, 0) if obs.ndim == 1 else obs
        net_output = net.apply(p, obs, env_idx=env_idx)  # (1, …)

        # Handle both policy networks (return distributions) and Q-networks (return Q-values)
        if isinstance(net_output, tuple):
            # Policy network case: returns (dists, values, dormant_ratio)
            dists, _, _ = net_output
        else:
            # Q-network case: returns Q-values, convert to Categorical distribution
            q_values = net_output  # (A, num_actions)
            dists = distrax.Categorical(logits=q_values)

        return jnp.sum(dists.log_prob(act))  # scalar

    batched_grad = jax.vmap(jax.grad(_logp), in_axes=(None, 0, 0))  # map over steps

    # -----------------------------------------------------------------
    # one environment step inside lax.scan
    # -----------------------------------------------------------------
    def step_fn(carry, _):
        rng, env_state, obs, fisher_acc = carry
        rng, key_sample, key_step = jax.random.split(rng, 3)

        obs_batch = _prep_obs(obs, use_cnn=use_cnn)

        # Ensure obs_batch has the right shape for the network
        # If obs_batch is from _prep_obs, it has shape (num_agents, obs_dim)
        # We need to take the first agent's observation and ensure it has batch dim 1
        if obs_batch.ndim > 1 and obs_batch.shape[0] > 1:
            obs_batch = obs_batch[0:1]  # Take first agent, keep batch dimension

        # sample & log-prob in one forward pass (batched)
        net_output = net.apply(params, obs_batch, env_idx=env_idx)

        # Handle both policy networks (return distributions) and Q-networks (return Q-values)
        if isinstance(net_output, tuple):
            # Policy network case: returns (dists, values, dormant_ratio)
            dists, _, _ = net_output
            actions = dists.sample(seed=key_sample)  # (A,)
        else:
            # Q-network case: returns Q-values, convert to Categorical distribution
            q_values = net_output  # (A, num_actions)
            dists = distrax.Categorical(logits=q_values)
            actions = dists.sample(seed=key_sample)  # (A,)

        actions = jax.lax.stop_gradient(actions)  # no grad through sample

        # grad(log π)²
        # Since we're only using the first agent, we need to handle the gradient computation differently
        # Create a single observation for gradient computation
        single_obs = obs_batch[0] if obs_batch.shape[0] > 0 else obs_batch
        single_action = actions[0] if actions.shape[0] > 0 else actions

        # Compute gradient for single observation/action pair
        g_single = jax.grad(_logp)(params, single_obs, single_action)
        g2 = jax.tree_util.tree_map(lambda x: x ** 2, g_single)

        fisher_acc = jax.tree_util.tree_map(jnp.add, fisher_acc, g2)

        # env step (batched over agents internally by env)
        # Replicate the action for all agents since we only computed one action
        action_value = actions[0] if actions.shape[0] > 0 else actions
        act_dict = { agent: action_value for agent in env.agents }
        obs_next, env_state, _, done_info, _ = env.step(key_step, env_state, act_dict)

        # Handle both single-agent (boolean) and multi-agent (dictionary) done values
        if isinstance(done_info, dict):
            done = done_info["__all__"]
        else:
            # Single-agent environment returns done as boolean directly
            done = done_info

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

    if normalize_importance:
        total_abs = jax.tree_util.tree_reduce(lambda a, x: a + jnp.sum(jnp.abs(x)),
                                              fisher_tot, 0.)
        denom = total_abs / (jax.tree_util.tree_reduce(
            lambda a, x: a + x.size, fisher_tot, 0) + 1e-8)
        fisher_tot = jax.tree_util.tree_map(lambda x: x / (denom + 1e-8),
                                            fisher_tot)

    return fisher_tot  # FrozenDict
