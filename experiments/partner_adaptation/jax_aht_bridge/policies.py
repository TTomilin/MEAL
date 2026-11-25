from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

import experiments.partner_adaptation.jax_aht_bridge  # noqa: F401
from agents.mlp_actor_critic_agent import ActorWithConditionalCriticPolicy as AhtConditionalCriticPolicy
from agents.mlp_actor_critic_agent import MLPActorCriticPolicy as AhtMLPPolicy
from experiments.partner_adaptation.jax_aht_bridge import compat
from experiments.partner_adaptation.partner_agents.agent_interface import AgentPolicy

ACTOR_TYPES = ("mlp", "actor_with_conditional_critic")


def make_jaxaht_policy(actor_type: str, obs_dim: int, action_dim: int = 6, pop_size: int = 1,
                       activation: str = "tanh", fc_hidden_dim: int = 64):
    if actor_type == "mlp":
        return AhtMLPPolicy(action_dim, obs_dim, activation=activation, fc_hidden_dim=fc_hidden_dim)
    if actor_type == "actor_with_conditional_critic":
        return AhtConditionalCriticPolicy(action_dim, obs_dim, pop_size, activation=activation,
                                          fc_hidden_dim=fc_hidden_dim)
    raise ValueError(f"unsupported jax-aht actor_type {actor_type!r}; supported: {ACTOR_TYPES}")


class JaxMarlCompatPolicy(AgentPolicy):

    def __init__(self, inner, height: int, width: int, test_mode: bool):
        if not isinstance(inner, (AhtMLPPolicy, AhtConditionalCriticPolicy)):
            raise TypeError(
                f"JaxMarlCompatPolicy only supports feed-forward jax-aht policies "
                f"(MLPActorCriticPolicy, ActorWithConditionalCriticPolicy), got {type(inner).__name__}; "
                f"recurrent policies require hidden-state forwarding")
        super().__init__(inner.action_dim, inner.obs_dim)
        self.inner = inner
        self.height, self.width = int(height), int(width)
        self.test_mode = bool(test_mode)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng, aux_obs=None, env_state=None,
                   test_mode=False, env_id_idx=0):
        obs_jm = compat.meal_obs_to_jaxmarl(obs, self.height, self.width)
        avail = jnp.ones(obs.shape[:-1] + (self.action_dim,)) if avail_actions is None else avail_actions
        avail_jm = compat.avail_meal_to_jaxmarl(avail)
        if self.test_mode:
            action, new_hstate = self.inner.get_action(params, obs_jm, done, avail_jm, hstate, rng, test_mode=True)
            return compat.jaxmarl_action_to_meal(action), new_hstate
        if isinstance(self.inner, AhtConditionalCriticPolicy):
            pi, _ = self.inner.network.apply(
                params, (obs_jm, jnp.zeros(obs_jm.shape[:-1] + (self.inner.pop_size,)), avail_jm))
        else:
            pi, _ = self.inner.network.apply(params, (obs_jm, avail_jm))
        noise = jax.lax.optimization_barrier(jax.random.gumbel(rng, pi.logits.shape, pi.logits.dtype))
        action = jnp.argmax(pi.logits + noise, axis=-1)
        return compat.jaxmarl_action_to_meal(action), None

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng, aux_obs=None,
                                env_state=None):
        raise NotImplementedError("imported jax-aht partners are frozen; only get_action is supported")

    def init_hstate(self, batch_size: int, aux_info=None):
        return None

    def init_params(self, rng):
        return self.inner.init_params(rng)
