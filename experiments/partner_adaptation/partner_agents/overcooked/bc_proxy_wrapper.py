from functools import partial

import jax
import jax.numpy as jnp

import experiments.partner_adaptation.jax_aht_bridge  # noqa: F401
from agents.overcooked.bc_agent import ACTION_DIM, BCHState, BCModel, BCPolicy, BCPolicyHState
from agents.overcooked.bc_featurizer import BCFeaturizer
from jaxmarl.environments.overcooked.layouts import overcooked_layouts as jaxmarl_layouts
from experiments.partner_adaptation.jax_aht_bridge import compat
from experiments.partner_adaptation.partner_agents.agent_interface import AgentPolicy

BC_MAX_STEPS = 400  # BCFeaturizer horizon used by jax-aht
URGENCY_CHANNEL = 25


def _unwrap(env_state):
    while hasattr(env_state, "env_state"):
        env_state = env_state.env_state
    return env_state


class BCProxyPolicy(AgentPolicy):
    needs_env_state = True

    def __init__(self, layout_name: str, height: int, width: int, num_pots: int, num_goals: int,
                 test_mode: bool = True, unblock_if_stuck: bool = True):
        super().__init__(ACTION_DIM, height * width * 26)
        self.height, self.width = int(height), int(width)
        self.num_pots, self.num_goals = int(num_pots), int(num_goals)
        self.test_mode = bool(test_mode)
        self.unblock_if_stuck = bool(unblock_if_stuck)
        self.network = BCModel(action_dim=ACTION_DIM, hidden_dims=(64, 64))
        self.featurizer = BCFeaturizer(jaxmarl_layouts[layout_name], num_pots=2, max_steps=BC_MAX_STEPS)

    def _seat(self, obs, state):
        # lookup instead of argmax: a batched u8 argmax takes ptxas tens of minutes to compile
        grid = obs.reshape(self.height, self.width, -1)
        return jnp.where(grid[state.agent_pos[0, 1], state.agent_pos[0, 0], 0] > 0, 0, 1), grid

    def init_hstate(self, batch_size=1, aux_info=None):
        agent_id = (aux_info or {}).get("agent_id", 1)
        if self.unblock_if_stuck:
            return BCPolicyHState.init_empty(agent_id).to_numpy()
        return BCPolicyHState(agent_id=jnp.asarray(agent_id), bc_state=jnp.array([])).to_numpy()

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng, aux_obs=None, env_state=None,
                   test_mode=False, env_id_idx=0):
        if env_state is None:
            raise ValueError("BCProxyPolicy needs env_state (it featurises the full state)")
        state = _unwrap(env_state)
        seat, grid = self._seat(obs, state)
        jm = compat.meal_state_to_jaxmarl(state, self.num_pots, self.num_goals)
        urgent = grid[..., URGENCY_CHANNEL].max() > 0
        jm = jm.replace(time=jnp.where(urgent, BC_MAX_STEPS - 1, 0))
        feat_obs = self.featurizer.featurize_state_as_array(jm)[seat]
        action_probs = jax.nn.softmax(self.network.apply({"params": params}, feat_obs))

        hstate = jnp.where(jnp.asarray(done).reshape(()), self.init_hstate(1, {"agent_id": seat}), hstate)
        if self.unblock_if_stuck:
            bc_hstate = BCHState.from_numpy(BCPolicyHState.from_numpy(hstate).bc_state)
            unstuck = BCPolicy._remove_indices_and_renormalize(action_probs, bc_hstate.actions)
            action_probs = jnp.where(bc_hstate.is_stuck(feat_obs), unstuck, action_probs)
        action = jnp.argmax(action_probs) if self.test_mode else jax.random.choice(rng, ACTION_DIM, p=action_probs)
        if self.unblock_if_stuck:
            hstate = BCPolicyHState(agent_id=seat, bc_state=bc_hstate.append(action, feat_obs).to_numpy()).to_numpy()
        return compat.jaxmarl_action_to_meal(action), hstate

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng, aux_obs=None,
                                env_state=None):
        raise NotImplementedError("the BC human proxy is frozen; only get_action is supported")

    def init_params(self, rng):
        return self.network.init(rng, jnp.zeros((self.featurizer._calculate_feature_size(),)))["params"]
