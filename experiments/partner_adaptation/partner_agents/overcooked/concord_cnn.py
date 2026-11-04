from functools import partial
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from experiments.partner_adaptation.partner_agents.agent_interface import AgentPolicy
from experiments.partner_adaptation.partner_agents.overcooked.concord_featurizer import NUM_CHANNELS, concord_featurize
from meal.env.overcooked.common import DIR_TO_VEC as MEAL_DIR_TO_VEC
from meal.env.overcooked.overcooked_env import Actions

CONV_CHANNELS = (32, 64, 32)
HIDDEN = 64
NUM_ACTIONS = 6
LAYER_NORM_EPS = 1e-5  # torch.nn.LayerNorm default
# Overcooked-AI Action.INDEX_TO_ACTION order: N, S, E, W, STAY, INTERACT
_OAI_MOVES = np.array([(0, -1), (0, 1), (1, 0), (-1, 0)])


def _oai_to_meal_actions() -> np.ndarray:
    meal = np.asarray(MEAL_DIR_TO_VEC).astype(np.int64)
    table = [int(np.flatnonzero((meal == v).all(axis=1))[0]) for v in _OAI_MOVES]
    return np.array(table + [int(Actions.stay), int(Actions.interact)], dtype=np.int32)


OVERCOOKED_AI_ACTION_TO_MEAL = _oai_to_meal_actions()


# means via matmul: XLA's fused LayerNorm reduction takes ptxas tens of minutes at 16384 envs (A100, jax 0.10.2)
class MatmulLayerNorm(nn.Module):
    epsilon: float = LAYER_NORM_EPS

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        scale = self.param("scale", nn.initializers.ones, (d,))
        bias = self.param("bias", nn.initializers.zeros, (d,))
        avg = jnp.full((d, d), 1.0 / d, x.dtype)
        prec = jax.lax.Precision.HIGHEST
        mean = jnp.dot(x, avg, precision=prec)
        c = x - mean
        var = jnp.dot(c * c, avg, precision=prec)
        return c * jax.lax.rsqrt(var + self.epsilon) * scale + bias


class ConcordActor(nn.Module):
    @nn.compact
    def __call__(self, x):
        lead = x.shape[:-3]
        x = x.reshape((-1,) + x.shape[-3:])
        for i, c in enumerate(CONV_CHANNELS):
            x = nn.relu(nn.Conv(c, (3, 3), padding=((1, 1), (1, 1)), name=f"conv_{i}")(x))
        x = jnp.transpose(x, (0, 3, 1, 2)).reshape(x.shape[0], -1)  # match torch Flatten order (C, W, H)
        for i in range(3):
            x = nn.relu(nn.Dense(HIDDEN, name=f"fc_{i}")(x))
            x = MatmulLayerNorm(name=f"ln_{i}")(x)
        logits = nn.Dense(NUM_ACTIONS, name="action_out")(x)
        return logits.reshape(lead + (NUM_ACTIONS,))


def torch_state_dict_to_flax(sd: Dict[str, np.ndarray]) -> Dict:
    a = {k: np.asarray(v, dtype=np.float32) for k, v in sd.items()}
    p = {}
    for i, t in enumerate((0, 2, 4)):
        p[f"conv_{i}"] = {"kernel": np.transpose(a[f"base.cnn.{t}.weight"], (2, 3, 1, 0)),
                          "bias": a[f"base.cnn.{t}.bias"]}
    for i, (fc, ln) in enumerate(((7, 9), (10, 12), (13, 15))):
        p[f"fc_{i}"] = {"kernel": a[f"base.cnn.{fc}.weight"].T, "bias": a[f"base.cnn.{fc}.bias"]}
        p[f"ln_{i}"] = {"scale": a[f"base.cnn.{ln}.weight"], "bias": a[f"base.cnn.{ln}.bias"]}
    p["action_out"] = {"kernel": a["act.action_out.linear.weight"].T, "bias": a["act.action_out.linear.bias"]}
    return {"params": p}


def _unwrap(env_state):
    while hasattr(env_state, "env_state"):
        env_state = env_state.env_state
    return env_state


# jnp.argmax as selects: XLA's batched argmax reduction takes ptxas tens of minutes (A100, jax 0.10.2)
def argmax_last(x):
    best, idx = x[..., 0], jnp.zeros(x.shape[:-1], jnp.int32)
    for i in range(1, x.shape[-1]):
        better = x[..., i] > best
        best, idx = jnp.where(better, x[..., i], best), jnp.where(better, i, idx)
    return idx


class ConcordCNNPolicy(AgentPolicy):
    needs_env_state = True

    def __init__(self, height: int, width: int, test_mode: bool = False):
        super().__init__(NUM_ACTIONS, height * width * 26)
        self.height, self.width = int(height), int(width)
        self.test_mode = bool(test_mode)
        self.network = ConcordActor()

    def _seat(self, obs, state):
        # lookup instead of argmax: a batched u8 argmax takes ptxas tens of minutes to compile
        own = obs.reshape(self.height, self.width, -1)[..., 0]
        return jnp.where(own[state.agent_pos[0, 1], state.agent_pos[0, 0]] > 0, 0, 1)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng, aux_obs=None, env_state=None,
                   test_mode=False, env_id_idx=0):
        if env_state is None:
            raise ValueError("ConcordCNNPolicy needs env_state (it featurises the full MEAL state)")
        state = _unwrap(env_state)
        seat = self._seat(obs, state)
        feats = jax.lax.switch(seat, [lambda s: concord_featurize(s, 0), lambda s: concord_featurize(s, 1)], state)
        logits = self.network.apply(params, feats)
        if not self.test_mode:  # barrier keeps threefry out of the XLA fusion (compile time)
            logits = logits + jax.lax.optimization_barrier(jax.random.gumbel(rng, logits.shape, logits.dtype))
        action = argmax_last(logits)
        return jnp.asarray(OVERCOOKED_AI_ACTION_TO_MEAL)[action], hstate

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng, aux_obs=None,
                                env_state=None):
        raise NotImplementedError("Concord partners are frozen; only get_action is supported")

    def init_params(self, rng):
        return self.network.init(rng, jnp.zeros((self.width, self.height, NUM_CHANNELS)))
