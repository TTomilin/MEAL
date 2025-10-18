import jax.numpy as jnp
from gymnax.environments import spaces


# Apply observation padding wrapper to environments that need it
class PadObsToMax:
    """Wrapper that pads observations to consistent dimensions"""

    def __init__(self, env, max_h: int, max_w: int):
        self.env = env
        self._max_height = int(max_h)
        self._max_width = int(max_w)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _pad_obs_dict(self, obs: dict) -> dict:
        # infer channels from first agent
        any_key = next(iter(obs))
        C = int(obs[any_key].shape[-1])
        target = (self._max_height, self._max_width, C)
        return {k: _pad_to(v, target) for k, v in obs.items()}

    def reset(self, key):
        obs, state = self.env.reset(key)
        return self._pad_obs_dict(obs), state

    def step(self, key, state, actions):
        obs, state, rew, done, info = self.env.step(key, state, actions)
        obs = self._pad_obs_dict(obs)
        return obs, state, rew, done, info

    # Make spaces consistent with padded shape (helps some code paths)
    def observation_space(self):
        # use underlying channels, override H,W
        box = self.env.observation_space()
        high = getattr(box, "high", 255)
        low = getattr(box, "low", 0)
        C = int(box.shape[-1])
        return spaces.Box(low, high, (self._max_height, self._max_width, C))


# -------------------------------------------------------------------
# helper: pad (or crop) an (H,W,C) grid to `target_shape`
# -------------------------------------------------------------------
def _pad_to(grid: jnp.ndarray, target_shape):
    th, tw, tc = target_shape  # target (height, width, channels)
    h, w, c = grid.shape  # current shape – *Python* ints
    assert c == tc, "channel mismatch"

    dh = th - h  # + ⇒ need pad, − ⇒ need crop
    dw = tw - w

    # amounts have to be Python ints so jnp.pad sees concrete values
    pad_top = max(dh // 2, 0)
    pad_bottom = max(dh - pad_top, 0)
    pad_left = max(dw // 2, 0)
    pad_right = max(dw - pad_left, 0)

    grid = jnp.pad(grid, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")

    # If the grid was *larger* than the target we crop back
    return grid[:th, :tw, :]
