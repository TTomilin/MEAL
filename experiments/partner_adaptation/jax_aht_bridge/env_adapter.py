from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from scipy.ndimage import label

import experiments.partner_adaptation.jax_aht_bridge  # noqa: F401
from envs.base_env import BaseEnv, WrappedEnvState
from jaxmarl.environments import spaces
from experiments.partner_adaptation.jax_aht_bridge import compat
from meal.env.overcooked.common import DIR_TO_VEC, OBJECT_TO_INDEX
from meal.env.overcooked.layouts.presets import overcooked_layouts
from meal.env.overcooked.overcooked_env import Actions, Overcooked

# OvercookedV1.BASE_REW_SHAPING_PARAMS
JAXAHT_BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": [3, 3],
    "PLATE_PICKUP_REWARD": [3, 3],
    "SOUP_PICKUP_REWARD": [5, 5],
    "ONION_PICKUP_REWARD": [0, 0],
    "COUNTER_PICKUP_REWARD": [0, 0],
    "COUNTER_DROP_REWARD": [0, 0],
    "ONION_HOLDING": [0, 0],
    "PLATE_HOLDING": [0, 0],
    "DISH_DISP_DISTANCE_REW": [0, 0],
    "POT_DISTANCE_REW": [0, 0],
    "SOUP_DISTANCE_REW": [0, 0],
}
_UNSUPPORTED_SHAPING = ("DISH_DISP_DISTANCE_REW", "POT_DISTANCE_REW", "SOUP_DISTANCE_REW")
_JAXAHT_POT_EMPTY_STATUS = 23  # OvercookedV1 samples pot status in [0, 24)


def _symmetric_shaping(params) -> Dict[str, np.ndarray]:
    merged = {k: list(v) for k, v in JAXAHT_BASE_REW_SHAPING_PARAMS.items()}
    for key, value in dict(params or {}).items():
        if key not in merged:
            raise ValueError(f"unknown reward shaping parameter {key!r}")
        if isinstance(value, (int, float)):
            value = [value, value]
        merged[key] = [float(v) for v in value]
    for key in _UNSUPPORTED_SHAPING:
        if any(v != 0 for v in merged[key]):
            raise ValueError(f"{key} is not implemented by jax-aht's OvercookedV1 and must be 0")
    return {k: np.asarray(v, dtype=np.float32) for k, v in merged.items()}


def _resolve_layout(layout) -> Tuple[FrozenDict, str]:
    if isinstance(layout, str):
        if layout not in overcooked_layouts:
            raise ValueError(f"Unknown MEAL layout {layout!r}")
        return FrozenDict(overcooked_layouts[layout]), layout
    return FrozenDict(layout), "custom"


class AhtDynamicsOvercooked(Overcooked):

    def _resolve_collisions(self, current, proposed):
        assert current.shape[0] == 2
        collision = jnp.all(proposed[0] == proposed[1])
        swap = jnp.all(proposed[0] == current[1]) & jnp.all(proposed[1] == current[0])
        return jnp.where(collision | swap, current, proposed).astype(jnp.uint32)


class MealOvercookedAHT(BaseEnv):

    def __init__(self, layout="cramped_room", random_reset: bool = True, random_obj_state: bool = False,
                 max_steps: int = 400, do_reward_shaping: bool = False, reward_shaping_params=None,
                 **meal_kwargs):
        layout_dict, self.layout_name = _resolve_layout(layout)
        self.env = AhtDynamicsOvercooked(layout=layout_dict, layout_name=self.layout_name, random_reset=False,
                              max_steps=max_steps, **meal_kwargs)
        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        assert self.num_agents == 2, "jax-aht's Overcooked interface is two-player"
        self.random_reset = bool(random_reset)
        self.random_obj_state = bool(random_obj_state)
        self.max_steps = max_steps
        self.do_reward_shaping = bool(do_reward_shaping)
        self.reward_shaping_params = _symmetric_shaping(reward_shaping_params)
        self.height, self.width = self.env.height, self.env.width
        self.num_pots = int(len(layout_dict["pot_idx"]))
        self.num_goals = int(len(layout_dict["goal_idx"]))
        self.agent_view_size = compat.JAXMARL_AGENT_VIEW_SIZE

        wall = np.zeros(self.height * self.width, dtype=bool)
        wall[np.asarray(layout_dict["wall_idx"])] = True
        components, num_components = label(~wall.reshape(self.height, self.width))
        self.free_space_map = jnp.asarray(components.reshape(-1), dtype=jnp.int32)
        self.num_components = int(num_components)
        self.free_mask = jnp.asarray(~wall, dtype=jnp.float32)

        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}

    def observation_space(self, agent: str = "") -> spaces.Box:
        return spaces.Box(0, 255, (self.height * self.width * self.env.obs_channels,))

    def action_space(self, agent: str = "") -> spaces.Discrete:
        return spaces.Discrete(self.env.num_actions)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        return {agent: jnp.ones(self.env.num_actions) for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.ndarray:
        return state.env_state.time

    def _flat_obs(self, obs: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
        return {a: compat.meal_obs_to_jaxmarl(obs[a]).flatten() for a in self.agents}

    def _sample_agent_idx(self, key: chex.PRNGKey) -> jnp.ndarray:
        cells = jnp.arange(self.height * self.width)
        if self.num_components >= 2:
            key, sub = jax.random.split(key)
            comps = jax.random.choice(sub, jnp.arange(1, self.num_components + 1), shape=(2,), replace=False)
            idx = []
            for i in range(2):
                key, sub = jax.random.split(key)
                mask = (self.free_space_map == comps[i]).astype(jnp.float32)
                idx.append(jax.random.choice(sub, cells, p=mask / mask.sum()))
            return jnp.stack(idx).astype(jnp.uint32)
        return jax.random.choice(key, cells, shape=(2,), p=self.free_mask / self.free_mask.sum(),
                                 replace=False).astype(jnp.uint32)

    def _randomise(self, key: chex.PRNGKey, state):
        k_pos, k_pot, k_inv = jax.random.split(key, 3)
        maze = state.maze_map
        if self.random_reset:
            new_idx = self._sample_agent_idx(k_pos)
            new_pos = jnp.stack([new_idx % self.width, new_idx // self.width], axis=1).astype(jnp.uint32)
            empty = jnp.array([OBJECT_TO_INDEX["empty"], 0, 0], dtype=maze.dtype)
            agent_tiles = maze[state.agent_pos[:, 1], state.agent_pos[:, 0], :]
            maze = maze.at[state.agent_pos[:, 1], state.agent_pos[:, 0], :].set(empty)
            maze = maze.at[new_pos[:, 1], new_pos[:, 0], :].set(agent_tiles)
            state = state.replace(agent_pos=new_pos)
        if self.random_obj_state:
            status = jax.random.randint(k_pot, (state.pot_pos.shape[0],), 0, _JAXAHT_POT_EMPTY_STATUS + 1)
            status = jnp.where(state.pot_mask, status, maze[state.pot_pos[:, 1], state.pot_pos[:, 0], 2])
            maze = maze.at[state.pot_pos[:, 1], state.pot_pos[:, 0], 2].set(status.astype(maze.dtype))
            items = jnp.array([OBJECT_TO_INDEX["empty"], OBJECT_TO_INDEX["onion"], OBJECT_TO_INDEX["plate"],
                               OBJECT_TO_INDEX["dish"]])
            state = state.replace(agent_inv=jax.random.choice(k_inv, items, shape=(self.num_agents,)))
        return state.replace(maze_map=maze)

    def _shaped_rewards(self, prev, actions_meal: jnp.ndarray, maze_after_own_turn, new_inv) -> jnp.ndarray:
        p = {k: jnp.asarray(v) for k, v in self.reward_shaping_params.items()}
        maze = prev.maze_map
        inv = prev.agent_inv
        out = []
        for i in range(self.num_agents):
            interact = actions_meal[i] == Actions.interact
            fwd = (prev.agent_pos[i].astype(jnp.int32) + DIR_TO_VEC[prev.agent_dir_idx[i]].astype(jnp.int32))
            fwd = jnp.clip(fwd, 0, jnp.array([self.width - 1, self.height - 1]))
            tile = maze[fwd[1], fwd[0], 0]
            status = maze[fwd[1], fwd[0], 2]
            inv_pre, inv_post = inv[i], new_inv[i]
            is_pot = tile == OBJECT_TO_INDEX["pot"]
            holding_onion = inv_pre == OBJECT_TO_INDEX["onion"]
            holding_plate = inv_pre == OBJECT_TO_INDEX["plate"]
            placement = is_pot & holding_onion & (status > self.env.cook_time)
            soup_pickup = is_pot & holding_plate & (status == 0)
            pickup = (inv_pre == OBJECT_TO_INDEX["empty"]) & (inv_post != OBJECT_TO_INDEX["empty"]) & ~is_pot
            from_pile = (tile == OBJECT_TO_INDEX["onion_pile"]) | (tile == OBJECT_TO_INDEX["plate_pile"])
            onion_pile_pickup = pickup & from_pile
            counter_pickup = pickup & ~from_pile
            drop = ((inv_pre != OBJECT_TO_INDEX["empty"]) & (inv_post == OBJECT_TO_INDEX["empty"])
                    & ((tile == OBJECT_TO_INDEX["wall"]) | (tile == OBJECT_TO_INDEX["empty"])))
            # OvercookedV1's plate check always matches wall padding (grey == plate == 5), so it is 0 there
            picked_plate = pickup & (inv_post == OBJECT_TO_INDEX["plate"])
            pots = maze[prev.pot_pos[:, 1], prev.pot_pos[:, 0], 2]
            notempty_pots = jnp.sum((pots != self.env.cook_time + 3) & prev.pot_mask)
            useful = 0 < notempty_pots
            no_plates = jnp.sum(maze_after_own_turn[i] == OBJECT_TO_INDEX["plate"]) == 0
            plate_pickup = picked_plate & useful & no_plates
            r = (placement * p["PLACEMENT_IN_POT_REW"][i]
                 + soup_pickup * p["SOUP_PICKUP_REWARD"][i]
                 + holding_onion * p["ONION_HOLDING"][i]
                 + holding_plate * p["PLATE_HOLDING"][i]
                 + onion_pile_pickup * p["ONION_PICKUP_REWARD"][i]
                 + counter_pickup * p["COUNTER_PICKUP_REWARD"][i]
                 + drop * p["COUNTER_DROP_REWARD"][i]
                 + plate_pickup * p["PLATE_PICKUP_REWARD"][i])
            out.append(jnp.where(interact, r, 0.0))
            inv = inv.at[i].set(new_inv[i])
        return jnp.stack(out).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        key, k_meal, k_rand = jax.random.split(key, 3)
        _, state = self.env.reset(k_meal)
        state = self._randomise(k_rand, state)
        obs = self.env.get_obs(state)
        wrapped = WrappedEnvState(state, jnp.zeros(self.num_agents), jnp.zeros(self.num_agents),
                                  jnp.empty((), dtype=jnp.int32))
        return self._flat_obs(obs), wrapped

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: WrappedEnvState, actions: Dict[str, chex.Array],
             reset_state: Optional[WrappedEnvState] = None):
        key, k_step, k_reset = jax.random.split(key, 3)
        prev = state.env_state
        acts = jnp.stack([compat.jaxmarl_action_to_meal(jnp.asarray(actions[a]).reshape(()))
                          for a in self.agents]).astype(jnp.int32)
        _, nxt, rewards, dones, info = self.env.step_env(k_step, prev, acts)

        base = sum(rewards[a] for a in self.agents)  # JaxMARL shares the delivery reward
        base_reward = jnp.full((self.num_agents,), base, dtype=jnp.float32)
        maze_views = jnp.stack([nxt.maze_map] * self.num_agents)
        shaped = self._shaped_rewards(prev, acts, maze_views, nxt.agent_inv)
        soups = jnp.stack([info["soups"][a] for a in self.agents])

        if reset_state is None:
            _, reset_wrapped = self.reset(k_reset)
            reset_env_state = reset_wrapped.env_state
        else:
            reset_env_state = reset_state.env_state
        done = dones["__all__"]
        env_state = jax.tree.map(lambda r, s: jnp.where(done, r, s), reset_env_state, nxt)
        obs = self._flat_obs(self.env.get_obs(env_state))

        out_rewards = {a: base_reward[i] + self.do_reward_shaping * shaped[i] for i, a in enumerate(self.agents)}
        base_return = state.base_return_so_far + base_reward
        infos = {"shaped_reward": shaped, "base_reward": base_reward, "base_return": base_return,
                 "soups": soups}
        new_state = WrappedEnvState(env_state=env_state,
                                    base_return_so_far=jnp.where(done, jnp.zeros(self.num_agents), base_return),
                                    avail_actions=jnp.zeros(self.num_agents),
                                    step=jnp.empty((), dtype=jnp.int32))
        return obs, new_state, out_rewards, dones, infos


def make_jaxaht_overcooked(layout: str = "cramped_room", **env_kwargs):
    from envs.overcooked.augmented_layouts import augmented_layouts
    from envs.overcooked.overcooked_wrapper import OvercookedWrapper

    kwargs = {"random_reset": True, "random_obj_state": False, "max_steps": 400, **env_kwargs}
    if "reward_shaping_params" in kwargs:
        kwargs["reward_shaping_params"] = FrozenDict(
            {k: ([v, v] if isinstance(v, (int, float)) else list(v)) for k, v in
             dict(kwargs["reward_shaping_params"]).items()})
    return OvercookedWrapper(layout=FrozenDict(augmented_layouts[layout]), **kwargs)
