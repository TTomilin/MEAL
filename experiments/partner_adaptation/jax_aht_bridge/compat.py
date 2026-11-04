from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import experiments.partner_adaptation.jax_aht_bridge  # noqa: F401
from jaxmarl.environments.overcooked import common as jaxmarl_common
from jaxmarl.environments.overcooked.overcooked import State as JaxMarlState
from meal.env.overcooked.common import DIR_TO_VEC as MEAL_DIR_TO_VEC
from meal.env.overcooked.common import OBJECT_TO_INDEX as MEAL_OBJECT_TO_INDEX
from meal.env.overcooked.overcooked_env import Actions as MealActions

NUM_ACTIONS = 6
NUM_DIRS = 4
OWN_ORI = slice(2, 6)
OTHER_ORI = slice(6, 10)
JAXMARL_AGENT_VIEW_SIZE = 5  # hard-coded in JaxMARL's Overcooked

# overcooked_ai_py.mdp.actions.Direction.INDEX_TO_DIRECTION
OVERCOOKED_AI_DIR_TO_VEC = np.array([(0, -1), (0, 1), (1, 0), (-1, 0)], dtype=np.int8)


def _dir_permutation(src_dir_to_vec, dst_dir_to_vec) -> np.ndarray:
    src = np.asarray(src_dir_to_vec).astype(np.int64)
    dst = np.asarray(dst_dir_to_vec).astype(np.int64)
    perm = np.array([int(np.flatnonzero((dst == v).all(axis=1))[0]) for v in src], dtype=np.int32)
    assert sorted(perm.tolist()) == list(range(NUM_DIRS)), (src, dst)
    return perm


def _action_table(dir_perm: np.ndarray) -> np.ndarray:
    table = np.zeros(NUM_ACTIONS, dtype=np.int32)
    table[:NUM_DIRS] = dir_perm
    table[4] = int(MealActions.stay)
    table[5] = int(MealActions.interact)
    return table


JAXMARL_DIR_TO_MEAL_DIR = _dir_permutation(jaxmarl_common.DIR_TO_VEC, MEAL_DIR_TO_VEC)
MEAL_DIR_TO_JAXMARL_DIR = _dir_permutation(MEAL_DIR_TO_VEC, jaxmarl_common.DIR_TO_VEC)
OVERCOOKED_AI_DIR_TO_MEAL_DIR = _dir_permutation(OVERCOOKED_AI_DIR_TO_VEC, MEAL_DIR_TO_VEC)
MEAL_DIR_TO_OVERCOOKED_AI_DIR = _dir_permutation(MEAL_DIR_TO_VEC, OVERCOOKED_AI_DIR_TO_VEC)

JAXMARL_ACTION_TO_MEAL = _action_table(JAXMARL_DIR_TO_MEAL_DIR)
MEAL_ACTION_TO_JAXMARL = _action_table(MEAL_DIR_TO_JAXMARL_DIR)
OVERCOOKED_AI_ACTION_TO_MEAL = _action_table(OVERCOOKED_AI_DIR_TO_MEAL_DIR)
MEAL_ACTION_TO_OVERCOOKED_AI = _action_table(MEAL_DIR_TO_OVERCOOKED_AI_DIR)


def _obs_channel_gather(num_channels: int = 26) -> np.ndarray:
    gather = np.arange(num_channels, dtype=np.int32)
    for block in (OWN_ORI, OTHER_ORI):
        start = block.start
        for j in range(NUM_DIRS):
            gather[start + j] = start + JAXMARL_DIR_TO_MEAL_DIR[j]
    return gather


MEAL_TO_JAXMARL_OBS_CHANNELS = _obs_channel_gather()
OBS_IS_IDENTITY = bool((MEAL_TO_JAXMARL_OBS_CHANNELS == np.arange(26)).all())
ACTIONS_ARE_IDENTITY = bool((JAXMARL_ACTION_TO_MEAL == np.arange(NUM_ACTIONS)).all())


def meal_obs_to_jaxmarl(obs: jnp.ndarray, height: int | None = None, width: int | None = None) -> jnp.ndarray:
    if OBS_IS_IDENTITY:
        return obs
    flat = obs.shape[-1] != 26
    if flat:
        assert height is not None and width is not None, "flattened obs needs height and width"
        obs = obs.reshape(obs.shape[:-1] + (height, width, 26))
    out = jnp.take(obs, jnp.asarray(MEAL_TO_JAXMARL_OBS_CHANNELS), axis=-1)
    return out.reshape(out.shape[:-3] + (-1,)) if flat else out


def jaxmarl_action_to_meal(action: jnp.ndarray) -> jnp.ndarray:
    if ACTIONS_ARE_IDENTITY:
        return action
    return jnp.asarray(JAXMARL_ACTION_TO_MEAL)[action].astype(action.dtype)


def meal_action_to_jaxmarl(action: jnp.ndarray) -> jnp.ndarray:
    if ACTIONS_ARE_IDENTITY:
        return action
    return jnp.asarray(MEAL_ACTION_TO_JAXMARL)[action].astype(action.dtype)


def avail_meal_to_jaxmarl(avail: jnp.ndarray) -> jnp.ndarray:
    if ACTIONS_ARE_IDENTITY:
        return avail
    return jnp.take(avail, jnp.asarray(JAXMARL_ACTION_TO_MEAL), axis=-1)


def overcookedai_action_to_meal(action: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(OVERCOOKED_AI_ACTION_TO_MEAL)[action].astype(jnp.int32)


def meal_state_to_jaxmarl(state, num_pots: int, num_goals: int,
                          agent_view_size: int = JAXMARL_AGENT_VIEW_SIZE) -> JaxMarlState:
    dir_meal_to_jm = jnp.asarray(MEAL_DIR_TO_JAXMARL_DIR)
    agent_dir_idx = dir_meal_to_jm[state.agent_dir_idx]
    maze = state.maze_map
    is_agent = maze[..., 0] == MEAL_OBJECT_TO_INDEX["agent"]
    maze = maze.at[..., 2].set(jnp.where(is_agent, dir_meal_to_jm[jnp.minimum(maze[..., 2], 3)].astype(maze.dtype),
                                         maze[..., 2]))
    pad = agent_view_size - 1
    wall = jnp.array([jaxmarl_common.OBJECT_TO_INDEX["wall"], jaxmarl_common.COLOR_TO_INDEX["grey"], 0],
                     dtype=maze.dtype)
    padded = jnp.broadcast_to(wall, (maze.shape[0] + 2 * pad, maze.shape[1] + 2 * pad, 3))
    padded = padded.at[pad:pad + maze.shape[0], pad:pad + maze.shape[1], :].set(maze)
    return JaxMarlState(
        agent_pos=state.agent_pos,
        agent_dir=jnp.asarray(jaxmarl_common.DIR_TO_VEC)[agent_dir_idx],
        agent_dir_idx=agent_dir_idx,
        agent_inv=state.agent_inv,
        goal_pos=state.goal_pos[:num_goals],
        pot_pos=state.pot_pos[:num_pots],
        wall_map=state.wall_map,
        maze_map=padded,
        time=state.time,
        terminal=state.terminal,
    )
