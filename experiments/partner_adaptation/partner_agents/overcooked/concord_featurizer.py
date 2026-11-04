import jax.numpy as jnp

from meal.env.overcooked.common import OBJECT_TO_INDEX

NUM_CHANNELS = 20
# Overcooked-AI and MEAL share direction order N, S, E, W (checked in tests/test_concord_partner.py)
OVERCOOKED_AI_DIR_OF_MEAL_DIR = jnp.arange(4)


def concord_featurize(state, seat) -> jnp.ndarray:
    maze = state.maze_map
    obj = maze[..., 0]
    status = maze[..., 2].astype(jnp.int32)
    h, w = obj.shape
    other = 1 - seat

    def at(pos, value=1.0):
        return jnp.zeros((h, w), jnp.float32).at[pos[1], pos[0]].set(value)

    pos_own, pos_other = state.agent_pos[seat], state.agent_pos[other]
    dir_own = OVERCOOKED_AI_DIR_OF_MEAL_DIR[state.agent_dir_idx[seat]]
    dir_other = OVERCOOKED_AI_DIR_OF_MEAL_DIR[state.agent_dir_idx[other]]
    ori_own = jnp.zeros((4, h, w), jnp.float32).at[dir_own].set(at(pos_own))
    ori_other = jnp.zeros((4, h, w), jnp.float32).at[dir_other].set(at(pos_other))

    pot = obj == OBJECT_TO_INDEX["pot"]
    onion_disp = obj == OBJECT_TO_INDEX["onion_pile"]
    dish_disp = obj == OBJECT_TO_INDEX["plate_pile"]
    serve = obj == OBJECT_TO_INDEX["goal"]
    counter = state.wall_map & ~(pot | onion_disp | dish_disp | serve)

    onions_in_pot = jnp.minimum(state.pot_empty_status - status, state.max_onions_in_pot) * pot
    cook_time = jnp.where(status <= state.pot_full_status, state.pot_full_status - status, 0) * pot

    held = jnp.zeros((h, w), jnp.int32)
    for i in range(state.agent_pos.shape[0]):
        held = held.at[state.agent_pos[i, 1], state.agent_pos[i, 0]].set(state.agent_inv[i].astype(jnp.int32))

    def items(kind):
        return ((obj == OBJECT_TO_INDEX[kind]) | (held == OBJECT_TO_INDEX[kind])).astype(jnp.float32)

    layers = jnp.concatenate([
        at(pos_own)[None], at(pos_other)[None], ori_own, ori_other,
        jnp.stack([pot, counter, onion_disp, dish_disp, serve]).astype(jnp.float32),
        jnp.stack([onions_in_pot, cook_time]).astype(jnp.float32),
        jnp.stack([items("dish"), items("plate"), items("onion")]),
    ], axis=0)
    return jnp.transpose(layers, (2, 1, 0))  # Overcooked-AI encoding is (W, H, 20), indexed [x, y]
