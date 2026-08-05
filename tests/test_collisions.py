"""3-agent movement collisions: three agents scripted to rotate around a
triangle of positions must end up exactly cycled (chef-1 -> chef-2's spot,
etc.), i.e. simultaneous moves into each other's tiles resolve as a swap
rather than agents clipping through / colliding."""
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from meal.env.overcooked import Overcooked
from meal.env.overcooked.layouts.presets import cramped_room

A = {'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5}


def test_triple_agent_collisions():
    env = Overcooked(layout=FrozenDict(cramped_room), num_agents=3, random_reset=False,
                      max_steps=50, start_idx=(6, 8, 12))  # fixed spawn for all 3 agents
    rng = jax.random.PRNGKey(0)
    _, state = env.reset(rng)
    init_pos = np.asarray(state.agent_pos)  # (3, 2)

    step_actions = [
        dict(agent_0=jnp.uint32(A['R']), agent_1=jnp.uint32(A['L']), agent_2=jnp.uint32(A['U'])),
        dict(agent_0=jnp.uint32(A['R']), agent_1=jnp.uint32(A['D']), agent_2=jnp.uint32(A['L'])),
        dict(agent_0=jnp.uint32(A['D']), agent_1=jnp.uint32(A['L']), agent_2=jnp.uint32(A['R'])),
        dict(agent_0=jnp.uint32(A['R']), agent_1=jnp.uint32(A['L']), agent_2=jnp.uint32(A['U'])),
    ]

    for act in step_actions:
        rng, key = jax.random.split(rng)
        _, state, _, _, _ = env.step_env(key, state, act)

    final_pos = np.asarray(state.agent_pos)
    assert np.array_equal(final_pos[0], init_pos[1]), "chef-1 didn't move to chef-2's position!"
    assert np.array_equal(final_pos[1], init_pos[2]), "chef-2 didn't move to chef-3's position!"
    assert np.array_equal(final_pos[2], init_pos[0]), "chef-3 didn't move to chef-1's position!"
