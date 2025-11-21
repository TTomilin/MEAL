from functools import partial
from typing import Tuple

import chex
import jax
from flax import struct
from jax import numpy as jnp

from meal.env import State, MultiAgentEnv
from meal.wrappers.jaxmarl import JaxMARLWrapper


@struct.dataclass
class SlipperyTileState:
    env_state: State
    slippery_mask: chex.Array  # (H, W) bool
    will_slip_next: chex.Array  # (num_agents,) bool
    last_pos: chex.Array  # (num_agents, 2) agent positions at previous step


class SlipperyTiles(JaxMARLWrapper):
    """
    Slippery tiles wrapper.

    - At reset: mark 25% of grid tiles as slippery (fixed per episode).
    - If an agent steps *onto* a slippery tile at time t, then at time t+1
      their chosen action is replaced with a random move (up/down/left/right)
      with probability p_replace.
    """

    def __init__(self, env: MultiAgentEnv, slip_prob: float = 0.1):
        super().__init__(env)
        self.slip_prob = slip_prob  # slip probability when "armed"
        self.fraction_slippery = 0.25  # 25% of tiles are slippery

        # We assume an Overcooked-style env with height/width & num_agents
        self.height = int(self._env.height)
        self.width = int(self._env.width)
        self.num_agents = int(self._env.num_agents)

    def _update_slip_state(
            self,
            prev_state: SlipperyTileState,
            env_state: State,
    ) -> SlipperyTileState:
        """Compute will_slip_next/last_pos for the next step."""
        cur_pos = env_state.agent_pos  # (num_agents, 2)
        last_pos = prev_state.last_pos

        moved = jnp.any(cur_pos != last_pos, axis=-1)  # (num_agents,)

        y = cur_pos[:, 1]
        x = cur_pos[:, 0]
        on_slippery = prev_state.slippery_mask[y, x].astype(jnp.bool_)

        will_slip_next = moved & on_slippery

        return SlipperyTileState(
            env_state=env_state,
            slippery_mask=prev_state.slippery_mask,
            will_slip_next=will_slip_next,
            last_pos=cur_pos,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, SlipperyTileState]:
        # Reset underlying env
        key, env_key = jax.random.split(key)
        obs, env_state = self._env.reset(env_key)

        # ------------------------------------------------------------------
        # Sample a fixed slippery mask for this episode, ONLY on walkable tiles
        # Walkable = not a wall/counter AND not a goal tile.
        # ------------------------------------------------------------------
        H, W = self.height, self.width

        wall_map = env_state.wall_map  # (H, W) bool

        # Build a (H, W) mask of goal tiles
        goal_mask_layer = jnp.zeros_like(wall_map)
        goal_y = env_state.goal_pos[:, 1]
        goal_x = env_state.goal_pos[:, 0]
        goal_mask_layer = goal_mask_layer.at[goal_y, goal_x].set(env_state.goal_mask)

        # Walkable tiles are those that are not walls/counters and not goals
        walkable = (~wall_map) & (~goal_mask_layer)  # (H, W) bool

        # Sample approx. fraction_slippery of *walkable* tiles
        flat_walkable = walkable.reshape(-1)

        key, slip_key = jax.random.split(key)
        slip_draw = jax.random.bernoulli(
            slip_key,
            p=self.fraction_slippery,
            shape=flat_walkable.shape,
        )

        # A tile is slippery iff it is walkable and the Bernoulli says "yes"
        flat = jnp.logical_and(flat_walkable, slip_draw)
        slippery_mask = flat.reshape((H, W))

        # ------------------------------------------------------------------
        # Slippery state bookkeeping
        # ------------------------------------------------------------------
        will_slip_next = jnp.zeros((self.num_agents,), dtype=jnp.bool_)
        last_pos = env_state.agent_pos  # (num_agents, 2)

        state = SlipperyTileState(
            env_state=env_state,
            slippery_mask=slippery_mask,
            will_slip_next=will_slip_next,
            last_pos=last_pos,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            key: chex.PRNGKey,
            state: SlipperyTileState,
            action,
    ):
        """
        JaxMARL-style step: `action` is a pytree (e.g. dict of scalars).
        We apply slip to per-agent action indices, then call env.step.
        """
        # Split RNG: slip decision, direction choice, and env step
        key, slip_key, dir_key, env_key = jax.random.split(key, 4)

        # Flatten action pytree into (num_agents,)
        leaves, treedef = jax.tree_util.tree_flatten(action)
        actions_arr = jnp.stack([jnp.asarray(x).squeeze() for x in leaves], axis=0)
        num_agents = actions_arr.shape[0]

        # Bernoulli: which agents *could* slip
        slip_draw = jax.random.bernoulli(
            slip_key, p=self.slip_prob, shape=(num_agents,)
        )
        should_slip = slip_draw & state.will_slip_next

        # Random move in {0,1,2,3} = up/down/right/left
        rand_dirs = jax.random.randint(
            dir_key, shape=(num_agents,), minval=0, maxval=4
        )
        effective_arr = jnp.where(should_slip, rand_dirs, actions_arr)

        # Rebuild pytree with modified actions
        effective_leaves = [effective_arr[i] for i in range(len(leaves))]
        effective_action = jax.tree_util.tree_unflatten(treedef, effective_leaves)

        # Step underlying env
        obs, env_state, reward, done, info = self._env.step(
            env_key, state.env_state, effective_action
        )

        # Update slip state from new positions
        new_state = self._update_slip_state(state, env_state)

        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
            self,
            key: chex.PRNGKey,
            state: SlipperyTileState,
            actions,
    ):
        """
        MultiAgentEnv-style step with dict-of-actions, mirroring Overcooked.step_env.
        """
        # Convert dict -> (num_agents,) array of indices
        if isinstance(actions, dict):
            # Keep the env's agent ordering
            agent_ids = list(self._env.agents)
            action_indices = jnp.array(
                [actions[a].flatten()[0] for a in agent_ids],
                dtype=jnp.int32,
            )
        else:
            # Already an array
            action_indices = actions

        key, slip_key, dir_key, env_key = jax.random.split(key, 4)
        num_agents = action_indices.shape[0]

        # Which armed agents actually slip?
        slip_draw = jax.random.bernoulli(
            slip_key, p=self.slip_prob, shape=(num_agents,)
        )
        should_slip = slip_draw & state.will_slip_next

        # Random move in {0,1,2,3}
        rand_dirs = jax.random.randint(
            dir_key, shape=(num_agents,), minval=0, maxval=4
        )
        effective_indices = jnp.where(should_slip, rand_dirs, action_indices)

        # Rebuild actions in original format
        if isinstance(actions, dict):
            effective_actions = {
                a: effective_indices[i] for i, a in enumerate(agent_ids)
            }
        else:
            effective_actions = effective_indices

        # Step underlying env in its native API
        obs, env_state, rew, done, info = self._env.step_env(
            env_key, state.env_state, effective_actions
        )

        # Update slip state
        new_state = self._update_slip_state(state, env_state)

        return obs, new_state, rew, done, info
