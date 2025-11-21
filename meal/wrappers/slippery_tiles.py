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

    def __init__(self, env: MultiAgentEnv, p_replace: float = 0.1):
        super().__init__(env)
        self.p_replace = p_replace          # slip probability when "armed"
        self.fraction_slippery = 0.25       # 25% of tiles are slippery

        # We assume an Overcooked-style env with height/width & num_agents
        self.height = int(self._env.height)
        self.width = int(self._env.width)
        self.num_agents = int(self._env.num_agents)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, SlipperyTileState]:
        # Reset underlying env
        key, mask_key = jax.random.split(key)
        obs, env_state = self._env.reset(mask_key)

        # Sample a fixed slippery mask for this episode
        H, W = self.height, self.width
        num_cells = H * W
        num_slippery = int(self.fraction_slippery * num_cells)

        # Choose num_slippery cells uniformly without replacement
        perm = jax.random.permutation(key, num_cells)
        chosen = perm[:num_slippery]
        flat = jnp.zeros((num_cells,), dtype=jnp.bool_).at[chosen].set(True)
        slippery_mask = flat.reshape((H, W))

        # Initially nobody is armed to slip on the next step
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
        1) Use state.will_slip_next to decide which agents *may* slip this step.
        2) For those agents, with prob p_replace override their action
           by a random move in {up, down, right, left}.
        3) Step the underlying env with the effective actions.
        4) From the new positions, mark agents that *just entered* a slippery
           tile as will_slip_next for the next step.
        """

        # Split RNG: slip decision, direction choice, and env step
        key, slip_key, dir_key, env_key = jax.random.split(key, 4)

        # Flatten the action pytree to a 1D array: one scalar per agent
        leaves, treedef = jax.tree_util.tree_flatten(action)
        actions_arr = jnp.stack([jnp.asarray(x).squeeze() for x in leaves], axis=0)
        # actions_arr shape: (num_agents,)

        # Sanity: we assume the number of leaves matches num_agents
        # (this is the usual case for dict{agent: scalar_action})
        # If it doesn't, JAX will complain anyway when shapes misalign.
        num_agents = actions_arr.shape[0]

        # Bernoulli: which agents actually slip this step?
        # Only those that were "armed" in will_slip_next can slip.
        slip_draw = jax.random.bernoulli(slip_key, p=self.p_replace, shape=(num_agents,))
        should_slip = slip_draw & state.will_slip_next

        # For slipping agents: pick a random move action in {0,1,2,3}
        # These are the indices for up/down/right/left in your Actions enum.
        rand_dirs = jax.random.randint(dir_key, shape=(num_agents,), minval=0, maxval=4)
        slip_actions = rand_dirs

        effective_arr = jnp.where(should_slip, slip_actions, actions_arr)

        # Rebuild the action pytree with the modified actions
        effective_leaves = [effective_arr[i] for i in range(len(leaves))]
        effective_action = jax.tree_util.tree_unflatten(treedef, effective_leaves)

        # Step underlying env with the modified actions
        obs, env_state, reward, done, info = self._env.step(
            env_key, state.env_state, effective_action
        )

        # --- Update slip state for NEXT step ---------------------------------
        # Current positions after env step
        cur_pos = env_state.agent_pos  # (num_agents, 2)
        last_pos = state.last_pos

        # Did the agent actually move?
        moved = jnp.any(cur_pos != last_pos, axis=-1)  # (num_agents,)

        # Are the new positions on a slippery tile?
        y = cur_pos[:, 1]
        x = cur_pos[:, 0]
        on_slippery = state.slippery_mask[y, x].astype(jnp.bool_)  # (num_agents,)

        # "Armed" to slip on the next timestep iff they *just stepped onto* a slippery tile
        will_slip_next = moved & on_slippery

        new_state = SlipperyTileState(
            env_state=env_state,
            slippery_mask=state.slippery_mask,
            will_slip_next=will_slip_next,
            last_pos=cur_pos,
        )

        return obs, new_state, reward, done, info
