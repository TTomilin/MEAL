from enum import IntEnum
from typing import Tuple, Dict

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.core.frozen_dict import FrozenDict
from jax import lax

from jax_marl.environments import MultiAgentEnv
from jax_marl.environments import spaces
from jax_marl.environments.overcooked_environment.common import (
    OBJECT_TO_INDEX,
    OBJECT_INDEX_TO_VEC,
    DIR_TO_VEC,
    make_overcooked_map)
from jax_marl.environments.overcooked_environment.layouts import overcooked_layouts as layouts

BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,  # reward for putting ingredients
    "PLATE_PICKUP_REWARD": 3,  # reward for picking up a plate
    "SOUP_PICKUP_REWARD": 5,  # reward for picking up a ready soup
    "DROP_COUNTER_REWARD": 0,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}


class Actions(IntEnum):
    # Turn left, turn right, move forward
    up = 0
    down = 1
    right = 2
    left = 3
    stay = 4
    interact = 5


@struct.dataclass
class State:
    agent_pos: chex.Array  # (n,2)
    agent_dir: chex.Array  # (n,2)
    agent_dir_idx: chex.Array  # (n,)
    agent_inv: chex.Array  # (n,)
    goal_pos: chex.Array  # (g,2)
    pot_pos: chex.Array  # (p,2)
    wall_map: chex.Array  # (H,W) bool
    maze_map: chex.Array  # (H+pad, W+pad, 3)
    time: int
    terminal: bool
    task_id: int


# Pot status indicated by an integer, which ranges from 23 to 0
POT_EMPTY_STATUS = 23  # 22 = 1 onion in pot; 21 = 2 onions in pot; 20 = 3 onions in pot
POT_FULL_STATUS = 20  # 3 onions. Below this status, pot is cooking, and status acts like a countdown timer.
POT_READY_STATUS = 0
MAX_ONIONS_IN_POT = 3  # A pot has at most 3 onions. A soup contains exactly 3 onions.

URGENCY_CUTOFF = 40  # When this many time steps remain, the urgency layer is flipped on
DELIVERY_REWARD = 20


class Overcooked(MultiAgentEnv):
    """Vanilla Overcooked"""

    def __init__(
            self,
            layout: dict | None = None,
            layout_name="cramped_room",
            random_reset: bool = False,
            max_steps: int = 400,
            task_id: int = 0,
            num_agents: int = 2,
            start_idx: tuple[int, ...] | None = None,
    ):
        super().__init__(num_agents=num_agents)

        # self.obs_shape = (agent_view_size, agent_view_size, 3)
        # Observations given by 26 channels, most of which are boolean masks
        self.height = layout["height"]
        self.width = layout["width"]
        self.env_layers = 16  # Number of environment layers (static, dynamic, pot, soup, etc.)
        self.obs_channels = 18 + 4 * num_agents  # 26 when n = 2
        self.obs_shape = (self.width, self.height, self.obs_channels)

        self.agent_view_size = 5  # Hard coded. Only affects map padding -- not observations.
        self.layout = layout if layout is not None else FrozenDict(layouts["cramped_room"])
        self.layout_name = layout_name
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.start_idx = None if start_idx is None else jnp.array(start_idx, jnp.uint32)

        self.action_set = jnp.array(list(Actions), dtype=jnp.uint8)

        self.random_reset = random_reset
        self.max_steps = max_steps
        self.task_id = task_id

    # ─────────────────────────  observation  ──────────────────────

    def _pos_layers(self, state: State) -> chex.Array:
        """(n,H,W) – layer i has 1 at agent_i position."""
        H, W = self.height, self.width
        layers = jnp.zeros((self.num_agents, H, W), jnp.uint8)
        y, x = state.agent_pos[:, 1], state.agent_pos[:, 0]
        return layers.at[jnp.arange(self.num_agents), y, x].set(1)

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Return a full observation, of size (height x width x n_layers), where n_layers = 26.
        Layers are of shape (height x width) and  are binary (0/1) except where indicated otherwise.
        The obs is very sparse (most elements are 0), which prob. contributes to generalization problems in Overcooked.
        A v2 of this environment should have much more efficient observations, e.g. using item embeddings

        The list of channels is below. Agent-specific layers are ordered so that an agent perceives its layers first.
        Env layers are the same (and in same order) for both agents.

        Agent positions :
        0. position of agent i (1 at agent loc, 0 otherwise)
        1. position of agent (1-i)

        Agent orientations :
        2-5. agent_{i}_orientation_0 to agent_{i}_orientation_3 (layers are entirely zero except for the one orientation
        layer that matches the agent orientation. That orientation has a single 1 at the agent coordinates.)
        6-9. agent_{i-1}_orientation_{dir}

        Static env positions (1 where object of type X is located, 0 otherwise.):
        10. pot locations
        11. counter locations (table)
        12. onion pile locations
        13. tomato pile locations (tomato layers are included for consistency, but this env does not support tomatoes)
        14. plate pile locations
        15. delivery locations (goal)

        Pot and soup specific layers. These are non-binary layers:
        16. number of onions in pot (0,1,2,3) for elements corresponding to pot locations. Nonzero only for pots that
        have NOT started cooking yet. When a pot starts cooking (or is ready), the corresponding element is set to 0
        17. number of tomatoes in pot.
        18. number of onions in soup (0,3) for elements corresponding to either a cooking/done pot or to a soup (dish)
        ready to be served. This is a useless feature since all soups have exactly 3 onions, but it made sense in the
        full Overcooked where recipes can be a mix of tomatoes and onions
        19. number of tomatoes in soup
        20. pot cooking time remaining. [19 -> 1] for pots that are cooking. 0 for pots that are not cooking or done
        21. soup done. (Binary) 1 for pots done cooking and for locations containing a soup (dish). O otherwise.

        Variable env layers (binary):
        22. plate locations
        23. onion locations
        24. tomato locations

        Urgency:
        25. Urgency. The entire layer is 1 there are 40 or fewer remaining time steps. 0 otherwise
        """
        H, W = self.height, self.width
        pad = (state.maze_map.shape[0] - H) // 2
        maze = state.maze_map[pad:-pad, pad:-pad]  # (H,W,3)

        # ────────────────────── build the 16 env layers ────────────────────────
        obj = maze[:, :, 0]  # tile indices
        pot_mask = (obj == OBJECT_TO_INDEX["pot"])
        dish_mask = (obj == OBJECT_TO_INDEX["dish"])

        pot_status = maze[:, :, 2] * pot_mask  # 0–23   at pot tiles

        onions_in_pot = jnp.minimum(POT_EMPTY_STATUS - pot_status,
                                    MAX_ONIONS_IN_POT) * (pot_status >= POT_FULL_STATUS)

        onions_in_soup = (jnp.minimum(POT_EMPTY_STATUS - pot_status,
                                      MAX_ONIONS_IN_POT) * (pot_status < POT_FULL_STATUS)
                          + MAX_ONIONS_IN_POT * dish_mask)

        pot_cook_time = pot_status * (pot_status < POT_FULL_STATUS)
        soup_ready = pot_mask * (pot_status == POT_READY_STATUS) + dish_mask
        urgency = jnp.ones_like(obj, jnp.uint8) * ((self.max_steps - state.time) < URGENCY_CUTOFF)

        env_layers = jnp.stack([
            pot_mask.astype(jnp.uint8),  # 10
            (obj == OBJECT_TO_INDEX["wall"]).astype(jnp.uint8),
            (obj == OBJECT_TO_INDEX["onion_pile"]).astype(jnp.uint8),
            jnp.zeros_like(obj, jnp.uint8),  # tomato‐pile (unused)
            (obj == OBJECT_TO_INDEX["plate_pile"]).astype(jnp.uint8),
            (obj == OBJECT_TO_INDEX["goal"]).astype(jnp.uint8),  # 15
            onions_in_pot.astype(jnp.uint8),
            jnp.zeros_like(obj, jnp.uint8),  # tomatoes in pot (unused)
            onions_in_soup.astype(jnp.uint8),
            jnp.zeros_like(obj, jnp.uint8),  # tomatoes in soup (unused)
            pot_cook_time.astype(jnp.uint8),  # 20
            soup_ready.astype(jnp.uint8),
            (obj == OBJECT_TO_INDEX["plate"]).astype(jnp.uint8),
            (obj == OBJECT_TO_INDEX["onion"]).astype(jnp.uint8),
            jnp.zeros_like(obj, jnp.uint8),  # tomatoes (unused)
            urgency.astype(jnp.uint8),  # 25
        ], axis=0)  # → (16,H,W)

        # ────────────────────── agent-specific layers ──────────────────────────
        pos_layers = self._pos_layers(state)  # (n,H,W)

        # orientation one-hot layers
        ori_layers = jnp.zeros((4 * self.num_agents, H, W), jnp.uint8)
        idx = jnp.arange(self.num_agents)
        ori_layers = ori_layers.at[4 * idx + state.agent_dir_idx, :, :].set(
            pos_layers)

        # ────────────────────── assemble per-agent views ───────────────────────
        views: Dict[str, chex.Array] = {}
        for i in range(self.num_agents):
            own_pos = pos_layers[i:i + 1]  # 1 layer
            others_pos = jnp.delete(pos_layers, i, axis=0)
            own_ori = ori_layers[4 * i:4 * (i + 1)]  # 4 layers
            others_ori = jnp.delete(ori_layers,
                                    slice(4 * i, 4 * (i + 1)), axis=0)  # 4(n-1)

            layers = jnp.concatenate([
                own_pos,
                others_pos.sum(0, keepdims=True),  # aggregate pos
                own_ori,
                others_ori,
                env_layers,
            ], axis=0)  # (C,H,W)

            views[f"agent_{i}"] = jnp.transpose(layers, (1, 2, 0))  # (H,W,C)

        return views

    # ───────────────────────── movement / step ────────────────────

    def _proposed_positions(self, state: State, action: chex.Array):
        move_mask = action < 4  # 0-3 are the directional actions
        step_vec = DIR_TO_VEC[action.clip(max=3)]
        proposed = jnp.clip(
            state.agent_pos + move_mask[:, None] * step_vec,
            a_min=0,
            a_max=jnp.array((self.width - 1, self.height - 1), jnp.uint32),
        ).astype(jnp.uint32)

        # block by walls / goals
        wall_block = state.wall_map[proposed[:, 1], proposed[:, 0]]
        goal_block = (proposed[:, None, :] == state.goal_pos[None, :, :]).all(-1).any(-1)
        blocked = wall_block | goal_block | (~move_mask)
        return jnp.where(blocked[:, None], state.agent_pos, proposed)

    def _resolve_collisions(self, current, proposed):
        n = current.shape[0]

        # same destination (collision)  ────────────────────────────────
        same_dest = (proposed[:, None, :] == proposed[None, :, :]).all(-1)
        coll = (same_dest.sum(-1) > 1)  # True if ≥2 agents share a tile

        # swap places (i ↔ j)  ─────────────────────────────────────────
        if n == 1:  # no other agents → no swap test
            blocked = coll
        else:
            swap = ((proposed[:, None, :] == current[None, :, :]).all(-1) &
                    (proposed[None, :, :] == current[:, None, :]).all(-1))

            # ignore i==j diagonal ─ we only care about pairs (i ≠ j)
            swap = swap & (~jnp.eye(n, dtype=bool))
            blocked = coll | swap.any(-1)

        return jnp.where(blocked[:, None], current, proposed).astype(jnp.uint32)

    def step_agents(self, key, state, action):
        assert action.shape == (self.num_agents,)
        # positions ------------------------------------------------------------
        proposed = self._proposed_positions(state, action)
        agent_pos = self._resolve_collisions(state.agent_pos, proposed).astype(jnp.uint32)

        # directions -----------------------------------------------------------
        agent_dir_idx = jnp.where(action < 4, action, state.agent_dir_idx)
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        # >>> this is the square the agent is facing <<<
        fwd_pos_all = agent_pos + agent_dir  # shape (n, 2)

        # ---------------------------------------------------------------------
        # interactions – sequential scan to mimic original ordering
        # ---------------------------------------------------------------------
        def body(carry, idx):
            maze, inv, rew, shaped = carry
            # only process when the agent actually pressed INTERACT
            maze_new, inv_i, r_i, s_i = lax.cond(
                action[idx] == Actions.interact,
                lambda _: self.process_interact(
                    maze, state.wall_map, fwd_pos_all,
                    inv, idx, state.agent_pos, agent_pos,
                    agent_dir_idx, state.pot_pos),
                # no-op branch
                lambda _: (maze, inv[idx], 0., 0.),
                operand=None,
            )
            inv = inv.at[idx].set(inv_i)
            rew = rew + r_i
            shaped = shaped.at[idx].set(s_i)
            return (maze_new, inv, rew, shaped), None

        init_carry = (state.maze_map, state.agent_inv,
                      jnp.float32(0.), jnp.zeros(self.num_agents, jnp.float32))

        (maze_map, agent_inv, reward, shaped_r), _ = lax.scan(body, init_carry, jnp.arange(self.num_agents))

        # ─── tick every pot exactly once per env-step ────────────────────────
        pad = (maze_map.shape[0] - self.height) // 2
        pot_x, pot_y = state.pot_pos[:, 0], state.pot_pos[:, 1]

        def _tick(pot):
            status = pot[-1]
            cooking = (status <= POT_FULL_STATUS) & (status > POT_READY_STATUS)
            return pot.at[-1].set(jnp.where(cooking, status - 1, status))

        pots = jax.vmap(_tick)(maze_map[pad + pot_y, pad + pot_x])
        maze_map = maze_map.at[pad + pot_y, pad + pot_x, :].set(pots)

        # ─── repaint agents (always, not only on INTERACT) ───────────────────
        pad = (maze_map.shape[0] - self.height) // 2
        empty_vec = OBJECT_INDEX_TO_VEC[OBJECT_TO_INDEX["empty"]]
        maze_map = maze_map.at[pad + state.agent_pos[:, 1], pad + state.agent_pos[:, 0], :].set(empty_vec)

        def _agent_vec(dir_idx, idx):
            return jnp.array([OBJECT_TO_INDEX["agent"], 2 * idx, dir_idx], dtype=jnp.uint8)

        agent_tiles = jax.vmap(_agent_vec)(agent_dir_idx, jnp.arange(self.num_agents))
        maze_map = maze_map.at[pad + agent_pos[:, 1], pad + agent_pos[:, 0], :].set(agent_tiles)

        new_state = state.replace(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
            maze_map=maze_map,
            terminal=False,
        )
        return new_state, reward, shaped_r

    def step_env(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""

        # convert incoming dict → jnp.array([a0,a1,…])
        if isinstance(actions, dict):
            act_arr = jnp.array([actions[a] for a in self.agents], dtype=jnp.uint8)
        else:
            act_arr = actions

        state, reward, shaped_rewards = self.step_agents(key, state, act_arr)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)

        # package outputs back into dict form
        # convert reward → per-agent dict
        if reward.ndim == 0:  # scalar -> same for all
            rew_dict = {a: reward for a in self.agents}
        else:  # vector length n
            rew_dict = {a: reward[i] for i, a in enumerate(self.agents)}

        # shaped reward is already a length-n vector
        shaped_dict = {a: shaped_rewards[i] for i, a in enumerate(self.agents)}
        done_dict = {a: done for a in self.agents} | {"__all__": done}

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rew_dict,
            done_dict,
            {'shaped_reward': shaped_dict},
        )

    def reset(
            self,
            key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], State]:
        """Reset environment state based on `self.random_reset`

        If True, everything is randomized, including agent inventories and positions, pot states and items on counters
        If False, only resample agent orientations

        In both cases, the environment layout is determined by `self.layout`
        """

        # Whether to fully randomize the start state
        random_reset = self.random_reset
        layout = self.layout

        h = self.height
        w = self.width
        num_agents = self.num_agents
        all_pos = np.arange(h * w, dtype=jnp.uint32)

        wall_idx = layout.get("wall_idx")
        occupied = jnp.zeros_like(all_pos).at[wall_idx].set(1)

        # -------------- choose starting squares ----------------------
        #  Priority:
        #  1. user-supplied `start_idx`
        #  2. if random_reset → ignore layout, sample every chef uniformly
        #  3. layout-provided squares + (if n_agents > layout) sample the remainder
        #
        #  all samples are drawn only from free counter tiles (no walls / fixtures)
        #
        provided_idx = (
            self.start_idx  # 1️⃣ explicit hard-coded spawn
            if self.start_idx is not None
            else (  # 2️⃣ fully random episode
                jnp.array([], jnp.uint32)
                if self.random_reset
                else layout.get("agent_idx", jnp.array([], jnp.uint32))  # 3️⃣ layout default
            )
        )

        n_provided = provided_idx.shape[0]
        n_missing = self.num_agents - n_provided
        agent_idx = provided_idx[: self.num_agents]

        if n_missing > 0:
            # ————————————————— mask out forbidden squares —————————————————
            occupied_mask = occupied.at[agent_idx].set(1)

            fixture_idx = (layout["onion_pile_idx"]
                           | layout["plate_pile_idx"]
                           | layout["goal_idx"]
                           | layout["pot_idx"])
            occupied_mask = occupied_mask.at[fixture_idx].set(1)

            # ————————————————— sample the remainder ——————————————————————
            # build a boolean “can-spawn-here” mask
            blocked = occupied_mask.at[(
                       layout["onion_pile_idx"]
                       | layout["plate_pile_idx"]
                       | layout["goal_idx"]
                       | layout["pot_idx"]
                   )].set(1)

            key, sub = jax.random.split(key)

            weights = (~blocked).astype(jnp.float32)      # 1 = floor, 0 = not allowed
            weights = weights / weights.sum()             # normalise for choice()

            extra_idx = jax.random.choice(
                   sub,                                       # PRNG
                   all_pos,                                   # *static* 1-D index vector
                   shape=(n_missing,),
                replace=False,
                p=weights,                                 # zero-probability ⇒ never picked
            )
            agent_idx = jnp.concatenate([agent_idx, extra_idx], axis=0)

        wall_map = occupied.reshape(h, w).astype(jnp.bool_)

        # Replace with fixed layout if applicable. Also randomize if agent position not provided
        # agent_idx = random_reset * agent_idx + (1 - random_reset) * layout.get("agent_idx", agent_idx)
        agent_pos = jnp.array([agent_idx % w, agent_idx // w], dtype=jnp.uint32).transpose()  # dim = n_agents x 2
        occupied = occupied.at[agent_idx].set(1)

        key, subkey = jax.random.split(key)
        agent_dir_idx = jax.random.choice(subkey, jnp.arange(len(DIR_TO_VEC), dtype=jnp.int32), shape=(num_agents,))
        agent_dir = DIR_TO_VEC.at[agent_dir_idx].get()  # dim = n_agents x 2

        # Keep track of empty counter space (table)
        empty_table_mask = jnp.zeros_like(all_pos)
        empty_table_mask = empty_table_mask.at[wall_idx].set(1)

        goal_idx = layout.get("goal_idx")
        goal_pos = jnp.array([goal_idx % w, goal_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[goal_idx].set(0)

        onion_pile_idx = layout.get("onion_pile_idx")
        onion_pile_pos = jnp.array([onion_pile_idx % w, onion_pile_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[onion_pile_idx].set(0)

        plate_pile_idx = layout.get("plate_pile_idx")
        plate_pile_pos = jnp.array([plate_pile_idx % w, plate_pile_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[plate_pile_idx].set(0)

        pot_idx = layout.get("pot_idx")
        pot_pos = jnp.array([pot_idx % w, pot_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[pot_idx].set(0)

        key, subkey = jax.random.split(key)
        # Pot status is determined by a number between 0 (inclusive) and 24 (exclusive)
        # 23 corresponds to an empty pot (default)
        pot_status = jax.random.randint(subkey, (pot_idx.shape[0],), 0, 24)
        pot_status = pot_status * random_reset + (1 - random_reset) * jnp.ones((pot_idx.shape[0])) * 23

        onion_pos = jnp.array([])
        plate_pos = jnp.array([])
        dish_pos = jnp.array([])

        maze_map = make_overcooked_map(
            wall_map,
            goal_pos,
            agent_pos,
            agent_dir_idx,
            plate_pile_pos,
            onion_pile_pos,
            pot_pos,
            pot_status,
            onion_pos,
            plate_pos,
            dish_pos,
            pad_obs=True,
            num_agents=self.num_agents,
            agent_view_size=self.agent_view_size
        )

        # agent inventory (empty by default, can be randomized)
        key, subkey = jax.random.split(key)
        possible_items = jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['onion'],
                                    OBJECT_TO_INDEX['plate'], OBJECT_TO_INDEX['dish']])

        # inventories: length == num_agents
        default_inv = jnp.full((self.num_agents,), OBJECT_TO_INDEX["empty"])
        random_agent_inv = jax.random.choice(subkey, possible_items, shape=(self.num_agents,), replace=True)
        agent_inv = jnp.where(self.random_reset, random_agent_inv, default_inv)

        state = State(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
            goal_pos=goal_pos,
            pot_pos=pot_pos,
            wall_map=wall_map.astype(jnp.bool_),
            maze_map=maze_map,
            time=0,
            terminal=False,
            task_id=self.task_id
        )

        obs = self.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def process_interact(
            self,
            maze_map: chex.Array,
            wall_map: chex.Array,
            fwd_pos_all: chex.Array,
            inventory_all: chex.Array,
            player_idx: int,
            agent_pos_prev: chex.Array,
            agent_pos_curr: chex.Array,
            agent_dir_idx_all: chex.Array,
            pot_pos: chex.Array,
    ) -> Tuple[chex.Array, int, float, float]:
        """Assume agent took interact actions. Result depends on what agent is facing and what it is holding."""

        fwd_pos = fwd_pos_all[player_idx]
        inventory = inventory_all[player_idx]

        shaped_reward = 0.

        height = self.obs_shape[1]
        padding = (maze_map.shape[0] - height) // 2

        # Get object in front of agent (on the "table")
        maze_object_on_table = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0]].get()
        object_on_table = maze_object_on_table[0]  # Simple index

        # Booleans depending on what the object is
        object_is_pile = jnp.logical_or(object_on_table == OBJECT_TO_INDEX["plate_pile"],
                                        object_on_table == OBJECT_TO_INDEX["onion_pile"])
        object_is_pot = jnp.array(object_on_table == OBJECT_TO_INDEX["pot"])
        object_is_goal = jnp.array(object_on_table == OBJECT_TO_INDEX["goal"])
        object_is_agent = jnp.array(object_on_table == OBJECT_TO_INDEX["agent"])
        object_is_pickable = jnp.logical_or(
            jnp.logical_or(object_on_table == OBJECT_TO_INDEX["plate"], object_on_table == OBJECT_TO_INDEX["onion"]),
            object_on_table == OBJECT_TO_INDEX["dish"]
        )
        # Whether the object in front is counter space that the agent can drop on.
        is_table = jnp.logical_and(wall_map.at[fwd_pos[1], fwd_pos[0]].get(), ~object_is_pot)

        table_is_empty = jnp.logical_or(object_on_table == OBJECT_TO_INDEX["wall"],
                                        object_on_table == OBJECT_TO_INDEX["empty"])

        # Pot status (used if the object is a pot)
        pot_status = maze_object_on_table[-1]

        # Get inventory object, and related booleans
        inv_is_empty = jnp.array(inventory == OBJECT_TO_INDEX["empty"])
        object_in_inv = inventory
        holding_onion = jnp.array(object_in_inv == OBJECT_TO_INDEX["onion"])
        holding_plate = jnp.array(object_in_inv == OBJECT_TO_INDEX["plate"])
        holding_dish = jnp.array(object_in_inv == OBJECT_TO_INDEX["dish"])

        # Interactions with pot. 3 cases: add onion if missing, collect soup if ready, do nothing otherwise
        case_1 = (pot_status > POT_FULL_STATUS) * holding_onion * object_is_pot
        case_2 = (pot_status == POT_READY_STATUS) * holding_plate * object_is_pot
        case_3 = (pot_status > POT_READY_STATUS) * (pot_status <= POT_FULL_STATUS) * object_is_pot
        else_case = ~case_1 * ~case_2 * ~case_3

        # give reward for placing onion in pot, and for picking up soup
        shaped_reward += case_1 * BASE_REW_SHAPING_PARAMS["PLACEMENT_IN_POT_REW"]
        shaped_reward += case_2 * BASE_REW_SHAPING_PARAMS["SOUP_PICKUP_REWARD"]

        # Update pot status and object in inventory
        new_pot_status = \
            case_1 * (pot_status - 1) \
            + case_2 * POT_EMPTY_STATUS \
            + case_3 * pot_status \
            + else_case * pot_status
        new_object_in_inv = \
            case_1 * OBJECT_TO_INDEX["empty"] \
            + case_2 * OBJECT_TO_INDEX["dish"] \
            + case_3 * object_in_inv \
            + else_case * object_in_inv

        # Interactions with onion/plate piles and objects on counter
        # Pickup if: table, not empty, room in inv & object is not something unpickable (e.g. pot or goal)
        successful_pickup = is_table * ~table_is_empty * inv_is_empty * jnp.logical_or(object_is_pile,
                                                                                       object_is_pickable)
        successful_drop = is_table * table_is_empty * ~inv_is_empty
        successful_delivery = is_table * object_is_goal * holding_dish
        no_effect = jnp.logical_and(jnp.logical_and(~successful_pickup, ~successful_drop), ~successful_delivery)

        # Update object on table
        new_object_on_table = \
            no_effect * object_on_table \
            + successful_delivery * object_on_table \
            + successful_pickup * object_is_pile * object_on_table \
            + successful_pickup * object_is_pickable * OBJECT_TO_INDEX["wall"] \
            + successful_drop * object_in_inv

        # Update object in inventory
        new_object_in_inv = \
            no_effect * new_object_in_inv \
            + successful_delivery * OBJECT_TO_INDEX["empty"] \
            + successful_pickup * object_is_pickable * object_on_table \
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["plate_pile"]) * OBJECT_TO_INDEX["plate"] \
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["onion_pile"]) * OBJECT_TO_INDEX["onion"] \
            + successful_drop * OBJECT_TO_INDEX["empty"]

        # A drop is successful if the the agent was holding something and now it is not
        drop_occurred = (object_in_inv != OBJECT_TO_INDEX["empty"]) & (new_object_in_inv == OBJECT_TO_INDEX["empty"])
        # and if the new object on table is the same as the one in the inventory
        object_placed = new_object_on_table == object_in_inv
        # A drop is successful if both of the above are true and the conditions for a drop are met
        successfully_dropped_object = drop_occurred * object_placed * successful_drop
        shaped_reward += successfully_dropped_object * BASE_REW_SHAPING_PARAMS["DROP_COUNTER_REWARD"]

        # Apply inventory update
        has_picked_up_plate = successful_pickup * (new_object_in_inv == OBJECT_TO_INDEX["plate"])

        # number of plates in player hands < number ready/cooking/partially full pot
        num_plates_in_inv = jnp.sum(inventory == OBJECT_TO_INDEX["plate"])
        pot_loc_layer = jnp.array(maze_map[padding:-padding, padding:-padding, 0] == OBJECT_TO_INDEX["pot"],
                                  dtype=jnp.uint8)
        padded_map = maze_map[padding:-padding, padding:-padding, 2]
        num_notempty_pots = jnp.sum((padded_map != POT_EMPTY_STATUS) * pot_loc_layer)
        is_dish_pickup_useful = num_plates_in_inv < num_notempty_pots

        shaped_reward += has_picked_up_plate * is_dish_pickup_useful * BASE_REW_SHAPING_PARAMS[
            "PLATE_PICKUP_REWARD"]

        inventory = new_object_in_inv

        # Apply changes to maze
        new_maze_object_on_table = \
            object_is_pot * OBJECT_INDEX_TO_VEC[new_object_on_table].at[-1].set(new_pot_status) \
            + ~object_is_pot * ~object_is_agent * OBJECT_INDEX_TO_VEC[new_object_on_table] \
            + object_is_agent * maze_object_on_table

        maze_map = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0], :].set(new_maze_object_on_table)

        # Reward of 20 for a soup delivery
        reward = jnp.array(successful_delivery, dtype=float) * DELIVERY_REWARD
        return maze_map, inventory, reward, shaped_reward

    def is_terminal(self, state: State) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= self.max_steps
        return done_steps | state.terminal

    def get_eval_solved_rate_fn(self):
        def _fn(ep_stats):
            return ep_stats['return'] > 0

        return _fn

    @property
    def name(self) -> str:
        """Environment name."""
        return self.layout_name

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, agent_id="") -> spaces.Discrete:
        """Action space of the environment. Agent_id not used since action_space is uniform for all agents"""
        return spaces.Discrete(
            len(self.action_set),
            dtype=jnp.uint32
        )

    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, self.obs_shape)

    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        h = self.height
        w = self.width
        agent_view_size = self.agent_view_size
        return spaces.Dict({
            "agent_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "agent_dir": spaces.Discrete(4),
            "goal_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "maze_map": spaces.Box(0, 255, (w + agent_view_size, h + agent_view_size, 3), dtype=jnp.uint32),
            "time": spaces.Discrete(self.max_steps),
            "terminal": spaces.Discrete(2),
        })

    def max_steps(self) -> int:
        return self.max_steps
