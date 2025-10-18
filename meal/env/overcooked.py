import json
from enum import IntEnum
from typing import Tuple, Dict

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.core.frozen_dict import FrozenDict
from jax import lax

from meal.env import MultiAgentEnv
from meal.env.common import (
    OBJECT_TO_INDEX,
    OBJECT_INDEX_TO_VEC,
    DIR_TO_VEC,
    make_overcooked_map)
from meal.env.generation.layout_generator import generate_random_layout
from meal.env.layouts.presets import overcooked_layouts as layouts, _parse_layout_string
from meal.env.utils import spaces

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
    agent_pos: chex.Array
    agent_dir: chex.Array
    agent_dir_idx: chex.Array
    agent_inv: chex.Array
    goal_pos: chex.Array
    pot_pos: chex.Array
    wall_map: chex.Array
    maze_map: chex.Array
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
            layout=None,
            layout_name=None,
            difficulty: str = 'easy',
            random_reset: bool = False,
            random_agent_start: bool = False,
            max_steps: int = 1000,
            task_id: int = 0,
            num_agents: int = 2,
            agent_restrictions: dict = None,
            **env_kwargs
    ):
        super().__init__(num_agents=num_agents)

        # 1) explicit layout given
        if layout is not None:
            self.layout = layout if isinstance(layout, FrozenDict) else FrozenDict(layout)
            self.layout_name = layout_name or "custom"
        # 2) named preset layout given
        elif layout_name is not None:
            if layout_name not in layouts:
                raise ValueError(f"Unknown layout_name '{layout_name}'. Available: {sorted(layouts.keys())}")
            self.layout = FrozenDict(layouts[layout_name])
            self.layout_name = layout_name

            names = [f"file_{i}" for i in range(len(env_kwargs))]
            self.layout_name = names[task_id]
        # 3) otherwise: generate by difficulty
        else:
            grid, self.layout = generate_random_layout(num_agents=num_agents, difficulty=difficulty, **env_kwargs)
            self.layout_name = f"{difficulty}_gen_{task_id}"

        # Observations given by 26 channels, most of which are boolean masks
        self.height = self.layout["height"]
        self.width = self.layout["width"]
        self.env_layers = 16  # Number of environment layers (static, dynamic, pot, soup, etc.)
        self.obs_channels = 18 + 4 * num_agents  # 26 when n = 2
        self.obs_shape = (self.width, self.height, self.obs_channels)

        self.difficulty = difficulty
        self.agents = [f"agent_{i}" for i in range(num_agents)]

        self.action_set = jnp.array([
            Actions.up,
            Actions.down,
            Actions.right,
            Actions.left,
            Actions.stay,
            Actions.interact,
        ], dtype=jnp.uint8)

        self.random_reset = random_reset
        self.random_agent_start = random_agent_start
        self.max_steps = max_steps
        self.task_id = task_id
        self.agent_restrictions = agent_restrictions or {}

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
        maze = state.maze_map  # (H,W,3)

        # ────────────────────── build the 16 env layers ────────────────────────
        obj = maze[:, :, 0]  # tile indices
        pot_mask = (obj == OBJECT_TO_INDEX["pot"])
        dish_mask = (obj == OBJECT_TO_INDEX["dish"])

        pot_status = maze[:, :, 2] * pot_mask  # 0–23   at pot tiles

        onions_in_pot = jnp.minimum(POT_EMPTY_STATUS - pot_status,
                                    MAX_ONIONS_IN_POT) * (pot_status >= POT_FULL_STATUS)

        onions_in_soup = (jnp.minimum(POT_EMPTY_STATUS - pot_status,
                                      MAX_ONIONS_IN_POT) * (pot_status < POT_FULL_STATUS)
                          * pot_mask + MAX_ONIONS_IN_POT * dish_mask)

        pot_cook_time = pot_status * (pot_status < POT_FULL_STATUS)
        soup_ready = pot_mask * (pot_status == POT_READY_STATUS) + dish_mask
        urgency = jnp.ones_like(obj, jnp.uint8) * ((self.max_steps - state.time) < URGENCY_CUTOFF)

        # ────────────────────── agent-specific layers ──────────────────────────
        pos_layers = self._pos_layers(state)  # (n,H,W)

        # orientation one-hot layers
        ori_layers = jnp.zeros((4 * self.num_agents, H, W), jnp.uint8)
        idx = jnp.arange(self.num_agents)
        ori_layers = ori_layers.at[4 * idx + state.agent_dir_idx, :, :].set(
            pos_layers)

        # ────────────────────── assemble per-agent views ───────────────────────
        # Add agent inventory handling to environment layers
        agent_inv_items = jnp.expand_dims(state.agent_inv, (1, 2)) * pos_layers
        obj_with_inv = jnp.where(jnp.sum(pos_layers, 0), jnp.sum(agent_inv_items, 0), obj)
        soup_ready_with_inv = soup_ready + (jnp.sum(agent_inv_items, 0) == OBJECT_TO_INDEX["dish"]) * jnp.sum(
            pos_layers, 0)
        onions_in_soup_with_inv = onions_in_soup + (
                jnp.sum(agent_inv_items, 0) == OBJECT_TO_INDEX["dish"]) * 3 * jnp.sum(pos_layers, 0)

        # Rebuild env_layers with agent inventory
        env_layers_with_inv = jnp.stack([
            pot_mask.astype(jnp.uint8),  # 10
            (obj_with_inv == OBJECT_TO_INDEX["wall"]).astype(jnp.uint8),
            (obj_with_inv == OBJECT_TO_INDEX["onion_pile"]).astype(jnp.uint8),
            jnp.zeros_like(obj, jnp.uint8),  # tomato‐pile (unused)
            (obj_with_inv == OBJECT_TO_INDEX["plate_pile"]).astype(jnp.uint8),
            (obj_with_inv == OBJECT_TO_INDEX["goal"]).astype(jnp.uint8),  # 15
            onions_in_pot.astype(jnp.uint8),
            jnp.zeros_like(obj, jnp.uint8),  # tomatoes in pot (unused)
            onions_in_soup_with_inv.astype(jnp.uint8),
            jnp.zeros_like(obj, jnp.uint8),  # tomatoes in soup (unused)
            pot_cook_time.astype(jnp.uint8),  # 20
            soup_ready_with_inv.astype(jnp.uint8),
            (obj_with_inv == OBJECT_TO_INDEX["plate"]).astype(jnp.uint8),
            (obj_with_inv == OBJECT_TO_INDEX["onion"]).astype(jnp.uint8),
            jnp.zeros_like(obj, jnp.uint8),  # tomatoes (unused)
            urgency.astype(jnp.uint8),  # 25
        ], axis=0)  # → (16,H,W)

        # Generic n-agent observation structure
        views: Dict[str, chex.Array] = {}
        for i in range(self.num_agents):
            own_pos = pos_layers[i:i + 1]  # 1 layer
            others_pos = jnp.delete(pos_layers, i, axis=0)
            own_ori = ori_layers[4 * i:4 * (i + 1)]  # 4 layers
            others_ori = jnp.delete(ori_layers,
                                    slice(4 * i, 4 * (i + 1)), axis=0)  # 4(n-1)

            # Structure observations: [own_pos, others_pos, own_ori, others_ori, env_layers]
            # For 2 agents: others_pos is (1,H,W), for n>2: others_pos is aggregated to (1,H,W)
            if others_pos.shape[0] == 1:
                # Single other agent - use as is
                other_pos_layer = others_pos
            else:
                # Multiple other agents - aggregate their positions
                other_pos_layer = others_pos.sum(0, keepdims=True)

            layers = jnp.concatenate([
                own_pos,  # 1 layer: own position
                other_pos_layer,  # 1 layer: other agent(s) position
                own_ori,  # 4 layers: own orientation
                others_ori,  # 4(n-1) layers: other agents' orientations
                env_layers_with_inv,  # 16 layers: environment
            ], axis=0)

            views[f"agent_{i}"] = jnp.transpose(layers, (1, 2, 0))  # (H,W,C)

        return views

    # ───────────────────────── movement / step ────────────────────

    def _proposed_positions(self, state: State, action: chex.Array):
        # Match the original overcooked.py logic exactly
        is_move_action = jnp.logical_and(action != Actions.stay, action != Actions.interact)
        is_move_action_transposed = jnp.expand_dims(is_move_action, 0).transpose()

        # Calculate proposed positions like original
        fwd_pos = jnp.minimum(
            jnp.maximum(state.agent_pos + is_move_action_transposed * DIR_TO_VEC[jnp.minimum(action, 3)] \
                        + ~is_move_action_transposed * state.agent_dir, 0),
            jnp.array((self.width - 1, self.height - 1), dtype=jnp.uint32)
        )

        # Can't go past wall or goal - match original logic
        def _wall_or_goal(fwd_position, wall_map, goal_pos):
            fwd_wall = wall_map.at[fwd_position[1], fwd_position[0]].get()
            goal_collision = lambda pos, goal: jnp.logical_and(pos[0] == goal[0], pos[1] == goal[1])
            fwd_goal = jax.vmap(goal_collision, in_axes=(None, 0))(fwd_position, goal_pos)
            fwd_goal = jnp.any(fwd_goal)
            return jnp.asarray(fwd_wall), jnp.asarray(fwd_goal)

        fwd_pos_has_wall, fwd_pos_has_goal = jax.vmap(_wall_or_goal, in_axes=(0, None, None))(fwd_pos, state.wall_map,
                                                                                              state.goal_pos)
        fwd_pos_blocked = jnp.logical_or(fwd_pos_has_wall, fwd_pos_has_goal)
        fwd_pos_blocked = fwd_pos_blocked.flatten()[:self.num_agents]
        fwd_pos_blocked = fwd_pos_blocked.reshape((self.num_agents, 1))

        bounced = jnp.logical_or(fwd_pos_blocked, ~is_move_action_transposed)
        proposed = (bounced * state.agent_pos + (~bounced) * fwd_pos).astype(jnp.uint32)

        return proposed

    def _resolve_collisions(self, current, proposed):
        n = current.shape[0]

        # For 2 agents, match the original overcooked.py logic exactly
        if n == 2:
            # Check for collision (both agents going to same position)
            collision = jnp.all(proposed[0] == proposed[1])

            # If collision, both agents stay in place
            alice_pos = jnp.where(collision, current[0], proposed[0])
            bob_pos = jnp.where(collision, current[1], proposed[1])

            # Check for swapping places (agents passing through each other)
            swap_places = jnp.logical_and(
                jnp.all(proposed[0] == current[1]),
                jnp.all(proposed[1] == current[0]),
            )

            # If swapping and no collision, prevent the swap
            alice_pos = jnp.where(~collision & swap_places, current[0], alice_pos)
            bob_pos = jnp.where(~collision & swap_places, current[1], bob_pos)

            return jnp.array([alice_pos, bob_pos]).astype(jnp.uint32)

        else:
            # For n > 2 agents, use the original generic logic
            # same destination (collision)  ────────────────────────────────
            same_dest = (proposed[:, None, :] == proposed[None, :, :]).all(-1)
            coll = (same_dest.sum(-1) > 1)  # True if ≥2 agents share a tile

            # swap places and circular movements ──────────────────────────
            if n == 1:  # no other agents → no swap test
                blocked = coll
            else:
                # Check for direct pairwise swaps (i ↔ j)
                swap = ((proposed[:, None, :] == current[None, :, :]).all(-1) &
                        (proposed[None, :, :] == current[:, None, :]).all(-1))
                # ignore i==j diagonal ─ we only care about pairs (i ≠ j)
                swap = swap & (~jnp.eye(n, dtype=bool))

                # Check for circular movements (A→B→C→A, etc.)
                # An agent is part of a circular movement if it's moving to another agent's 
                # current position AND that other agent is also moving

                # Find agents that are actually moving (not staying in place)
                is_moving = ~(proposed == current).all(-1)

                # Create a matrix where entry (i,j) is True if agent i is moving to agent j's current position
                moving_to_agent_pos = (proposed[:, None, :] == current[None, :, :]).all(-1)
                moving_to_agent_pos = moving_to_agent_pos & (~jnp.eye(n, dtype=bool))  # ignore self

                # An agent is blocked if it's moving to another agent's position AND that other agent is also moving
                # This prevents both simple swaps and circular movements
                # For each agent, check if it's moving to any other moving agent's current position
                targets_moving_agents = moving_to_agent_pos & is_moving[None, :]  # broadcast is_moving
                circular_blocked = targets_moving_agents.any(axis=1)  # True if agent targets any moving agent

                blocked = coll | swap.any(-1) | circular_blocked

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
            maze, inv, rew, shaped, soups = carry
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
            rew = rew.at[idx].add(r_i)
            shaped = shaped.at[idx].set(s_i)
            # Calculate soups delivered by this agent
            soups_delivered = r_i / DELIVERY_REWARD
            soups = soups.at[idx].set(soups_delivered)
            return (maze_new, inv, rew, shaped, soups), None

        init_carry = (state.maze_map, state.agent_inv,
                      jnp.zeros((self.num_agents,), jnp.float32),
                      jnp.zeros((self.num_agents,), jnp.float32),
                      jnp.zeros((self.num_agents,), jnp.float32))

        (maze_map, agent_inv, reward, shaped_r, soups_delivered), _ = lax.scan(body, init_carry,
                                                                               jnp.arange(self.num_agents))

        # ─── tick every pot exactly once per env-step ────────────────────────
        pot_x, pot_y = state.pot_pos[:, 0], state.pot_pos[:, 1]

        def _tick(pot):
            status = pot[-1]
            cooking = (status <= POT_FULL_STATUS) & (status > POT_READY_STATUS)
            return pot.at[-1].set(jnp.where(cooking, status - 1, status))

        pots = jax.vmap(_tick)(maze_map[pot_y, pot_x])
        maze_map = maze_map.at[pot_y, pot_x, :].set(pots)

        # ─── repaint agents (always, not only on INTERACT) ───────────────────
        empty_vec = OBJECT_INDEX_TO_VEC[OBJECT_TO_INDEX["empty"]]
        maze_map = maze_map.at[state.agent_pos[:, 1], state.agent_pos[:, 0], :].set(empty_vec)

        def _agent_vec(dir_idx, idx):
            return jnp.array([OBJECT_TO_INDEX["agent"], 2 * idx, dir_idx], dtype=jnp.uint8)

        agent_tiles = jax.vmap(_agent_vec)(agent_dir_idx, jnp.arange(self.num_agents))
        maze_map = maze_map.at[agent_pos[:, 1], agent_pos[:, 0], :].set(agent_tiles)

        new_state = state.replace(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
            maze_map=maze_map,
            terminal=False,
        )
        return new_state, reward, shaped_r, soups_delivered

    def step_env(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""

        # convert incoming dict → jnp.array([a0,a1,…]) and apply action_set.take() like original
        if isinstance(actions, dict):
            action_indices = jnp.array([actions[a].flatten()[0] for a in self.agents], dtype=jnp.uint8)
        else:
            action_indices = actions

        # Use action_set.take() to convert indices to action values, matching original behavior
        act_arr = self.action_set.take(indices=action_indices)

        state, reward, shaped_rewards, soups_delivered = self.step_agents(key, state, act_arr)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)

        # package outputs back into dict form
        rew_dict = {a: reward[i] for i, a in enumerate(self.agents)}

        # shaped reward is already a length-n vector
        shaped_dict = {a: shaped_rewards[i] for i, a in enumerate(self.agents)}
        # Add soups dictionary
        soups_dict = {a: soups_delivered[i] for i, a in enumerate(self.agents)}
        done_dict = {a: done for a in self.agents} | {"__all__": done}
        info = {'shaped_reward': shaped_dict, 'soups': soups_dict}

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rew_dict,
            done_dict,
            info,
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

        # Agent positioning - match original overcooked.py logic
        wall_map = occupied.reshape(h, w).astype(jnp.bool_)

        # Reset agent position + dir - like original
        key, subkey = jax.random.split(key)
        agent_idx = jax.random.choice(subkey, all_pos, shape=(num_agents,),
                                      p=(~occupied.astype(jnp.bool_)).astype(jnp.float32), replace=False)

        # Replace with fixed layout if applicable - like original
        layout_agent_idx = layout.get("agent_idx", jnp.array([], dtype=jnp.uint32))
        if not random_reset and len(layout_agent_idx) > 0:
            if len(layout_agent_idx) >= num_agents:
                # Use the first num_agents positions from the layout
                agent_idx = layout_agent_idx[:num_agents]

                # If there are unused agent spawn positions in the layout, treat them as walls/counters
                if len(layout_agent_idx) > num_agents:
                    unused_agent_positions = layout_agent_idx[num_agents:]
                    # Add unused agent spawn positions to the wall map so they appear as counters
                    wall_map = wall_map.at[unused_agent_positions // w, unused_agent_positions % w].set(True)
                    occupied = occupied.at[unused_agent_positions].set(1)
            else:
                # Layout has fewer agent positions than requested
                # Use all available layout positions and generate random positions for the rest
                available_positions = layout_agent_idx
                needed_positions = num_agents - len(layout_agent_idx)

                # Mark layout positions as occupied
                occupied = occupied.at[available_positions].set(1)

                # Generate random positions for additional agents
                key, subkey = jax.random.split(key)
                additional_positions = jax.random.choice(
                    subkey, all_pos, shape=(needed_positions,),
                    p=(~occupied.astype(jnp.bool_)).astype(jnp.float32), replace=False
                )

                # Combine layout and random positions
                agent_idx = jnp.concatenate([available_positions, additional_positions])

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
        # Pot status is determined by a number between 0 (inclusive) and pot_empty_status+1 (exclusive)
        # pot_empty_status corresponds to an empty pot (default 23, or 8 for tests with soup_cook_time=5)
        pot_status = jax.random.randint(subkey, (pot_idx.shape[0],), 0, POT_EMPTY_STATUS + 1)
        pot_status = pot_status * random_reset + (1 - random_reset) * jnp.ones((pot_idx.shape[0])) * POT_EMPTY_STATUS

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
            num_agents=self.num_agents,
        )

        # agent inventory (empty by default, can be randomized)
        key, subkey = jax.random.split(key)
        possible_items = jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['onion'],
                                    OBJECT_TO_INDEX['plate'], OBJECT_TO_INDEX['dish']])
        random_agent_inv = jax.random.choice(subkey, possible_items, shape=(num_agents,), replace=True)
        agent_inv = random_reset * random_agent_inv + \
                    (1 - random_reset) * jnp.full((num_agents,), OBJECT_TO_INDEX['empty'])

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

        # Get object in front of agent (on the "table")
        maze_object_on_table = maze_map.at[fwd_pos[1], fwd_pos[0]].get()
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
        base_pickup_condition = is_table * ~table_is_empty * inv_is_empty * jnp.logical_or(object_is_pile,
                                                                                           object_is_pickable)

        # Apply agent restrictions if they exist
        agent_key = f"agent_{player_idx}"
        can_pick_onions = jnp.array(True)
        can_pick_plates = jnp.array(True)

        if self.agent_restrictions:
            can_pick_onions = jnp.array(not self.agent_restrictions.get(
                f"{agent_key}_cannot_pick_onions", False))
            can_pick_plates = jnp.array(not self.agent_restrictions.get(
                f"{agent_key}_cannot_pick_plates", False))

        # Check if the object being picked up is restricted
        picking_up_onion = jnp.logical_or(object_on_table == OBJECT_TO_INDEX["onion_pile"],
                                          object_on_table == OBJECT_TO_INDEX["onion"])
        picking_up_plate = jnp.logical_or(object_on_table == OBJECT_TO_INDEX["plate_pile"],
                                          object_on_table == OBJECT_TO_INDEX["plate"])

        # Agent can pick up the object if not restricted
        can_pick_this_object = jnp.logical_or(
            jnp.logical_and(picking_up_onion, can_pick_onions),
            jnp.logical_or(
                jnp.logical_and(picking_up_plate, can_pick_plates),
                jnp.logical_and(~picking_up_onion, ~picking_up_plate)  # Other objects (dishes) are always allowed
            )
        )

        successful_pickup = base_pickup_condition * can_pick_this_object
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
        pot_loc_layer = jnp.array(maze_map[..., 0] == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8)
        padded_map = maze_map[..., 2]
        num_notempty_pots = jnp.sum((padded_map != POT_EMPTY_STATUS) * pot_loc_layer)
        is_dish_pickup_useful = num_plates_in_inv < num_notempty_pots

        plate_loc_layer = (maze_map == OBJECT_TO_INDEX["plate"]).astype(jnp.uint8)
        no_plates_on_counters = jnp.sum(plate_loc_layer) == 0

        shaped_reward += no_plates_on_counters * has_picked_up_plate * is_dish_pickup_useful * BASE_REW_SHAPING_PARAMS[
            "PLATE_PICKUP_REWARD"]

        inventory = new_object_in_inv

        # Apply changes to maze
        new_maze_object_on_table = \
            object_is_pot * OBJECT_INDEX_TO_VEC[new_object_on_table].at[-1].set(new_pot_status) \
            + ~object_is_pot * ~object_is_agent * OBJECT_INDEX_TO_VEC[new_object_on_table] \
            + object_is_agent * maze_object_on_table

        maze_map = maze_map.at[fwd_pos[1], fwd_pos[0], :].set(new_maze_object_on_table)

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
        return spaces.Dict({
            "agent_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "agent_dir": spaces.Discrete(4),
            "goal_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "maze_map": spaces.Box(0, 255, (w, h, 3), dtype=jnp.uint32),
            "time": spaces.Discrete(self.max_steps),
            "terminal": spaces.Discrete(2),
        })
