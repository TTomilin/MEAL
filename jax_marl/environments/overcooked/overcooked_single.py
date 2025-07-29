from enum import IntEnum
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.core.frozen_dict import FrozenDict
from jax import lax

from jax_marl.environments import spaces
from jax_marl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    OBJECT_INDEX_TO_VEC,
    DIR_TO_VEC,
    make_overcooked_map)
from jax_marl.environments.overcooked.layouts import overcooked_layouts as layouts, layout_grid_to_dict

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
    done = 6


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


# Pot status constants
POT_READY_STATUS = 0
MAX_ONIONS_IN_POT = 3  # A pot has at most 3 onions. A soup contains exactly 3 onions.

# Default soup cooking time (can be overridden in constructor)
DEFAULT_SOUP_COOK_TIME = 20

URGENCY_CUTOFF = 40  # When this many time steps remain, the urgency layer is flipped on
DELIVERY_REWARD = 20


class OvercookedSingle:
    """Single Agent Overcooked Environment"""

    def __init__(
            self,
            layout=None,
            layout_name="cramped_room",
            random_reset: bool = False,
            max_steps: int = 400,
            task_id: int = 0,
            soup_cook_time: int = DEFAULT_SOUP_COOK_TIME,
            num_agents: int = 1,
    ):
        # Hardcode to single agent
        self.num_agents = 1

        # Convert string layout to dictionary if needed
        if isinstance(layout, str):
            layout = layout_grid_to_dict(layout)
        elif layout is None:
            layout = FrozenDict(layouts["cramped_room"])

        # Observations given by 26 channels, most of which are boolean masks
        self.height = layout["height"]
        self.width = layout["width"]
        self.obs_shape = (self.width, self.height, 26)

        self.agent_view_size = 5  # Hard coded. Only affects map padding -- not observations.
        self.layout = layout
        self.layout_name = layout_name
        self.agents = ["agent_0"]

        self.action_set = jnp.array([
            Actions.up,
            Actions.down,
            Actions.right,
            Actions.left,
            Actions.stay,
            Actions.interact,
        ])

        self.random_reset = random_reset
        self.max_steps = max_steps
        self.task_id = task_id

        # Configure soup cooking time
        self.soup_cook_time = soup_cook_time
        self.pot_full_status = soup_cook_time
        self.pot_empty_status = soup_cook_time + MAX_ONIONS_IN_POT

    def step(
            self,
            key: chex.PRNGKey,
            state: State,
            action,
    ) -> Tuple[chex.Array, State, float, bool, dict]:
        """Perform single timestep state transition."""

        # Handle both direct actions and dictionary actions (for compatibility with multi-agent interfaces)
        if isinstance(action, dict):
            # Extract action for the single agent
            action_value = action[self.agents[0]]
        else:
            # Direct action value
            action_value = action

        act = self.action_set.take(indices=action_value)

        state, reward, shaped_reward, soups_delivered = self.step_agent(key, state, act)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)
        info = {
            'shaped_reward': shaped_reward,
            'soups': soups_delivered,
        }

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset(
            self,
            key: chex.PRNGKey,
    ) -> Tuple[chex.Array, State]:
        """Reset environment state based on `self.random_reset`"""

        # Whether to fully randomize the start state
        random_reset = self.random_reset
        layout = self.layout

        h = self.height
        w = self.width
        num_agents = 1  # Hardcoded for single agent
        all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)

        wall_idx = layout.get("wall_idx")

        occupied_mask = jnp.zeros_like(all_pos)
        occupied_mask = occupied_mask.at[wall_idx].set(1)
        wall_map = occupied_mask.reshape(h, w).astype(jnp.bool_)

        # Reset agent position + dir (single agent)
        key, subkey = jax.random.split(key)
        agent_idx = jax.random.choice(subkey, all_pos, shape=(1,),
                                      p=(~occupied_mask.astype(jnp.bool_)).astype(jnp.float32), replace=False)

        # Replace with fixed layout if applicable
        agent_idx = random_reset * agent_idx + (1 - random_reset) * layout.get("agent_idx", agent_idx)[:1]  # Take only first agent
        agent_pos = jnp.array([agent_idx[0] % w, agent_idx[0] // w], dtype=jnp.uint32).reshape(1, 2)  # Single agent
        occupied_mask = occupied_mask.at[agent_idx].set(1)

        key, subkey = jax.random.split(key)
        agent_dir_idx = jax.random.choice(subkey, jnp.arange(len(DIR_TO_VEC), dtype=jnp.int32), shape=(1,))
        agent_dir = DIR_TO_VEC.at[agent_dir_idx].get()

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
        pot_status = jax.random.randint(subkey, (pot_idx.shape[0],), 0, self.pot_empty_status + 1)
        pot_status = pot_status * random_reset + (1 - random_reset) * jnp.ones((pot_idx.shape[0])) * self.pot_empty_status

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
            num_agents=1,  # Single agent
            agent_view_size=self.agent_view_size
        )

        # agent inventory (empty by default, can be randomized)
        key, subkey = jax.random.split(key)
        possible_items = jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['onion'],
                                    OBJECT_TO_INDEX['plate'], OBJECT_TO_INDEX['dish']])
        random_agent_inv = jax.random.choice(subkey, possible_items, shape=(1,), replace=True)
        agent_inv = random_reset * random_agent_inv + \
                    (1 - random_reset) * jnp.array([OBJECT_TO_INDEX['empty']])

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

    def get_obs(self, state: State) -> chex.Array:
        """Return a full observation for single agent"""

        width = self.obs_shape[0]
        height = self.obs_shape[1]
        n_channels = self.obs_shape[2]
        padding = (state.maze_map.shape[0] - height) // 2

        maze_map = state.maze_map[padding:-padding, padding:-padding, 0]
        soup_loc = jnp.array(maze_map == OBJECT_TO_INDEX["dish"], dtype=jnp.uint8)

        pot_loc_layer = jnp.array(maze_map == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8)
        pot_status = state.maze_map[padding:-padding, padding:-padding, 2] * pot_loc_layer
        onions_in_pot_layer = jnp.minimum(self.pot_empty_status - pot_status, MAX_ONIONS_IN_POT) * (
                pot_status >= self.pot_full_status)
        onions_in_soup_layer = jnp.minimum(self.pot_empty_status - pot_status, MAX_ONIONS_IN_POT) * (
                pot_status < self.pot_full_status) \
                               * pot_loc_layer + MAX_ONIONS_IN_POT * soup_loc
        pot_cooking_time_layer = pot_status * (pot_status < self.pot_full_status)
        soup_ready_layer = pot_loc_layer * (pot_status == POT_READY_STATUS) + soup_loc
        urgency_layer = jnp.ones(maze_map.shape, dtype=jnp.uint8) * ((self.max_steps - state.time) < URGENCY_CUTOFF)

        # Single agent position layer
        agent_pos_layer = jnp.zeros((height, width), dtype=jnp.uint8)
        agent_pos_layer = agent_pos_layer.at[state.agent_pos[0, 1], state.agent_pos[0, 0]].set(1)

        # Add agent inv: This works because loose items and agent cannot overlap
        agent_inv_item = state.agent_inv[0] * agent_pos_layer
        maze_map = jnp.where(agent_pos_layer, agent_inv_item, maze_map)
        soup_ready_layer = soup_ready_layer + (agent_inv_item == OBJECT_TO_INDEX["dish"]) * agent_pos_layer
        onions_in_soup_layer = onions_in_soup_layer + (agent_inv_item == OBJECT_TO_INDEX["dish"]) * 3 * agent_pos_layer

        env_layers = [
            jnp.array(maze_map == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8),  # Channel 10
            jnp.array(maze_map == OBJECT_TO_INDEX["wall"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["onion_pile"], dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),  # tomato pile
            jnp.array(maze_map == OBJECT_TO_INDEX["plate_pile"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["goal"], dtype=jnp.uint8),  # 15
            jnp.array(onions_in_pot_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),  # tomatoes in pot
            jnp.array(onions_in_soup_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),  # tomatoes in soup
            jnp.array(pot_cooking_time_layer, dtype=jnp.uint8),  # 20
            jnp.array(soup_ready_layer, dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["onion"], dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),  # tomatoes
            urgency_layer,  # 25
        ]

        # Agent direction layers (simplified for single agent)
        agent_direction_layers = jnp.zeros((8, height, width), dtype=jnp.uint8)
        # Only one agent, so we set its direction in the first 4 channels
        agent_direction_layers = agent_direction_layers.at[state.agent_dir_idx[0], :, :].set(agent_pos_layer)
        # The other agent channels (4-7) remain zero since there's no second agent

        # Single agent observation
        obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
        obs = obs.at[0].set(agent_pos_layer)  # Agent position
        obs = obs.at[1].set(jnp.zeros((height, width), dtype=jnp.uint8))  # No second agent
        obs = obs.at[2:10].set(agent_direction_layers)
        obs = obs.at[10:].set(jnp.stack(env_layers))

        obs = jnp.transpose(obs, (1, 2, 0))

        return obs

    def step_agent(
            self, key: chex.PRNGKey, state: State, action: chex.Array,
    ) -> Tuple[State, float, float, float]:
        """Process single agent action and compute rewards."""

        # Update agent position (forward action)
        is_move_action = jnp.logical_and(action != Actions.stay, action != Actions.interact)

        fwd_pos = jnp.minimum(
            jnp.maximum(state.agent_pos[0] + is_move_action * DIR_TO_VEC[jnp.minimum(action, 3)] \
                        + ~is_move_action * state.agent_dir[0], 0),
            jnp.array((self.width - 1, self.height - 1), dtype=jnp.uint32)
        )

        # Can't go past wall or goal
        fwd_wall = state.wall_map.at[fwd_pos[1], fwd_pos[0]].get()
        goal_collision = jnp.any(jnp.logical_and(fwd_pos[0] == state.goal_pos[:, 0], fwd_pos[1] == state.goal_pos[:, 1]))
        fwd_pos_blocked = jnp.logical_or(fwd_wall, goal_collision)

        bounced = jnp.logical_or(fwd_pos_blocked, ~is_move_action)

        # Update position
        agent_pos = jnp.where(bounced, state.agent_pos[0], fwd_pos).astype(jnp.uint32).reshape(1, 2)

        # Update agent direction
        agent_dir_idx = jnp.where(~is_move_action, state.agent_dir_idx, action).reshape(1)
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        # Handle interact
        fwd_pos = state.agent_pos[0] + state.agent_dir[0]
        maze_map = state.maze_map
        is_interact_action = (action == Actions.interact)

        # Compute the effect of interact
        candidate_maze_map, agent_inv, reward, shaped_reward = self.process_interact(maze_map,
                                                                                     state.wall_map,
                                                                                     fwd_pos,
                                                                                     state.agent_inv[0])

        maze_map = jax.lax.select(is_interact_action, candidate_maze_map, maze_map)
        agent_inv = jax.lax.select(is_interact_action, agent_inv, state.agent_inv[0])
        reward = jax.lax.select(is_interact_action, reward, 0.)
        shaped_reward = jax.lax.select(is_interact_action, shaped_reward, 0.)

        soups_delivered = reward / DELIVERY_REWARD

        agent_inv = agent_inv.reshape(1)

        # Update agent component in maze_map
        agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], agent_dir_idx[0]], dtype=jnp.uint8)
        agent_x_prev, agent_y_prev = state.agent_pos[0]
        agent_x, agent_y = agent_pos[0]
        empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)

        # Compute padding
        height = self.obs_shape[1]
        padding = (state.maze_map.shape[0] - height) // 2

        maze_map = maze_map.at[padding + agent_y_prev, padding + agent_x_prev, :].set(empty)
        maze_map = maze_map.at[padding + agent_y, padding + agent_x, :].set(agent)

        # Update pot cooking status
        def _cook_pots(pot):
            pot_status = pot[-1]
            is_cooking = jnp.array(pot_status <= self.pot_full_status)
            not_done = jnp.array(pot_status > POT_READY_STATUS)
            pot_status = is_cooking * not_done * (pot_status - 1) + (~is_cooking) * pot_status
            return pot.at[-1].set(pot_status)

        pot_x = state.pot_pos[:, 0]
        pot_y = state.pot_pos[:, 1]
        pots = maze_map.at[padding + pot_y, padding + pot_x].get()
        pots = jax.vmap(_cook_pots, in_axes=0)(pots)
        maze_map = maze_map.at[padding + pot_y, padding + pot_x, :].set(pots)

        return (
            state.replace(
                agent_pos=agent_pos,
                agent_dir_idx=agent_dir_idx,
                agent_dir=agent_dir,
                agent_inv=agent_inv,
                maze_map=maze_map,
                terminal=False),
            reward,
            shaped_reward,
            soups_delivered
        )

    def process_interact(
            self,
            maze_map: chex.Array,
            wall_map: chex.Array,
            fwd_pos: chex.Array,
            inventory: chex.Array):
        """Process interact action for single agent."""

        shaped_reward = 0.

        height = self.obs_shape[1]
        padding = (maze_map.shape[0] - height) // 2

        # Get object in front of agent
        maze_object_on_table = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0]].get()
        object_on_table = maze_object_on_table[0]

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
        is_table = jnp.logical_and(wall_map.at[fwd_pos[1], fwd_pos[0]].get(), ~object_is_pot)

        table_is_empty = jnp.logical_or(object_on_table == OBJECT_TO_INDEX["wall"],
                                        object_on_table == OBJECT_TO_INDEX["empty"])

        # Pot status
        pot_status = maze_object_on_table[-1]

        # Get inventory object
        inv_is_empty = jnp.array(inventory == OBJECT_TO_INDEX["empty"])
        object_in_inv = inventory
        holding_onion = jnp.array(object_in_inv == OBJECT_TO_INDEX["onion"])
        holding_plate = jnp.array(object_in_inv == OBJECT_TO_INDEX["plate"])
        holding_dish = jnp.array(object_in_inv == OBJECT_TO_INDEX["dish"])

        # Interactions with pot
        case_1 = (pot_status > self.pot_full_status) * holding_onion * object_is_pot
        case_2 = (pot_status == POT_READY_STATUS) * holding_plate * object_is_pot
        case_3 = (pot_status > POT_READY_STATUS) * (pot_status <= self.pot_full_status) * object_is_pot
        else_case = ~case_1 * ~case_2 * ~case_3

        # Shaped rewards
        shaped_reward += case_1 * BASE_REW_SHAPING_PARAMS["PLACEMENT_IN_POT_REW"]
        shaped_reward += case_2 * BASE_REW_SHAPING_PARAMS["SOUP_PICKUP_REWARD"]

        # Update pot status and inventory
        new_pot_status = \
            case_1 * (pot_status - 1) \
            + case_2 * self.pot_empty_status \
            + case_3 * pot_status \
            + else_case * pot_status
        new_object_in_inv = \
            case_1 * OBJECT_TO_INDEX["empty"] \
            + case_2 * OBJECT_TO_INDEX["dish"] \
            + case_3 * object_in_inv \
            + else_case * object_in_inv

        # Interactions with piles and objects on counter
        base_pickup_condition = is_table * ~table_is_empty * inv_is_empty * jnp.logical_or(object_is_pile, object_is_pickable)

        successful_pickup = base_pickup_condition
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

        # Drop reward
        drop_occurred = (object_in_inv != OBJECT_TO_INDEX["empty"]) & (new_object_in_inv == OBJECT_TO_INDEX["empty"])
        object_placed = new_object_on_table == object_in_inv
        successfully_dropped_object = drop_occurred * object_placed * successful_drop
        shaped_reward += successfully_dropped_object * BASE_REW_SHAPING_PARAMS["DROP_COUNTER_REWARD"]

        # Plate pickup reward
        has_picked_up_plate = successful_pickup * (new_object_in_inv == OBJECT_TO_INDEX["plate"])
        pot_loc_layer = jnp.array(maze_map[padding:-padding, padding:-padding, 0] == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8)
        padded_map = maze_map[padding:-padding, padding:-padding, 2]
        num_notempty_pots = jnp.sum((padded_map != self.pot_empty_status) * pot_loc_layer)
        is_dish_pickup_useful = inventory != OBJECT_TO_INDEX["plate"]  # Simplified for single agent

        plate_loc_layer = jnp.array(maze_map == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8)
        no_plates_on_counters = jnp.sum(plate_loc_layer) == 0

        shaped_reward += no_plates_on_counters * has_picked_up_plate * is_dish_pickup_useful * BASE_REW_SHAPING_PARAMS["PLATE_PICKUP_REWARD"]

        inventory = new_object_in_inv

        # Apply changes to maze
        new_maze_object_on_table = \
            object_is_pot * OBJECT_INDEX_TO_VEC[new_object_on_table].at[-1].set(new_pot_status) \
            + ~object_is_pot * ~object_is_agent * OBJECT_INDEX_TO_VEC[new_object_on_table] \
            + object_is_agent * maze_object_on_table

        maze_map = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0], :].set(new_maze_object_on_table)

        # Delivery reward
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
        return f"{self.layout_name}_single"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set), dtype=jnp.uint32)

    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, self.obs_shape)

    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        h = self.height
        w = self.width
        agent_view_size = self.agent_view_size
        return spaces.Dict({
            "agent_pos": spaces.Box(0, max(w, h), (1, 2), dtype=jnp.uint32),  # Single agent
            "agent_dir": spaces.Discrete(4),
            "goal_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "maze_map": spaces.Box(0, 255, (w + agent_view_size, h + agent_view_size, 3), dtype=jnp.uint32),
            "time": spaces.Discrete(self.max_steps),
            "terminal": spaces.Discrete(2),
        })

    @property
    def max_steps_property(self) -> int:
        return self.max_steps
