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


class Overcooked(MultiAgentEnv):
    """Vanilla Overcooked"""

    def __init__(
            self,
            layout=None,
            layout_name="cramped_room",
            random_reset: bool = False,
            max_steps: int = 400,
            task_id: int = 0,
            num_agents: int = 2,
            agent_restrictions: dict = None,
            soup_cook_time: int = DEFAULT_SOUP_COOK_TIME,
    ):
        super().__init__(num_agents=num_agents)

        # Convert string layout to dictionary if needed
        if isinstance(layout, str):
            layout = layout_grid_to_dict(layout)
        elif layout is None:
            layout = FrozenDict(layouts["cramped_room"])

        # self.obs_shape = (agent_view_size, agent_view_size, 3)
        # Observations given by 26 channels, most of which are boolean masks
        self.height = layout["height"]
        self.width = layout["width"]
        self.obs_shape = (self.width, self.height, 26)

        self.agent_view_size = 5  # Hard coded. Only affects map padding -- not observations.
        self.layout = layout
        self.layout_name = layout_name
        self.agents = ["agent_0", "agent_1"]

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

        # Store agent restrictions
        self.agent_restrictions = agent_restrictions or {}

        # Configure soup cooking time
        self.soup_cook_time = soup_cook_time
        # Pot status values based on configurable cook time
        # POT_FULL_STATUS = soup_cook_time (e.g., 20 for default, 5 for tests)
        # POT_EMPTY_STATUS = POT_FULL_STATUS + MAX_ONIONS_IN_POT (e.g., 23 for default, 8 for tests)
        self.pot_full_status = soup_cook_time
        self.pot_empty_status = soup_cook_time + MAX_ONIONS_IN_POT

    def step_env(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""

        acts = self.action_set.take(indices=jnp.array([actions["agent_0"], actions["agent_1"]]))

        state, individual_rewards, shaped_rewards, soups_delivered = self.step_agents(key, state, acts)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)
        # Now each agent gets their individual delivery reward instead of shared reward
        rewards = {"agent_0": individual_rewards[0], "agent_1": individual_rewards[1]}
        shaped_rewards = {"agent_0": shaped_rewards[0], "agent_1": shaped_rewards[1]}
        soups_delivered = {"agent_0": soups_delivered[0], "agent_1": soups_delivered[1]}
        dones = {"agent_0": done, "agent_1": done, "__all__": done}
        info = {
            'shaped_reward': shaped_rewards,
            'soups': soups_delivered,
        }

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
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
        all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)

        wall_idx = layout.get("wall_idx")

        occupied_mask = jnp.zeros_like(all_pos)
        occupied_mask = occupied_mask.at[wall_idx].set(1)
        wall_map = occupied_mask.reshape(h, w).astype(jnp.bool_)

        # Reset agent position + dir
        key, subkey = jax.random.split(key)
        agent_idx = jax.random.choice(subkey, all_pos, shape=(num_agents,),
                                      p=(~occupied_mask.astype(jnp.bool_)).astype(jnp.float32), replace=False)

        # Replace with fixed layout if applicable. Also randomize if agent position not provided
        agent_idx = random_reset * agent_idx + (1 - random_reset) * layout.get("agent_idx", agent_idx)
        agent_pos = jnp.array([agent_idx % w, agent_idx // w], dtype=jnp.uint32).transpose()  # dim = n_agents x 2
        occupied_mask = occupied_mask.at[agent_idx].set(1)

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
            num_agents=self.num_agents,
            agent_view_size=self.agent_view_size
        )

        # agent inventory (empty by default, can be randomized)
        key, subkey = jax.random.split(key)
        possible_items = jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['onion'],
                                    OBJECT_TO_INDEX['plate'], OBJECT_TO_INDEX['dish']])
        random_agent_inv = jax.random.choice(subkey, possible_items, shape=(num_agents,), replace=True)
        agent_inv = random_reset * random_agent_inv + \
                    (1 - random_reset) * jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['empty']])

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

        width = self.obs_shape[0]
        height = self.obs_shape[1]
        n_channels = self.obs_shape[2]
        padding = (state.maze_map.shape[0] - height) // 2

        maze_map = state.maze_map[padding:-padding, padding:-padding, 0]
        soup_loc = jnp.array(maze_map == OBJECT_TO_INDEX["dish"], dtype=jnp.uint8)

        pot_loc_layer = jnp.array(maze_map == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8)
        pot_status = state.maze_map[padding:-padding, padding:-padding, 2] * pot_loc_layer
        onions_in_pot_layer = jnp.minimum(self.pot_empty_status - pot_status, MAX_ONIONS_IN_POT) * (
                pot_status >= self.pot_full_status)  # 0/1/2/3, as long as not cooking or not done
        onions_in_soup_layer = jnp.minimum(self.pot_empty_status - pot_status, MAX_ONIONS_IN_POT) * (
                pot_status < self.pot_full_status) \
                               * pot_loc_layer + MAX_ONIONS_IN_POT * soup_loc  # 0/3, as long as cooking or done
        pot_cooking_time_layer = pot_status * (pot_status < self.pot_full_status)  # Timer: 19 to 0
        soup_ready_layer = pot_loc_layer * (pot_status == POT_READY_STATUS) + soup_loc  # Ready soups, plated or not
        urgency_layer = jnp.ones(maze_map.shape, dtype=jnp.uint8) * ((self.max_steps - state.time) < URGENCY_CUTOFF)

        agent_pos_layers = jnp.zeros((2, height, width), dtype=jnp.uint8)
        agent_pos_layers = agent_pos_layers.at[0, state.agent_pos[0, 1], state.agent_pos[0, 0]].set(1)
        agent_pos_layers = agent_pos_layers.at[1, state.agent_pos[1, 1], state.agent_pos[1, 0]].set(1)

        # Add agent inv: This works because loose items and agent cannot overlap
        agent_inv_items = jnp.expand_dims(state.agent_inv, (1, 2)) * agent_pos_layers
        maze_map = jnp.where(jnp.sum(agent_pos_layers, 0), agent_inv_items.sum(0), maze_map)
        soup_ready_layer = soup_ready_layer \
                           + (jnp.sum(agent_inv_items, 0) == OBJECT_TO_INDEX["dish"]) * jnp.sum(agent_pos_layers, 0)
        onions_in_soup_layer = onions_in_soup_layer \
                               + (jnp.sum(agent_inv_items, 0) == OBJECT_TO_INDEX["dish"]) * 3 * jnp.sum(
            agent_pos_layers, 0)

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

        # Agent related layers
        agent_direction_layers = jnp.zeros((8, height, width), dtype=jnp.uint8)
        dir_layer_idx = state.agent_dir_idx + jnp.array([0, 4])
        agent_direction_layers = agent_direction_layers.at[dir_layer_idx, :, :].set(agent_pos_layers)

        # Both agent see their layers first, then the other layer
        alice_obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
        alice_obs = alice_obs.at[0:2].set(agent_pos_layers)

        alice_obs = alice_obs.at[2:10].set(agent_direction_layers)
        alice_obs = alice_obs.at[10:].set(jnp.stack(env_layers))

        bob_obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
        bob_obs = bob_obs.at[0].set(agent_pos_layers[1]).at[1].set(agent_pos_layers[0])
        bob_obs = bob_obs.at[2:6].set(agent_direction_layers[4:]).at[6:10].set(agent_direction_layers[0:4])
        bob_obs = bob_obs.at[10:].set(jnp.stack(env_layers))

        alice_obs = jnp.transpose(alice_obs, (1, 2, 0))
        bob_obs = jnp.transpose(bob_obs, (1, 2, 0))

        return {"agent_0": alice_obs, "agent_1": bob_obs}

    def step_agents(
            self, key: chex.PRNGKey, state: State, action: chex.Array,
    ) -> Tuple[State, Tuple[float], Tuple[float], Tuple[float]]:
        """
        Process agent actions and compute rewards.

        REWARD SYSTEM EXPLANATION:
        This method handles the complex reward calculation for the Overcooked environment.
        There are two types of rewards:

        1. DELIVERY REWARDS (main reward):
           - Each agent gets DELIVERY_REWARD (20 points) when they successfully deliver a soup
           - Individual delivery rewards are calculated in process_interact()
           - alice_reward and bob_reward contain individual delivery rewards
           - These are summed into a single 'reward' value (line 530)
           - Both agents receive this same total reward in step_env() (lines 131-132)
           - Individual soup counts are preserved in alice_soups/bob_soups for attribution

        2. SHAPED REWARDS (auxiliary rewards):
           - Rewards for intermediate actions: placing onions (3), picking up plates (3), picking up soup (5)
           - These are calculated individually per agent in process_interact()
           - Returned as separate tuple (alice_shaped_reward, bob_shaped_reward)
           - Used by IPPO_CL.py to implement different reward distribution strategies

        The reward distribution issue: By default, both agents get the same total delivery reward
        regardless of who delivered. The individual_rewards setting in IPPO_CL.py fixes this
        by using the soup count information to properly attribute delivery rewards.
        """
        # Update agent position (forward action)
        is_move_action = jnp.logical_and(action != Actions.stay, action != Actions.interact)
        is_move_action_transposed = jnp.expand_dims(is_move_action, 0).transpose()  # Necessary to broadcast correctly

        fwd_pos = jnp.minimum(
            jnp.maximum(state.agent_pos + is_move_action_transposed * DIR_TO_VEC[jnp.minimum(action, 3)] \
                        + ~is_move_action_transposed * state.agent_dir, 0),
            jnp.array((self.width - 1, self.height - 1), dtype=jnp.uint32)
        )

        # Can't go past wall or goal
        def _wall_or_goal(fwd_position, wall_map, goal_pos):
            fwd_wall = wall_map.at[fwd_position[1], fwd_position[0]].get()
            goal_collision = lambda pos, goal: jnp.logical_and(pos[0] == goal[0], pos[1] == goal[1])
            fwd_goal = jax.vmap(goal_collision, in_axes=(None, 0))(fwd_position, goal_pos)
            # fwd_goal = jnp.logical_and(fwd_position[0] == goal_pos[0], fwd_position[1] == goal_pos[1])
            fwd_goal = jnp.any(fwd_goal)
            # Ensure we return scalars
            return jnp.asarray(fwd_wall), jnp.asarray(fwd_goal)

        fwd_pos_has_wall, fwd_pos_has_goal = jax.vmap(_wall_or_goal, in_axes=(0, None, None))(fwd_pos, state.wall_map,
                                                                                              state.goal_pos)

        fwd_pos_blocked = jnp.logical_or(fwd_pos_has_wall, fwd_pos_has_goal)
        # Ensure we have the right shape - flatten and take first num_agents elements
        fwd_pos_blocked = fwd_pos_blocked.flatten()[:self.num_agents]
        fwd_pos_blocked = fwd_pos_blocked.reshape((self.num_agents, 1))

        bounced = jnp.logical_or(fwd_pos_blocked, ~is_move_action_transposed)

        # Agents can't overlap
        # Hardcoded for 2 agents (call them Alice and Bob)
        agent_pos_prev = jnp.array(state.agent_pos)
        fwd_pos = (bounced * state.agent_pos + (~bounced) * fwd_pos).astype(jnp.uint32)
        collision = jnp.all(fwd_pos[0] == fwd_pos[1])

        # No collision = No movement. This matches original Overcooked env.
        alice_pos = jnp.where(
            collision,
            state.agent_pos[0],  # collision and Bob bounced
            fwd_pos[0],
        )
        bob_pos = jnp.where(
            collision,
            state.agent_pos[1],  # collision and Alice bounced
            fwd_pos[1],
        )

        # Prevent swapping places (i.e. passing through each other)
        swap_places = jnp.logical_and(
            jnp.all(fwd_pos[0] == state.agent_pos[1]),
            jnp.all(fwd_pos[1] == state.agent_pos[0]),
        )
        alice_pos = jnp.where(
            ~collision * swap_places,
            state.agent_pos[0],
            alice_pos
        )
        bob_pos = jnp.where(
            ~collision * swap_places,
            state.agent_pos[1],
            bob_pos
        )

        fwd_pos = fwd_pos.at[0].set(alice_pos)
        fwd_pos = fwd_pos.at[1].set(bob_pos)
        agent_pos = fwd_pos.astype(jnp.uint32)

        # Update agent direction
        agent_dir_idx = ~is_move_action * state.agent_dir_idx + is_move_action * action
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        # Handle interacts. Agent 1 first, agent 2 second, no collision handling.
        # This matches the original Overcooked
        fwd_pos = state.agent_pos + state.agent_dir
        maze_map = state.maze_map
        is_interact_action = (action == Actions.interact)

        # Compute the effect of interact first, then apply it if needed
        candidate_maze_map, alice_inv, alice_reward, alice_shaped_reward = self.process_interact(maze_map,
                                                                                                 state.wall_map,
                                                                                                 fwd_pos,
                                                                                                 state.agent_inv, 0)
        alice_interact = is_interact_action[0]
        bob_interact = is_interact_action[1]

        maze_map = jax.lax.select(alice_interact,
                                  candidate_maze_map,
                                  maze_map)
        alice_inv = jax.lax.select(alice_interact,
                                   alice_inv,
                                   state.agent_inv[0])
        alice_reward = jax.lax.select(alice_interact, alice_reward, 0.)
        alice_shaped_reward = jax.lax.select(alice_interact, alice_shaped_reward, 0.)

        candidate_maze_map, bob_inv, bob_reward, bob_shaped_reward = self.process_interact(maze_map, state.wall_map,
                                                                                           fwd_pos, state.agent_inv, 1)
        maze_map = jax.lax.select(bob_interact,
                                  candidate_maze_map,
                                  maze_map)
        bob_inv = jax.lax.select(bob_interact,
                                 bob_inv,
                                 state.agent_inv[1])
        bob_reward = jax.lax.select(bob_interact, bob_reward, 0.)
        bob_shaped_reward = jax.lax.select(bob_interact, bob_shaped_reward, 0.)

        alice_soups = alice_reward / DELIVERY_REWARD
        bob_soups = bob_reward / DELIVERY_REWARD

        agent_inv = jnp.array([alice_inv, bob_inv])

        # Update agent component in maze_map
        def _get_agent_updates(agent_dir_idx, agent_pos, agent_pos_prev, agent_idx):
            agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'] + agent_idx * 2, agent_dir_idx],
                              dtype=jnp.uint8)
            agent_x_prev, agent_y_prev = agent_pos_prev
            agent_x, agent_y = agent_pos
            return agent_x, agent_y, agent_x_prev, agent_y_prev, agent

        vec_update = jax.vmap(_get_agent_updates, in_axes=(0, 0, 0, 0))
        agent_x, agent_y, agent_x_prev, agent_y_prev, agent_vec = vec_update(agent_dir_idx, agent_pos, agent_pos_prev,
                                                                             jnp.arange(self.num_agents))
        empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)

        # Compute padding, added automatically by map maker function
        height = self.obs_shape[1]
        padding = (state.maze_map.shape[0] - height) // 2

        maze_map = maze_map.at[padding + agent_y_prev, padding + agent_x_prev, :].set(empty)
        maze_map = maze_map.at[padding + agent_y, padding + agent_x, :].set(agent_vec)

        # Update pot cooking status
        def _cook_pots(pot):
            pot_status = pot[-1]
            is_cooking = jnp.array(pot_status <= self.pot_full_status)
            not_done = jnp.array(pot_status > POT_READY_STATUS)
            pot_status = is_cooking * not_done * (pot_status - 1) + (
                ~is_cooking) * pot_status  # defaults to zero if done
            return pot.at[-1].set(pot_status)

        pot_x = state.pot_pos[:, 0]
        pot_y = state.pot_pos[:, 1]
        pots = maze_map.at[padding + pot_y, padding + pot_x].get()
        pots = jax.vmap(_cook_pots, in_axes=0)(pots)
        maze_map = maze_map.at[padding + pot_y, padding + pot_x, :].set(pots)

        # Return individual delivery rewards separately
        individual_rewards = (alice_reward, bob_reward)

        return (
            state.replace(
                agent_pos=agent_pos,
                agent_dir_idx=agent_dir_idx,
                agent_dir=agent_dir,
                agent_inv=agent_inv,
                maze_map=maze_map,
                terminal=False),
            individual_rewards,
            (alice_shaped_reward, bob_shaped_reward),
            (alice_soups, bob_soups)
        )

    def process_interact(
            self,
            maze_map: chex.Array,
            wall_map: chex.Array,
            fwd_pos_all: chex.Array,
            inventory_all: chex.Array,
            player_idx: int):
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
        case_1 = (pot_status > self.pot_full_status) * holding_onion * object_is_pot
        case_2 = (pot_status == POT_READY_STATUS) * holding_plate * object_is_pot
        case_3 = (pot_status > POT_READY_STATUS) * (pot_status <= self.pot_full_status) * object_is_pot
        else_case = ~case_1 * ~case_2 * ~case_3

        # give reward for placing onion in pot, and for picking up soup
        shaped_reward += case_1 * BASE_REW_SHAPING_PARAMS["PLACEMENT_IN_POT_REW"]
        shaped_reward += case_2 * BASE_REW_SHAPING_PARAMS["SOUP_PICKUP_REWARD"]

        # Update pot status and object in inventory
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

        # Interactions with onion/plate piles and objects on counter
        # Pickup if: table, not empty, room in inv & object is not something unpickable (e.g. pot or goal)
        base_pickup_condition = is_table * ~table_is_empty * inv_is_empty * jnp.logical_or(object_is_pile,
                                                                                           object_is_pickable)

        # Apply agent restrictions if they exist
        agent_key = f"agent_{player_idx}"
        can_pick_onions = jnp.array(True)
        can_pick_plates = jnp.array(True)

        if self.agent_restrictions:
            can_pick_onions = jnp.array(not self.agent_restrictions.get(f"{agent_key}_cannot_pick_onions", False))
            can_pick_plates = jnp.array(not self.agent_restrictions.get(f"{agent_key}_cannot_pick_plates", False))

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
        pot_loc_layer = jnp.array(maze_map[padding:-padding, padding:-padding, 0] == OBJECT_TO_INDEX["pot"],
                                  dtype=jnp.uint8)
        padded_map = maze_map[padding:-padding, padding:-padding, 2]
        num_notempty_pots = jnp.sum((padded_map != self.pot_empty_status) * pot_loc_layer)
        is_dish_pickup_useful = num_plates_in_inv < num_notempty_pots

        plate_loc_layer = jnp.array(maze_map == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8)
        no_plates_on_counters = jnp.sum(plate_loc_layer) == 0

        shaped_reward += no_plates_on_counters * has_picked_up_plate * is_dish_pickup_useful * BASE_REW_SHAPING_PARAMS[
            "PLATE_PICKUP_REWARD"]

        inventory = new_object_in_inv

        # Apply changes to maze
        new_maze_object_on_table = \
            object_is_pot * OBJECT_INDEX_TO_VEC[new_object_on_table].at[-1].set(new_pot_status) \
            + ~object_is_pot * ~object_is_agent * OBJECT_INDEX_TO_VEC[new_object_on_table] \
            + object_is_agent * maze_object_on_table

        maze_map = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0], :].set(new_maze_object_on_table)

        # DELIVERY REWARD CALCULATION:
        # Each agent gets DELIVERY_REWARD (20 points) individually when they deliver a soup
        # successful_delivery is 1 if this agent delivered a soup, 0 otherwise
        # This individual reward gets summed with other agents' rewards in step_agents()
        # The individual attribution is preserved in the soup count information
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
