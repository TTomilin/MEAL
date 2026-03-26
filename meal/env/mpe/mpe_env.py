"""
MPE SimpleSpread with procedurally-generated obstacle layouts for MEAL.

## Environment design

``MPESpreadEnv`` extends the MPE physics engine with static circular obstacles.
Each task has a unique obstacle field (positions + radii) sampled at sequence-creation
time and frozen for all episodes of that task — analogous to Overcooked grid layouts
and JaxNav polygon maps.

### Entity ordering in ``state.p_pos``

    [0 .. A)          : agents   (moveable=True,  collide=True)
    [A .. A+L)        : landmarks (moveable=False, collide=False)
    [A+L .. A+L+K)    : obstacles (moveable=False, collide=True)   ← new

### Physics

Agents collide with obstacles via JAX-compatible soft contact forces (logaddexp
penetration model already in ``SimpleMPE``).  Obstacles never move because their
``moveable`` flag is False, which zeros out the velocity update in
``_integrate_state``.

### Observation per agent (dim = 4 + L*2 + (A-1)*4 + K*2)

    vel_self (2)
    pos_self (2)
    landmark_rel (L * 2)     — relative to this agent
    other_agent_rel ((A-1)*2) — relative positions of other agents
    other_agent_comm ((A-1)*2) — communication (zeros; agents are silent)
    obstacle_rel (K * 2)     — relative positions of ALL obstacles

### Task diversity

``make_mpe_sequence`` procedurally places K obstacles at random continuous
positions with random radii.  With K≥3 and continuous placement the task space
is effectively infinite.  All structural parameters (A, L, K) are fixed per
sequence so that ``jax.lax.switch`` branches share identical state/obs shapes.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from meal.env.mpe.default_params import (
    AGENT_COLOUR,
    CONTACT_FORCE,
    CONTACT_MARGIN,
    DAMPING,
    DISCRETE_ACT,
    DT,
    OBS_COLOUR,
)
from meal.env.mpe.simple import SimpleMPE, State
from meal.env.mpe.spaces import Box, Discrete

# Gray colour for obstacles (distinct from green agents and dark landmarks)
OBSTACLE_COLOUR = (160, 160, 160)


# ---------------------------------------------------------------------------
# Procedural obstacle layout generation (NumPy, done at sequence build time)
# ---------------------------------------------------------------------------

def _sample_obstacle_layout(
    seed: int,
    num_obstacles: int,
    world_size: float = 0.85,
    min_radius: float = 0.07,
    max_radius: float = 0.16,
    min_sep: float = 0.06,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample non-overlapping obstacle positions and radii.

    Returns
    -------
    positions : (K, 2) float array  in [-world_size, world_size]
    radii     : (K,)   float array  in [min_radius, max_radius]
    """
    rng = np.random.default_rng(seed)
    positions: List[np.ndarray] = []
    radii: List[float] = []

    for _ in range(num_obstacles):
        placed = False
        for _attempt in range(200):
            r = rng.uniform(min_radius, max_radius)
            # Keep obstacle circle within world bounds
            pos = rng.uniform(-world_size + r, world_size - r, size=2)

            # Check separation from already-placed obstacles
            ok = True
            for other_pos, other_r in zip(positions, radii):
                if np.linalg.norm(pos - other_pos) < r + other_r + min_sep:
                    ok = False
                    break
            if ok:
                positions.append(pos)
                radii.append(float(r))
                placed = True
                break

        if not placed:
            # Fallback: tiny obstacle at a random edge-safe spot (accepts overlap)
            r = min_radius
            pos = rng.uniform(-world_size + r, world_size - r, size=2)
            positions.append(pos)
            radii.append(r)

    return np.array(positions, dtype=np.float32), np.array(radii, dtype=np.float32)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MPESpreadEnv(SimpleMPE):
    """MPE cooperative coverage task with static obstacle layouts.

    Parameters
    ----------
    num_agents : int
        Number of cooperative agents (fixed per sequence).
    num_landmarks : int
        Number of coverage targets (fixed per sequence).
    num_obstacles : int
        Number of static circular obstacles (fixed per sequence).
    obstacle_positions : (K, 2) array
        Fixed obstacle centres for this task instance.
    obstacle_radii : (K,) array
        Obstacle radii for this task instance.
    local_ratio : float
        Weight of individual vs global coverage reward.
        0 = fully global, 1 = fully local.
    max_steps : int
        Episode length.
    task_id : int, optional
        Index within the CL sequence (for logging).
    """

    def __init__(
        self,
        num_agents: int = 3,
        num_landmarks: int = 3,
        num_obstacles: int = 4,
        obstacle_positions: Optional[np.ndarray] = None,
        obstacle_radii: Optional[np.ndarray] = None,
        local_ratio: float = 0.5,
        max_steps: int = 100,
        task_id: Optional[int] = None,
    ):
        self._num_goals = num_landmarks      # actual landmarks (for reward/obs)
        self._num_obstacles = num_obstacles  # static obstacles

        # ── agent names (only agents; obstacles/landmarks are not agents) ──
        agents = [f"agent_{i}" for i in range(num_agents)]

        # ── "landmark" list seen by the base class includes real landmarks + obstacles ──
        landmark_names = [f"landmark_{i}" for i in range(num_landmarks)]
        obstacle_names = [f"obstacle_{i}" for i in range(num_obstacles)]
        all_non_agents = landmark_names + obstacle_names  # base class calls these "landmarks"

        num_total_non_agents = num_landmarks + num_obstacles

        # ── physics arrays (indexed: agents first, then non-agents) ──
        if obstacle_radii is None:
            obstacle_radii = np.full(num_obstacles, 0.10, dtype=np.float32)
        obstacle_radii = np.asarray(obstacle_radii, dtype=np.float32)

        rad = jnp.concatenate([
            jnp.full((num_agents,), 0.15),      # agent radius
            jnp.full((num_landmarks,), 0.05),   # landmark radius (small)
            jnp.asarray(obstacle_radii),         # obstacle radii (task-specific)
        ])

        # agents and obstacles collide; landmarks do not
        collide = jnp.array(
            [True] * num_agents + [False] * num_landmarks + [True] * num_obstacles,
            dtype=jnp.bool_,
        )

        # only agents are moveable
        moveable = jnp.array(
            [True] * num_agents + [False] * num_total_non_agents,
            dtype=jnp.bool_,
        )

        # visual colours
        colour = (
            [AGENT_COLOUR] * num_agents
            + [OBS_COLOUR] * num_landmarks
            + [OBSTACLE_COLOUR] * num_obstacles
        )

        # fixed observation dimension
        # vel(2) + pos(2) + landmark_rel(L*2) + other_agents_rel+comm((A-1)*4) + obstacle_rel(K*2)
        obs_dim = 4 + num_landmarks * 2 + (num_agents - 1) * 4 + num_obstacles * 2
        observation_spaces = {
            a: Box(-jnp.inf, jnp.inf, (obs_dim,)) for a in agents
        }

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=num_total_non_agents,   # base class lumps landmarks+obstacles
            landmarks=all_non_agents,
            action_type=DISCRETE_ACT,
            observation_spaces=observation_spaces,
            dim_c=2,
            colour=colour,
            rad=rad,
            collide=collide,
            moveable=moveable,
        )

        # store reward param
        self.local_ratio = local_ratio

        # store fixed obstacle layout (frozen per task)
        if obstacle_positions is None:
            obstacle_positions = np.zeros((num_obstacles, 2), dtype=np.float32)
        self._obstacle_positions = jnp.asarray(obstacle_positions, dtype=jnp.float32)

        # task metadata
        self._task_id = task_id
        self._map_id = (
            f"mpe_a{num_agents}_l{num_landmarks}_k{num_obstacles}"
            f"_r{local_ratio:.1f}_t{task_id if task_id is not None else 0}"
        )

    # ------------------------------------------------------------------
    # MEAL-compatible accessors (no required agent argument)
    # ------------------------------------------------------------------

    def action_space(self, agent=None):
        a = agent if agent is not None else self.agents[0]
        return self.action_spaces[a]

    def observation_space(self, agent=None):
        a = agent if agent is not None else self.agents[0]
        return self.observation_spaces[a]

    @property
    def map_id(self) -> str:
        return self._map_id

    # ------------------------------------------------------------------
    # Reset: randomise agents and landmarks; fix obstacles
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, State]:
        key_a, key_l = jax.random.split(key)

        agent_pos = jax.random.uniform(
            key_a, (self.num_agents, 2), minval=-1.0, maxval=1.0
        )
        landmark_pos = jax.random.uniform(
            key_l, (self._num_goals, 2), minval=-1.0, maxval=1.0
        )

        # Obstacles are at their fixed task positions
        p_pos = jnp.concatenate([agent_pos, landmark_pos, self._obstacle_positions], axis=0)

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, 2)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents,), False),
            step=0,
        )
        return self.get_obs(state), state

    # ------------------------------------------------------------------
    # Observations: landmarks + other agents + obstacles
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        A = self.num_agents
        L = self._num_goals
        K = self._num_obstacles

        @partial(jax.vmap, in_axes=(0,))
        def _obs(aidx: int) -> chex.Array:
            own_vel = state.p_vel[aidx]   # (2,)
            own_pos = state.p_pos[aidx]   # (2,)

            # Landmark positions relative to this agent
            landmark_abs = state.p_pos[A : A + L]          # (L, 2)
            landmark_rel = landmark_abs - own_pos           # (L, 2)

            # Other-agent positions and comm (roll trick to exclude self)
            all_agent_pos = state.p_pos[:A]                # (A, 2)
            other_pos = all_agent_pos - own_pos             # (A, 2) relative
            other_pos = jnp.roll(other_pos, shift=A - aidx - 1, axis=0)[: A - 1]
            other_pos = jnp.roll(other_pos, shift=aidx, axis=0)

            comm = jnp.roll(state.c[:A], shift=A - aidx - 1, axis=0)[: A - 1]
            comm = jnp.roll(comm, shift=aidx, axis=0)

            # Obstacle positions relative to this agent
            obstacle_abs = state.p_pos[A + L :]            # (K, 2)
            obstacle_rel = obstacle_abs - own_pos           # (K, 2)

            return jnp.concatenate([
                own_vel.flatten(),        # 2
                own_pos.flatten(),        # 2
                landmark_rel.flatten(),   # L*2
                other_pos.flatten(),      # (A-1)*2
                comm.flatten(),           # (A-1)*dim_c = (A-1)*2
                obstacle_rel.flatten(),   # K*2
            ])

        obs_arr = _obs(self.agent_range)                    # (A, obs_dim)
        return {a: obs_arr[i] for i, a in enumerate(self.agents)}

    # ------------------------------------------------------------------
    # Rewards: coverage of actual landmarks only (not obstacles)
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=[0])
    def rewards(self, state: State) -> Dict[str, float]:
        A = self.num_agents
        L = self._num_goals

        # Agent–agent collision penalty (local component)
        @partial(jax.vmap, in_axes=(0, None))
        def _collisions(agent_idx: int, other_idx: int):
            return jax.vmap(self.is_collision, in_axes=(None, 0, None))(
                agent_idx, other_idx, state
            )

        c = _collisions(self.agent_range, self.agent_range)

        def _agent_local_rew(aidx: int) -> chex.Array:
            return -1.0 * jnp.sum(c[aidx])

        # Global coverage: for each landmark, reward = -min distance from any agent
        def _land(land_pos: chex.Array) -> chex.Array:
            dists = jnp.linalg.norm(state.p_pos[:A] - land_pos, axis=1)
            return -jnp.min(dists)

        actual_landmarks = state.p_pos[A : A + L]          # (L, 2)
        global_rew = jnp.sum(jax.vmap(_land)(actual_landmarks))

        return {
            a: _agent_local_rew(i) * self.local_ratio + global_rew * (1 - self.local_ratio)
            for i, a in enumerate(self.agents)
        }

    # ------------------------------------------------------------------
    # Step: add coverage metrics to info
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict) -> Tuple:
        # Run base physics
        u, comm_u = self.set_actions(actions)
        # Pad comm to dim_c if needed (base class pattern)
        if comm_u.shape[1] < self.dim_c:
            comm_u = jnp.concatenate(
                [comm_u, jnp.zeros((self.num_agents, self.dim_c - comm_u.shape[1]))], axis=1
            )

        key, key_w = jax.random.split(key)
        p_pos, p_vel = self._world_step(key_w, state, u)

        key_c = jax.random.split(key, self.num_agents)
        new_c = self._apply_comm_action(key_c, comm_u, self.c_noise, self.silent)

        done = jnp.full((self.num_agents,), state.step >= self.max_steps)
        state = state.replace(p_pos=p_pos, p_vel=p_vel, c=new_c, done=done, step=state.step + 1)

        rewards = self.rewards(state)
        obs = self.get_obs(state)
        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones["__all__"] = jnp.all(done)

        # Coverage metrics
        A, L = self.num_agents, self._num_goals
        agent_pos = state.p_pos[:A]
        landmark_pos = state.p_pos[A : A + L]

        def _min_dist(l_pos: chex.Array) -> chex.Array:
            return jnp.min(jnp.linalg.norm(agent_pos - l_pos, axis=-1))

        min_dists = jax.vmap(_min_dist)(landmark_pos)        # (L,)
        coverage_reward = -jnp.sum(min_dists)
        # Threshold = agent_radius + landmark_radius (physically touching)
        num_covered = jnp.sum(min_dists < 0.20).astype(jnp.float32)
        coverage_fraction = num_covered / self._num_goals   # primary success metric [0,1]

        info = {
            "coverage_reward": coverage_reward,
            "num_covered": num_covered,
            "coverage_fraction": coverage_fraction,
        }
        return obs, state, rewards, dones, info


# ---------------------------------------------------------------------------
# Sequence factory
# ---------------------------------------------------------------------------

def make_mpe_sequence(
    sequence_length: int,
    seed: int,
    num_agents: int = 3,
    num_landmarks: int = 3,
    num_obstacles: int = 4,
    max_steps: int = 100,
    local_ratio: float = 0.5,
    obstacle_min_radius: float = 0.07,
    obstacle_max_radius: float = 0.16,
) -> List[MPESpreadEnv]:
    """Create a CL sequence of MPE obstacle-layout tasks.

    Each task has a unique procedurally-generated obstacle field.
    Structural parameters (num_agents, num_landmarks, num_obstacles) are fixed
    so that all tasks share identical state/observation shapes — required for
    ``jax.lax.switch`` across tasks.

    Parameters
    ----------
    sequence_length : int
        Number of tasks.
    seed : int
        Master seed; task i uses seed ``seed + i``.
    num_agents : int
        Agents per task (fixed).
    num_landmarks : int
        Coverage landmarks per task (fixed).
    num_obstacles : int
        Static obstacles per task (fixed count, unique placement per task).
    max_steps : int
        Episode length.
    local_ratio : float
        Reward localisation weight (0=global, 1=local).
    obstacle_min_radius, obstacle_max_radius : float
        Obstacle radius sampling bounds.
    """
    envs: List[MPESpreadEnv] = []
    for t in range(sequence_length):
        positions, radii = _sample_obstacle_layout(
            seed=seed + t,
            num_obstacles=num_obstacles,
            min_radius=obstacle_min_radius,
            max_radius=obstacle_max_radius,
        )
        env = MPESpreadEnv(
            num_agents=num_agents,
            num_landmarks=num_landmarks,
            num_obstacles=num_obstacles,
            obstacle_positions=positions,
            obstacle_radii=radii,
            local_ratio=local_ratio,
            max_steps=max_steps,
            task_id=t,
        )
        envs.append(env)
    return envs
