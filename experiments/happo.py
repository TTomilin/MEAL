"""
HAPPO – Heterogeneous-Agent Proximal Policy Optimisation (homogeneous / shared-policy variant)

Algorithm:
  • Centralized critic (CTDE): V(s^{1..n}) conditioned on concatenated observations.
  • Decentralized actors: shared policy network trained with per-agent gradient steps.
  • Sequential per-agent update with M-factor (HAPPO core):
      For each agent i in order:
          L_i = E[ min( r_i · M^{i-1} · Â,  clip(r_i) · M^{i-1} · Â ) ]
          M^i = M^{i-1} · r_i  (stop-gradient; accumulated IS ratio from all updated agents)
    This provides a monotonic joint policy improvement guarantee.

Based on IPPO (ippo.py) with the following key changes:
  1. Separate Actor + Critic networks (decoupled_mlp.py).
  2. runner_state carries (actor_ts, critic_ts, ...) instead of a single train_state.
  3. Rollout stores global_state (per-env) for the centralized critic.
  4. _update_minbatch does sequential per-agent actor updates (M-factor) + one critic update.
  5. CL (EWC/MAS/L2/FT/AGEM) operates on actor params only.

Reference: Kuba et al. 2021, "Trust Region Policy Optimisation in Multi-Agent Reinforcement
Learning", https://arxiv.org/abs/2109.11251
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence, Optional, List, Literal, NamedTuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from flax.core.frozen_dict import unfreeze
from flax.training.train_state import TrainState
from jax._src.flatten_util import ravel_pytree

from experiments.continual.agem import (
    AGEM, init_agem_memory, sample_task_slot,
    compute_memory_gradient, agem_project, update_agem_memory,
)
from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.l2 import L2
from experiments.continual.mas import MAS
from experiments.evaluation import evaluate_all_envs, make_eval_fn
from experiments.model.decoupled_mlp import Actor, Critic
from experiments.utils import (
    batchify, unbatchify,
    add_eval_metrics, init_cl_state,
    create_visualizer,
)
from meal import make_sequence
from meal.env.utils.max_soup_calculator import calculate_max_soup
from meal.wrappers.logging import LogWrapper

import os
from datetime import datetime
from tensorboardX import SummaryWriter


# ---------------------------------------------------------------------------
# Transition NamedTuple for HAPPO
# ---------------------------------------------------------------------------

class Transition_HAPPO(NamedTuple):
    """
    Per-step transition for HAPPO.

    Fields with (num_actors,) shape cover all agents in all envs, batchified
    in the same order as IPPO (agent-0 envs first, then agent-1 envs, etc.).
    global_state has shape (num_envs, global_dim) – one row per *environment*,
    not per actor – so the scan axis adds a leading num_steps dim.
    """
    done: jnp.ndarray         # (num_actors,)
    action: jnp.ndarray       # (num_actors,)
    value: jnp.ndarray        # (num_actors,)  – tiled from per-env critic output
    reward: jnp.ndarray       # (num_actors,)
    log_prob: jnp.ndarray     # (num_actors,)
    obs: jnp.ndarray          # (num_actors, obs_dim)  – batchified local observations
    global_state: jnp.ndarray # (num_envs, global_dim) – for centralized critic


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING / PPO PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    alg_name: Literal["happo"] = "happo"
    lr: float = 1e-3
    anneal_lr: bool = False
    num_envs: int = 2048
    num_steps: int = 400
    steps_per_task: float = 1e8
    update_epochs: int = 8
    num_minibatches: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0

    # Reward shaping
    reward_shaping: bool = True
    reward_shaping_horizon: float = 2.5e7

    # Reward distribution settings
    sparse_rewards: bool = False
    individual_rewards: bool = False

    # ═══════════════════════════════════════════════════════════════════════════
    # NETWORK ARCHITECTURE PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    activation: str = "relu"
    use_cnn: bool = False
    use_layer_norm: bool = True
    use_agent_id: bool = False  # Optionally condition actor on agent index

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTINUAL LEARNING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    cl_method: Optional[str] = None
    reg_coef: Optional[float] = None
    use_task_id: bool = True
    use_multihead: bool = True
    normalize_importance: bool = False
    regularize_heads: bool = False
    reset_optimizer: bool = True

    # Regularization method specific parameters
    importance_episodes: int = 5
    importance_stride: int = 5
    importance_steps: int = 500
    importance_mode: str = "online"
    importance_decay: float = 0.9

    # AGEM specific parameters
    agem_memory_size: int = 100000
    agem_sample_size: int = 1024
    agem_gradient_scale: float = 1.0

    # ═══════════════════════════════════════════════════════════════════════════
    # ENVIRONMENT PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    env_name: str = "overcooked"
    num_agents: int = 2
    seq_length: int = 10
    repeat_sequence: int = 1
    strategy: str = "generate"
    layouts: Optional[Sequence[str]] = field(default_factory=lambda: [])
    env_kwargs: Optional[Sequence[dict]] = None
    difficulty: Optional[str] = None
    single_task_idx: Optional[int] = None
    random_reset: bool = False
    random_agent_start: bool = True
    complementary_restrictions: bool = False
    separated_agents: bool = False

    # Non-stationarity environment parameters
    sticky_actions: bool = False
    slippery_tiles: bool = False
    random_pot_size: bool = False
    random_cook_time: bool = False
    non_stationary: bool = False

    # ═══════════════════════════════════════════════════════════════════════════
    # EVALUATION PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    evaluation: bool = True
    eval_num_episodes: int = 5
    record_video: bool = False
    video_length: int = 250
    log_interval: int = 5
    renderer_version: str = "v1"

    # ═══════════════════════════════════════════════════════════════════════════
    # LOGGING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    use_wandb: bool = True
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    entity: Optional[str] = ""
    project: str = "MEAL"
    tags: List[str] = field(default_factory=list)

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    seed: int = 30
    num_seeds: int = 1

    # ═══════════════════════════════════════════════════════════════════════════
    # RUNTIME COMPUTED PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    num_actors: int = 0
    num_updates: int = 0
    minibatch_size: int = 0   # per-agent minibatch size


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def create_global_state_for_critic(obs_dict, agent_list, num_envs: int, use_cnn: bool = False):
    """
    Concatenate all agents' observations into a single global state for the centralized critic.

    MLP mode:  (num_envs, sum_of_flat_obs_dims)
    CNN mode:  (num_envs, H, W, num_agents * C)
    """
    if use_cnn:
        agent_obs = [obs_dict[a] for a in agent_list]
        return jnp.concatenate(agent_obs, axis=-1)   # stack along channel dim
    else:
        agent_obs = [obs_dict[a].reshape(num_envs, -1) for a in agent_list]
        return jnp.concatenate(agent_obs, axis=-1)   # (num_envs, total_obs_dim)


class HAPPOActorWrapper:
    """
    Wraps the Actor network to expose a 3-value apply signature
    ``(pi, value_placeholder, dormant_ratio)`` required by
    ``make_eval_fn``, ``compute_memory_gradient``, and CL importance functions.
    """

    def __init__(self, actor_net: Actor):
        self._net = actor_net
        self.action_dim = actor_net.action_dim  # needed by make_eval_fn

    def apply(self, params, obs, *, env_idx=0):
        pi, dormant = self._net.apply(params, obs, env_idx=env_idx)
        return pi, jnp.array(0.0), dormant


# ---------------------------------------------------------------------------
# Rollout helper for video recording
# ---------------------------------------------------------------------------

def rollout_for_video_happo(rng, cfg, actor_ts, env, actor_wrapper, env_idx=0, max_steps=300):
    """Collect a single episode for visualisation using the actor network."""
    rng, env_rng = jax.random.split(rng)
    obs, state = env.reset(env_rng)
    done = False
    step_count = 0
    states = [env.unwrap_env_state(state)]

    while not done and step_count < max_steps:
        obs_dict = {}
        for agent_id, obs_v in obs.items():
            expected_shape = env.observation_space().shape
            if obs_v.ndim == len(expected_shape):
                obs_b = jnp.expand_dims(obs_v, axis=0)
            else:
                obs_b = obs_v
            if not cfg.use_cnn:
                obs_b = jnp.reshape(obs_b, (obs_b.shape[0], -1))
            obs_dict[agent_id] = obs_b

        actions = {}
        act_keys = jax.random.split(rng, env.num_agents)
        for i, agent_id in enumerate(env.agents):
            pi, _, _ = actor_wrapper.apply(actor_ts.params, obs_dict[agent_id], env_idx=env_idx)
            actions[agent_id] = jnp.squeeze(pi.sample(seed=act_keys[i]), axis=0)

        rng, key_step = jax.random.split(rng)
        next_obs, next_state, reward, done_info, info = env.step(key_step, state, actions)
        done = done_info["__all__"]

        obs, state = next_obs, next_state
        step_count += 1
        states.append(env.unwrap_env_state(state))

    return states


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    jax.config.update("jax_platform_name", "gpu")
    print("Device:", jax.devices())

    cfg = tyro.cli(Config)

    if cfg.non_stationary:
        cfg.sticky_actions = True
        cfg.slippery_tiles = True
        cfg.random_pot_size = True
        cfg.random_cook_time = True

    if cfg.sparse_rewards and cfg.individual_rewards:
        raise ValueError(
            "Cannot enable both sparse_rewards and individual_rewards simultaneously."
        )

    if cfg.single_task_idx is not None:
        cfg.cl_method = "ft"
    if cfg.cl_method is None:
        raise ValueError(
            "cl_method is required (e.g. ewc, mas, l2, ft, agem)."
        )

    seq_length = cfg.seq_length
    strategy = cfg.strategy
    seed = cfg.seed
    difficulty = cfg.difficulty

    if cfg.reg_coef is None:
        if cfg.cl_method.lower() == "ewc":
            cfg.reg_coef = 1e11
        elif cfg.cl_method.lower() == "mas":
            cfg.reg_coef = 1e9
        elif cfg.cl_method.lower() == "l2":
            cfg.reg_coef = 1e7

    method_map = dict(
        ewc=EWC(mode=cfg.importance_mode, decay=cfg.importance_decay),
        mas=MAS(mode=cfg.importance_mode, decay=cfg.importance_decay),
        l2=L2(),
        ft=FT(),
        agem=AGEM(memory_size=cfg.agem_memory_size, sample_size=cfg.agem_sample_size),
    )
    cl = method_map[cfg.cl_method.lower()]

    envs = make_sequence(
        sequence_length=seq_length,
        strategy=strategy,
        env_id=cfg.env_name,
        seed=seed,
        num_agents=cfg.num_agents,
        max_steps=cfg.num_steps,
        random_reset=cfg.random_reset,
        layout_names=cfg.layouts,
        difficulty=cfg.difficulty,
        repeat_sequence=cfg.repeat_sequence,
        random_agent_start=cfg.random_agent_start,
        complementary_restrictions=cfg.complementary_restrictions,
        separated_agents=cfg.separated_agents,
        sticky_actions=cfg.sticky_actions,
        slippery_tiles=cfg.slippery_tiles,
        random_pot_size=cfg.random_pot_size,
        random_cook_time=cfg.random_cook_time,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    network_arch = "cnn" if cfg.use_cnn else "mlp"
    run_name = (
        f"happo_{cfg.cl_method}_{difficulty}_{cfg.num_agents}agents_"
        f"{network_arch}_seq{seq_length}_{strategy}_seed_{seed}_{timestamp}"
    )
    exp_dir = os.path.join("runs", run_name)

    if cfg.use_wandb:
        wandb_tags = cfg.tags if cfg.tags is not None else []
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(
            project=cfg.project,
            config=asdict(cfg),
            sync_tensorboard=True,
            mode=cfg.wandb_mode,
            tags=wandb_tags,
            group=cfg.cl_method.upper(),
            name=run_name,
            id=run_name,
        )

    writer = SummaryWriter(exp_dir)
    rows = []
    for key, value in vars(cfg).items():
        value_str = str(value).replace("\n", "<br>").replace("|", "\\|")
        rows.append(f"|{key}|{value_str}|")
    writer.add_text("hyperparameters", f"|param|value|\n|-|-|\n" + "\n".join(rows))

    # Wrap environments
    env_names = []
    max_soup_vals = []
    goal_counts = []
    pot_counts = []
    for i, env in enumerate(envs):
        env = LogWrapper(env, replace_info=False)
        env_names.append(env.layout_name)
        max_soup_vals.append(
            calculate_max_soup(env.layout, env.max_steps, n_agents=env.num_agents)
        )
        goal_counts.append(env.layout["goal_idx"].shape[0])
        pot_counts.append(env.layout["pot_idx"].shape[0])

    max_soup_vals = jnp.asarray(max_soup_vals, dtype=jnp.float32)

    temp_env = envs[0]
    num_agents = temp_env.num_agents
    agents = temp_env.agents

    cfg.num_actors = num_agents * cfg.num_envs
    cfg.num_updates = int(cfg.steps_per_task // cfg.num_steps // cfg.num_envs)
    # Per-agent minibatch size (total / num_agents / num_minibatches)
    cfg.minibatch_size = (cfg.num_envs * cfg.num_steps) // cfg.num_minibatches

    def linear_schedule(count):
        frac = 1.0 - (count // (cfg.num_minibatches * cfg.update_epochs)) / cfg.num_updates
        return cfg.lr * frac

    # ── Network creation ─────────────────────────────────────────────────────
    obs_shape = temp_env.observation_space().shape
    if cfg.use_cnn:
        local_obs_dim = obs_shape
        global_obs_dim = (obs_shape[0], obs_shape[1], obs_shape[2] * num_agents)
    else:
        local_obs_dim_flat = int(np.prod(obs_shape))
        global_obs_dim_flat = local_obs_dim_flat * num_agents

    actor_network = Actor(
        action_dim=temp_env.action_space().n,
        activation=cfg.activation,
        num_tasks=seq_length,
        use_multihead=cfg.use_multihead,
        use_task_id=cfg.use_task_id,
        use_cnn=cfg.use_cnn,
        use_layer_norm=cfg.use_layer_norm,
        use_agent_id=cfg.use_agent_id,
        num_agents=num_agents,
        num_envs=cfg.num_envs,
    )

    critic_network = Critic(
        activation=cfg.activation,
        num_tasks=seq_length,
        use_multihead=cfg.use_multihead,
        use_task_id=cfg.use_task_id,
        use_cnn=cfg.use_cnn,
        use_layer_norm=cfg.use_layer_norm,
    )

    rng = jax.random.PRNGKey(cfg.seed)
    rng, actor_rng, critic_rng = jax.random.split(rng, 3)

    if cfg.use_cnn:
        actor_init_x = jnp.zeros((1, *local_obs_dim))
        critic_init_x = jnp.zeros((1, *global_obs_dim))
    else:
        actor_init_x = jnp.zeros((1, local_obs_dim_flat))
        critic_init_x = jnp.zeros((1, global_obs_dim_flat))

    actor_params = actor_network.init(actor_rng, actor_init_x, env_idx=0)
    critic_params = critic_network.init(critic_rng, critic_init_x, env_idx=0)

    # JIT compile the network apply functions
    actor_network.apply = jax.jit(actor_network.apply)
    critic_network.apply = jax.jit(critic_network.apply)

    # Wrapper providing (pi, value_placeholder, dormant) for CL / eval compatibility
    actor_wrapper = HAPPOActorWrapper(actor_network)

    # ── Optimizers ───────────────────────────────────────────────────────────
    actor_tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=linear_schedule if cfg.anneal_lr else cfg.lr, eps=1e-5),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=linear_schedule if cfg.anneal_lr else cfg.lr, eps=1e-5),
    )

    actor_ts = TrainState.create(
        apply_fn=actor_network.apply,
        params=actor_params,
        tx=actor_tx,
    )
    critic_ts = TrainState.create(
        apply_fn=critic_network.apply,
        params=critic_params,
        tx=critic_tx,
    )

    reset_fns = tuple(env.reset for env in envs)
    step_fns = tuple(env.step for env in envs)

    def reset_switch(key, task_idx):
        return jax.lax.switch(task_idx, reset_fns, key)

    def step_switch(key, state, actions, task_idx):
        return jax.lax.switch(task_idx, step_fns, key, state, actions)

    # Evaluation and importance functions use actor_wrapper (3-value apply)
    evaluate_env = make_eval_fn(
        reset_switch, step_switch, actor_wrapper, agents, seq_length, cfg.num_steps, cfg.use_cnn
    )

    importance_fn = cl.make_importance_fn(
        reset_switch, step_switch, actor_wrapper, agents, cfg.use_cnn,
        cfg.importance_episodes, cfg.importance_steps, cfg.normalize_importance,
        cfg.importance_stride,
    )

    # ── Training function ────────────────────────────────────────────────────

    @jax.jit
    def train_on_environment(rng, actor_ts, critic_ts, cl_state, env_idx):
        """Train on a single environment (task) using HAPPO."""

        # Optionally reset the optimiser state at the start of each task
        if cfg.reset_optimizer:
            actor_ts = actor_ts.replace(
                tx=actor_tx,
                opt_state=actor_tx.init(actor_ts.params),
            )
            critic_ts = critic_ts.replace(
                tx=critic_tx,
                opt_state=critic_tx.init(critic_ts.params),
            )

        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, cfg.num_envs)
        obsv, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(reset_rng)

        reward_shaping_horizon = cfg.steps_per_task / 2
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.0,
            end_value=0.0,
            transition_steps=reward_shaping_horizon,
        )

        # ── Inner training loop ──────────────────────────────────────────────

        def _update_step(runner_state, _):
            """Collect a trajectory and perform one full PPO update."""

            # ── Trajectory collection ────────────────────────────────────────
            def _env_step(runner_state, _):
                actor_ts, critic_ts, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state

                rng, _rng = jax.random.split(rng)

                # Local observations for the actor (all agents, batchified)
                obs_batch = batchify(last_obs, agents, cfg.num_actors, not cfg.use_cnn)
                # (num_actors, obs_dim) – agent-0 envs first, then agent-1 envs

                # Actor: shared policy, processes all agents together
                pi, _, _ = actor_wrapper.apply(actor_ts.params, obs_batch, env_idx=env_idx)
                action = pi.sample(seed=_rng)       # (num_actors,)
                log_prob = pi.log_prob(action)       # (num_actors,)

                # Centralized critic: takes concatenated (global) observations
                global_state = create_global_state_for_critic(
                    last_obs, agents, cfg.num_envs, cfg.use_cnn
                )  # (num_envs, global_dim)
                value_per_env, _ = critic_network.apply(
                    critic_ts.params, global_state, env_idx=env_idx
                )  # (num_envs,)

                # Tile value to match per-actor layout:
                # jnp.tile([v0,v1,...,vN], 2) = [v0,v1,...,vN, v0,v1,...,vN]
                # which matches [a0_e0, a0_e1, ..., a0_eN, a1_e0, ..., a1_eN]
                value = jnp.tile(value_per_env, num_agents)  # (num_actors,)

                env_act = unbatchify(action, agents, cfg.num_envs, num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, cfg.num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    lambda k, s, a: step_switch(k, s, a, jnp.int32(env_idx))
                )(rng_step, env_state, env_act)

                current_timestep = update_step * cfg.num_steps * cfg.num_envs

                if cfg.sparse_rewards:
                    pass
                elif cfg.individual_rewards:
                    reward = jax.tree_util.tree_map(
                        lambda x, y: x + y * rew_shaping_anneal(current_timestep),
                        reward, info["shaped_reward"],
                    )
                else:
                    total_delivery_reward = sum(reward[a] for a in agents)
                    shared_delivery_rewards = {a: total_delivery_reward for a in agents}
                    reward = jax.tree_util.tree_map(
                        lambda x, y: x + y * rew_shaping_anneal(current_timestep),
                        shared_delivery_rewards, info["shaped_reward"],
                    )

                transition = Transition_HAPPO(
                    done=batchify(done, agents, cfg.num_actors, not cfg.use_cnn).squeeze(),
                    action=action,
                    value=value,
                    reward=batchify(reward, agents, cfg.num_actors).squeeze(),
                    log_prob=log_prob,
                    obs=obs_batch,
                    global_state=global_state,   # (num_envs, global_dim)
                )

                steps_for_env = steps_for_env + cfg.num_envs
                runner_state = (actor_ts, critic_ts, env_state, obsv, update_step, steps_for_env, rng, cl_state)
                return runner_state, (transition, info)

            runner_state, (traj_batch, info) = jax.lax.scan(
                _env_step, runner_state, xs=None, length=cfg.num_steps
            )
            actor_ts, critic_ts, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state

            # Bootstrap value from centralized critic
            last_global_state = create_global_state_for_critic(
                last_obs, agents, cfg.num_envs, cfg.use_cnn
            )
            last_val_per_env, _ = critic_network.apply(
                critic_ts.params, last_global_state, env_idx=env_idx
            )
            last_val = jnp.tile(last_val_per_env, num_agents)  # (num_actors,)

            # ── GAE ──────────────────────────────────────────────────────────
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + cfg.gamma * next_value * (1 - done) - value
                    gae = delta + cfg.gamma * cfg.gae_lambda * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # ── Network update ────────────────────────────────────────────────

            def _update_epoch(update_state, _):
                actor_ts, critic_ts, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state

                per_agent_bs = cfg.num_steps * cfg.num_envs  # batch size per agent
                mb_size = per_agent_bs // cfg.num_minibatches  # minibatch size per agent

                # ── Split trajectory into per-agent slices ────────────────────
                # traj_batch.obs: (num_steps, num_actors, obs_dim)
                # Agent i occupies columns [i*num_envs : (i+1)*num_envs] in the num_actors dim.

                def slice_agent(x, i):
                    """(T, num_actors, ...) or (T, num_actors) → (T*num_envs, ...)"""
                    s, e = i * cfg.num_envs, (i + 1) * cfg.num_envs
                    sliced = x[:, s:e]                   # (T, num_envs, ...)
                    return sliced.reshape(per_agent_bs, *sliced.shape[2:])

                # Stacked shapes: (num_agents, per_agent_bs, ...)
                agent_obs   = jnp.stack([slice_agent(traj_batch.obs,      i) for i in range(num_agents)])
                agent_acts  = jnp.stack([slice_agent(traj_batch.action,   i) for i in range(num_agents)])
                agent_logps = jnp.stack([slice_agent(traj_batch.log_prob, i) for i in range(num_agents)])
                agent_vals  = jnp.stack([slice_agent(traj_batch.value,    i) for i in range(num_agents)])
                agent_advs  = jnp.stack([slice_agent(advantages,          i) for i in range(num_agents)])
                agent_tgts  = jnp.stack([slice_agent(targets,             i) for i in range(num_agents)])

                # Critic data (per-env)
                # traj_batch.global_state: (num_steps, num_envs, global_dim)
                critic_gs   = traj_batch.global_state.reshape(per_agent_bs, -1)
                # Use agent-0's value/targets for the critic (same as MAPPO; correct for shared rewards)
                critic_vals = traj_batch.value[:, :cfg.num_envs].reshape(per_agent_bs)
                critic_tgts = targets[:, :cfg.num_envs].reshape(per_agent_bs)

                # ── Shuffle (same permutation for all agents and critic) ───────
                rng, _rng = jax.random.split(rng)
                perm = jax.random.permutation(_rng, per_agent_bs)

                agent_obs   = jnp.take(agent_obs,   perm, axis=1)
                agent_acts  = jnp.take(agent_acts,  perm, axis=1)
                agent_logps = jnp.take(agent_logps, perm, axis=1)
                agent_vals  = jnp.take(agent_vals,  perm, axis=1)
                agent_advs  = jnp.take(agent_advs,  perm, axis=1)
                agent_tgts  = jnp.take(agent_tgts,  perm, axis=1)
                critic_gs   = jnp.take(critic_gs,   perm, axis=0)
                critic_vals = jnp.take(critic_vals,  perm, axis=0)
                critic_tgts = jnp.take(critic_tgts,  perm, axis=0)

                # ── Reshape into minibatches for jax.lax.scan ─────────────────
                # (num_agents, per_agent_bs, ...) → (num_mb, num_agents, mb_size, ...)

                def make_agent_mbs(x):
                    """(num_agents, per_agent_bs, ...) → (num_mb, num_agents, mb_size, ...)"""
                    n_a = x.shape[0]
                    rest = x.shape[2:]
                    x_mb = x.reshape(n_a, cfg.num_minibatches, mb_size, *rest)
                    return jnp.swapaxes(x_mb, 0, 1)

                def make_critic_mbs(x):
                    """(per_agent_bs, ...) → (num_mb, mb_size, ...)"""
                    return x.reshape(cfg.num_minibatches, mb_size, *x.shape[1:])

                def make_critic_mbs_1d(x):
                    """(per_agent_bs,) → (num_mb, mb_size)"""
                    return x.reshape(cfg.num_minibatches, mb_size)

                xs = (
                    make_agent_mbs(agent_obs),         # (num_mb, num_agents, mb_size, obs_dim)
                    make_agent_mbs(agent_acts),         # (num_mb, num_agents, mb_size)
                    make_agent_mbs(agent_logps),
                    make_agent_mbs(agent_vals),
                    make_agent_mbs(agent_advs),
                    make_agent_mbs(agent_tgts),
                    make_critic_mbs(critic_gs),         # (num_mb, mb_size, global_dim)
                    make_critic_mbs_1d(critic_vals),    # (num_mb, mb_size)
                    make_critic_mbs_1d(critic_tgts),
                )

                # ── Minibatch update function ──────────────────────────────────

                def _update_minbatch(carry, xs_mb):
                    """
                    One minibatch update:
                      1. Critic MSE update.
                      2. Sequential per-agent HAPPO actor updates with M-factor.
                    """
                    actor_ts, critic_ts, cl_state, rng = carry
                    rng, agem_rng = jax.random.split(rng)

                    (agent_obs_mb, agent_acts_mb, agent_logps_mb,
                     agent_vals_mb, agent_advs_mb, agent_tgts_mb,
                     critic_gs_mb, critic_vals_mb, critic_tgts_mb) = xs_mb
                    # Per-agent shapes: (num_agents, mb_size, ...)
                    # Critic shapes:    (mb_size, ...)

                    # ── 1. Critic update ─────────────────────────────────────
                    def critic_loss_fn(critic_params):
                        value, _ = critic_network.apply(
                            critic_params, critic_gs_mb, env_idx=env_idx
                        )  # (mb_size,)
                        v_clipped = critic_vals_mb + (value - critic_vals_mb).clip(
                            -cfg.clip_eps, cfg.clip_eps
                        )
                        vl = jnp.maximum(
                            jnp.square(value - critic_tgts_mb),
                            jnp.square(v_clipped - critic_tgts_mb),
                        )
                        return 0.5 * vl.mean(), vl.mean()

                    (critic_loss_val, _), critic_grads = jax.value_and_grad(
                        critic_loss_fn, has_aux=True
                    )(critic_ts.params)
                    critic_ts = critic_ts.apply_gradients(grads=critic_grads)

                    # ── 2. Sequential HAPPO actor updates ────────────────────
                    M = jnp.ones(agent_obs_mb.shape[1])  # (mb_size,)  initialised to 1

                    total_actor_loss = jnp.array(0.0)
                    total_entropy    = jnp.array(0.0)
                    total_cl_penalty = jnp.array(0.0)
                    agem_stats = {}

                    for i in range(num_agents):
                        obs_i   = agent_obs_mb[i]    # (mb_size, obs_dim)
                        act_i   = agent_acts_mb[i]   # (mb_size,)
                        logp_i  = agent_logps_mb[i]  # (mb_size,)
                        val_i   = agent_vals_mb[i]   # (mb_size,)
                        adv_i   = agent_advs_mb[i]   # (mb_size,)
                        tgt_i   = agent_tgts_mb[i]   # (mb_size,)

                        # Closure over M (stop-gradient'd cumulative IS ratio)
                        M_i = jax.lax.stop_gradient(M)

                        def happo_actor_loss(actor_params):
                            pi, _ = actor_network.apply(
                                actor_params, obs_i, env_idx=env_idx
                            )
                            log_prob = pi.log_prob(act_i)
                            ratio = jnp.exp(log_prob - logp_i)

                            # Normalise per-agent advantage
                            adv_norm = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)

                            # HAPPO core: scale by accumulated M from previous agents
                            happo_adv = M_i * adv_norm

                            loss_unclipped = ratio * happo_adv
                            loss_clipped   = (
                                jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                                * happo_adv
                            )
                            actor_loss = -jnp.minimum(loss_unclipped, loss_clipped).mean()
                            entropy    = pi.entropy().mean()
                            cl_penalty = cl.penalty(actor_params, cl_state, cfg.reg_coef)

                            total = actor_loss - cfg.ent_coef * entropy + cl_penalty
                            return total, (actor_loss, entropy, cl_penalty)

                        (total_i, (al_i, e_i, cp_i)), grads_i = jax.value_and_grad(
                            happo_actor_loss, has_aux=True
                        )(actor_ts.params)

                        # ── Apply gradient (with AGEM projection if enabled) ──
                        if cfg.cl_method.lower() == "agem" and cl_state is not None:
                            # Convert PPO gradient → Adam update (before applying)
                            ppo_updates_i, new_opt_state_i = actor_ts.tx.update(
                                grads_i, actor_ts.opt_state, actor_ts.params
                            )

                            past_sizes = cl_state.sizes.at[env_idx].set(0)

                            def apply_agem_proj(agem_rng_i, past_sizes):
                                max_tasks = cl_state.obs.shape[0]
                                spt = max(cfg.agem_sample_size // max_tasks, 1)

                                grads_mem = None
                                ppo_stats_sum = {
                                    "agem/ppo_total_loss": jnp.array(0.0),
                                    "agem/ppo_value_loss": jnp.array(0.0),
                                    "agem/ppo_actor_loss": jnp.array(0.0),
                                    "agem/ppo_entropy":    jnp.array(0.0),
                                }

                                for t in range(cl_state.obs.shape[0]):
                                    agem_rng_i, task_rng = jax.random.split(agem_rng_i)
                                    t_obs, t_acts, t_logp, t_advs, t_tgts, t_vals = (
                                        sample_task_slot(cl_state, t, spt, task_rng)
                                    )
                                    t_grads, t_stats = compute_memory_gradient(
                                        actor_wrapper, actor_ts.params,
                                        cfg.clip_eps, 0.0, cfg.ent_coef,
                                        t_obs, t_acts, t_advs, t_logp, t_tgts, t_vals,
                                        env_idx=t,
                                    )
                                    mask = (past_sizes[t] > 0).astype(jnp.float32)
                                    t_grads = jax.tree_util.tree_map(
                                        lambda g: g * mask, t_grads
                                    )
                                    grads_mem = (
                                        t_grads if grads_mem is None
                                        else jax.tree_util.tree_map(
                                            lambda a, b: a + b, grads_mem, t_grads
                                        )
                                    )
                                    for k in ppo_stats_sum:
                                        ppo_stats_sum[k] = ppo_stats_sum[k] + t_stats[k] * mask

                                n_active = jnp.sum((past_sizes > 0).astype(jnp.float32)) + 1e-8
                                ppo_stats = {k: v / n_active for k, v in ppo_stats_sum.items()}

                                mem_updates, _ = actor_ts.tx.update(
                                    grads_mem, actor_ts.opt_state, actor_ts.params
                                )
                                projected, proj_stats = agem_project(ppo_updates_i, mem_updates)

                                mem_norm = jnp.linalg.norm(ravel_pytree(mem_updates)[0])
                                total_used = jnp.sum(cl_state.sizes)
                                total_cap  = (
                                    cl_state.obs.shape[0] * cl_state.max_size_per_task
                                )
                                proj_stats["agem/mem_grad_norm_raw"]     = mem_norm
                                proj_stats["agem/memory_fullness_pct"]   = (
                                    total_used / total_cap
                                ) * 100.0
                                return projected, {**ppo_stats, **proj_stats}

                            def no_agem_proj():
                                empty = {
                                    "agem/agem_alpha":               jnp.array(0.0),
                                    "agem/agem_dot_g":               jnp.array(0.0),
                                    "agem/agem_final_grad_norm":     jnp.array(0.0),
                                    "agem/agem_is_proj":             jnp.array(False),
                                    "agem/agem_mem_grad_norm":       jnp.array(0.0),
                                    "agem/agem_ppo_grad_norm":       jnp.array(0.0),
                                    "agem/agem_projected_grad_norm": jnp.array(0.0),
                                    "agem/mem_grad_norm_raw":        jnp.array(0.0),
                                    "agem/memory_fullness_pct":      jnp.array(0.0),
                                    "agem/ppo_actor_loss":           jnp.array(0.0),
                                    "agem/ppo_entropy":              jnp.array(0.0),
                                    "agem/ppo_total_loss":           jnp.array(0.0),
                                    "agem/ppo_value_loss":           jnp.array(0.0),
                                }
                                return ppo_updates_i, empty

                            agem_rng, agem_rng_i = jax.random.split(agem_rng)
                            final_updates_i, agem_stats = jax.lax.cond(
                                jnp.sum(past_sizes) > 0,
                                lambda: apply_agem_proj(agem_rng_i, past_sizes),
                                lambda: no_agem_proj(),
                            )

                            new_params = optax.apply_updates(actor_ts.params, final_updates_i)
                            actor_ts = actor_ts.replace(
                                step=actor_ts.step + 1,
                                params=new_params,
                                opt_state=new_opt_state_i,
                            )
                        else:
                            actor_ts = actor_ts.apply_gradients(grads=grads_i)

                        # ── Update M-factor for the next agent ─────────────────
                        pi_new, _ = actor_network.apply(
                            jax.lax.stop_gradient(actor_ts.params),
                            obs_i, env_idx=env_idx,
                        )
                        log_prob_new = pi_new.log_prob(act_i)
                        ratio_new = jax.lax.stop_gradient(
                            jnp.exp(log_prob_new - logp_i)
                        )
                        M = M * ratio_new

                        total_actor_loss = total_actor_loss + al_i
                        total_entropy    = total_entropy    + e_i
                        total_cl_penalty = total_cl_penalty + cp_i

                    avg_actor_loss = total_actor_loss / num_agents
                    avg_entropy    = total_entropy    / num_agents
                    avg_cl_penalty = total_cl_penalty / num_agents

                    loss_information = (
                        (critic_loss_val + avg_actor_loss,
                         (critic_loss_val, avg_actor_loss, avg_entropy, avg_cl_penalty)),
                        grads_i,   # last agent's gradients (for logging only)
                        agem_stats,
                    )
                    return (actor_ts, critic_ts, cl_state, rng), loss_information

                # ── End _update_minbatch ──────────────────────────────────────

                (actor_ts, critic_ts, cl_state, rng), loss_information = jax.lax.scan(
                    _update_minbatch,
                    (actor_ts, critic_ts, cl_state, rng),
                    xs,
                )

                total_loss, grads, agem_stats = loss_information
                loss_dict = {"total_loss": total_loss}
                if cfg.cl_method.lower() == "agem":
                    loss_dict["agem_stats"] = agem_stats

                update_state = (
                    actor_ts, critic_ts, traj_batch, advantages, targets,
                    steps_for_env, rng, cl_state,
                )
                return update_state, loss_dict

            # ── End _update_epoch ─────────────────────────────────────────────

            update_state = (
                actor_ts, critic_ts, traj_batch, advantages, targets,
                steps_for_env, rng, cl_state,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, xs=None, length=cfg.update_epochs
            )
            actor_ts, critic_ts, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state

            current_timestep = update_step * cfg.num_steps * cfg.num_envs
            metrics = jax.tree_util.tree_map(lambda x: x.mean(), info)

            # AGEM memory update (after epoch)
            if cfg.cl_method.lower() == "agem" and cl_state is not None:
                cl_state, rng = update_agem_memory(
                    cfg.agem_sample_size, env_idx, advantages, cl_state, rng, targets, traj_batch
                )

            update_step += 1

            # ── Metrics ──────────────────────────────────────────────────────
            metrics["General/env_index"]   = env_idx
            metrics["General/update_step"] = update_step
            metrics["General/steps_for_env"] = steps_for_env
            metrics["General/env_step"]    = update_step * cfg.num_steps * cfg.num_envs
            metrics["General/learning_rate"] = cfg.lr
            metrics["General/reward_shaping_anneal"] = rew_shaping_anneal(current_timestep)

            loss_dict = loss_info
            total_loss = loss_dict["total_loss"]
            critic_loss_val, avg_actor_loss, avg_entropy, avg_cl_penalty = total_loss[1]
            total_loss_scalar = total_loss[0]

            metrics["Losses/total_loss"]  = total_loss_scalar.mean()
            metrics["Losses/critic_loss"] = critic_loss_val.mean()
            metrics["Losses/actor_loss"]  = avg_actor_loss.mean()
            metrics["Losses/entropy"]     = avg_entropy.mean()
            metrics["Losses/reg_loss"]    = avg_cl_penalty.mean()

            if "agem_stats" in loss_dict:
                for k, v in loss_dict["agem_stats"].items():
                    if v.size > 0:
                        metrics[k] = v.mean()

            # Soup metrics
            T, E, A = cfg.num_steps, cfg.num_envs, num_agents
            soups_tea = jnp.stack([info["soups"][a] for a in agents], axis=-1)
            soups_per_env = soups_tea.sum(axis=(0, 2))
            done_tea = traj_batch.done.reshape(T, E, A)
            done_te  = done_tea[..., 0]
            episodes_per_env = done_te.sum(axis=0)
            mask = episodes_per_env > 0
            true_avg = jnp.where(mask, soups_per_env / jnp.maximum(episodes_per_env, 1), 0.0)
            n_fin = jnp.maximum(mask.sum(), 1)
            metrics["Soup/total"] = true_avg.sum() / n_fin
            max_per_episode = max_soup_vals[env_idx]
            metrics["Soup/scaled"] = jnp.where(
                max_per_episode > 0, (true_avg / max_per_episode).sum() / n_fin, 0.0
            )
            for ai, agent in enumerate(agents):
                soups_te = soups_tea[:, :, ai].sum(axis=0)
                per_agent = jnp.where(mask, soups_te / jnp.maximum(episodes_per_env, 1), 0.0)
                metrics[f"Soup/{agent}"] = per_agent.sum() / n_fin
            metrics.pop("soups", None)

            # Reward metrics
            for agent in agents:
                metrics[f"General/shaped_reward_{agent}"] = metrics["shaped_reward"][agent]
                metrics[f"General/shaped_reward_annealed_{agent}"] = (
                    metrics[f"General/shaped_reward_{agent}"]
                    * rew_shaping_anneal(current_timestep)
                )
            metrics.pop("shaped_reward", None)

            metrics["Advantage_Targets/advantages"] = advantages.mean()
            metrics["Advantage_Targets/targets"]    = targets.mean()

            # Dormant neuron ratio
            obs_batch_last = batchify(last_obs, agents, cfg.num_actors, not cfg.use_cnn)
            _, _, actor_dormant = actor_wrapper.apply(
                actor_ts.params, obs_batch_last, env_idx=env_idx
            )
            metrics["Neural_Activity/actor_dormant_ratio"] = actor_dormant

            # ── Evaluation and logging callback ───────────────────────────────
            def evaluate_and_log(rng, update_step):
                rng, eval_rng = jax.random.split(rng)

                def log_metrics(metrics, update_step):
                    if cfg.evaluation:
                        avg_rewards, avg_soups, _ = evaluate_all_envs(
                            eval_rng, actor_ts.params, seq_length, evaluate_env
                        )
                        metrics = add_eval_metrics(
                            avg_rewards, avg_soups, env_names, max_soup_vals, metrics
                        )

                    def callback(args):
                        metrics, update_step, env_counter = args
                        real_step = (env_counter - 1) * cfg.num_updates + update_step
                        for key, value in metrics.items():
                            writer.add_scalar(key, value, real_step)

                    jax.experimental.io_callback(
                        callback, None, (metrics, update_step, env_idx + 1)
                    )
                    return None

                def do_not_log(metrics, update_step):
                    return None

                jax.lax.cond(
                    (update_step % cfg.log_interval) == 0,
                    log_metrics, do_not_log, metrics, update_step,
                )

            evaluate_and_log(rng=rng, update_step=update_step)

            runner_state = (
                actor_ts, critic_ts, env_state, last_obs, update_step,
                steps_for_env, rng, cl_state,
            )
            return runner_state, metrics

        # ── End _update_step ─────────────────────────────────────────────────

        rng, train_rng = jax.random.split(rng)
        runner_state = (actor_ts, critic_ts, env_state, obsv, 0, 0, train_rng, cl_state)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, xs=None, length=cfg.num_updates
        )
        return runner_state, metrics

    # ── loop_over_envs ───────────────────────────────────────────────────────

    def loop_over_envs(rng, actor_ts, critic_ts, cl_state, envs):
        rng, *env_rngs = jax.random.split(rng, seq_length + 1)

        visualizer = None
        for task_idx, (task_rng, env) in enumerate(zip(env_rngs, envs)):
            print(f"Training on environment: {task_idx} - {env.layout_name}")
            runner_state, metrics = train_on_environment(
                task_rng, actor_ts, critic_ts, cl_state, task_idx
            )
            actor_ts  = runner_state[0]
            critic_ts = runner_state[1]
            cl_state  = runner_state[7]

            # CL importance update (actor params only)
            importance = importance_fn(actor_ts.params, task_idx, task_rng)
            cl_state = cl.update_state(cl_state, actor_ts.params, importance)

            # Video recording
            if cfg.record_video:
                if visualizer is None:
                    visualizer = create_visualizer(num_agents, cfg.env_name, cfg.renderer_version)
                start_time = time.time()
                states = rollout_for_video_happo(
                    task_rng, cfg, actor_ts, env, actor_wrapper,
                    task_idx, cfg.video_length,
                )
                print(f"Rollout for video took {time.time() - start_time:.2f}s.")
                start_time = time.time()
                file_path = f"{exp_dir}/task_{task_idx}_{env.layout_name}.mp4"
                visualizer.animate(states, out_path=file_path, task_idx=task_idx, env=env)
                print(f"Animating video took {time.time() - start_time:.2f}s.")

            # Save model (actor + critic)
            repo_root = Path(__file__).resolve().parent.parent
            path = (
                f"{repo_root}/checkpoints/overcooked/{cfg.cl_method}/"
                f"{run_name}/model_env_{task_idx + 1}"
            )
            save_params(path, actor_ts, critic_ts, env_kwargs=env.layout,
                        layout_name=env.layout_name, config=cfg)

            if cfg.single_task_idx is not None:
                break

    def save_params(path, actor_ts, critic_ts, env_kwargs=None, layout_name=None, config=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path + "_actor", "wb") as f:
            f.write(flax.serialization.to_bytes({"params": actor_ts.params}))
        with open(path + "_critic", "wb") as f:
            f.write(flax.serialization.to_bytes({"params": critic_ts.params}))

        if env_kwargs is not None or layout_name is not None or config is not None:
            def convert_frozen_dict(obj):
                if isinstance(obj, flax.core.frozen_dict.FrozenDict):
                    return {k: convert_frozen_dict(v) for k, v in unfreeze(obj).items()}
                elif isinstance(obj, dict):
                    return {k: convert_frozen_dict(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_frozen_dict(item) for item in obj]
                elif isinstance(obj, jax.Array):
                    arr = np.array(obj)
                    return arr.item() if arr.size == 1 else arr.tolist()
                return obj

            env_kwargs = convert_frozen_dict(env_kwargs)
            config_data = {"env_kwargs": env_kwargs, "layout_name": layout_name}
            if config is not None:
                config_dict = convert_frozen_dict({
                    "use_cnn":         config.use_cnn,
                    "num_tasks":       seq_length,
                    "use_multihead":   config.use_multihead,
                    "use_task_id":     config.use_task_id,
                    "use_layer_norm":  config.use_layer_norm,
                    "activation":      config.activation,
                    "strategy":        config.strategy,
                    "seed":            config.seed,
                })
                config_data.update(config_dict)
            config_data = convert_frozen_dict(config_data)
            with open(path + "_config.json", "w") as f:
                json.dump(config_data, f, indent=2)
        print("Model saved to", path)

    # ── Initialise CL state and run ──────────────────────────────────────────

    rng, train_rng = jax.random.split(rng)

    if cfg.cl_method.lower() == "agem":
        obs_dim_for_mem = envs[0].observation_space().shape
        if not cfg.use_cnn:
            obs_dim_for_mem = (int(np.prod(obs_dim_for_mem)),)
        cl_state = init_agem_memory(
            cfg.agem_memory_size, obs_dim_for_mem, max_tasks=seq_length
        )
    else:
        # CL state tracks actor params only (no critic regularisation)
        cl_state = init_cl_state(actor_ts.params, regularize_critic=False,
                                 regularize_heads=cfg.regularize_heads)

    loop_over_envs(train_rng, actor_ts, critic_ts, cl_state, envs)


if __name__ == "__main__":
    print("Running HAPPO...")
    main()
