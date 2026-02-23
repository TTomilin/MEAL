import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Sequence, Optional, List, Literal

import chex
import flax
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from tensorboardX import SummaryWriter

from experiments.continual.agem import (
    AGEM, init_qmix_agem_memory, sample_qmix_task_slot,
    compute_qmix_memory_gradient, update_qmix_agem_memory, agem_project,
)
from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.l2 import L2
from experiments.continual.mas import MAS
from experiments.evaluation import evaluate_all_envs
from experiments.model.mixing_network import MixingNetwork
from experiments.model.q_mlp import QNetwork
from experiments.utils import (
    batchify,
    unbatchify,
    init_cl_state,
    add_eval_metrics,
    create_visualizer,
)
from experiments.utils_vdn import (
    CustomTrainState,
    eps_greedy_exploration,
    batchify as vdn_batchify,
    unbatchify as vdn_unbatchify,
)
from meal import make_sequence
from meal.env.utils.max_soup_calculator import calculate_max_soup
from meal.wrappers.jaxmarl import CTRolloutManager
from meal.wrappers.logging import LogWrapper


@chex.dataclass(frozen=True)
class QMIXTimestep:
    obs: dict               # per-agent obs (without __all__)
    global_state: jnp.ndarray  # concatenated agent observations (num_steps, num_envs, state_dim)
    actions: dict
    avail_actions: dict
    rewards: dict
    dones: dict


@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING / QMIX PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    alg_name: str = "qmix"
    steps_per_task: float = 1e8
    num_envs: int = 2048
    num_steps: int = 128  # env steps collected per update before learning
    hidden_size: int = 256
    mixing_embed_dim: int = 32  # embedding size for the mixing network hypernetwork
    eps_start: float = 1.0
    eps_finish: float = 0.05
    eps_decay: float = 0.1  # fraction of num_updates over which eps decays
    max_grad_norm: float = 1.0
    update_epochs: int = 8   # passes over collected data per update
    num_minibatches: int = 16  # minibatches per epoch
    lr: float = 1e-3
    anneal_lr: bool = False
    gamma: float = 0.99
    tau: float = 1.0  # target network update rate (1 = hard copy)
    target_update_interval: int = 1

    # Reward distribution settings
    sparse_rewards: bool = False  # only shared delivery reward, no shaped rewards
    individual_rewards: bool = False  # individual delivery + individual shaped rewards

    # ═══════════════════════════════════════════════════════════════════════════
    # NETWORK ARCHITECTURE PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    activation: str = "relu"
    use_cnn: bool = False
    use_layer_norm: bool = True

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTINUAL LEARNING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    cl_method: Optional[str] = None
    reg_coef: Optional[float] = None
    use_task_id: bool = True
    use_multihead: bool = True
    normalize_importance: bool = False
    regularize_heads: bool = False

    # Regularization method specific parameters
    importance_episodes: int = 5
    importance_steps: int = 500
    importance_stride: int = 5
    importance_mode: str = "online"  # "online", "last" or "multi"
    importance_decay: float = 0.9  # EMA decay for online mode

    # AGEM specific parameters
    agem_memory_size: int = 100000
    agem_sample_size: int = 1024

    # ═══════════════════════════════════════════════════════════════════════════
    # ENVIRONMENT PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    env_name: str = "overcooked"
    num_agents: int = 2
    seq_length: int = 10
    repeat_sequence: int = 1
    strategy: str = "generate"
    layouts: Optional[Sequence[str]] = field(default_factory=lambda: [])
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
    max_episode_steps: int = 400  # env episode length; separate from num_steps (collection phase size)
    eval_num_envs: int = 128      # envs for evaluation (smaller than num_envs to save memory during eval)
    evaluation: bool = True
    eval_num_episodes: int = 5
    record_video: bool = False
    video_length: int = 250
    log_interval: int = 5

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


# ──────────────────────────────────────────────────────────────────────────────
# QMIX-specific helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_qmix_eval_fn(reset_switch, step_switch, network, agents, num_envs: int,
                      num_steps: int, use_cnn: bool):
    """
    Returns a JITted evaluate_env(rng, q_params, env_idx) -> (avg_reward, avg_soups).
    Due to the IGM property, greedy QMIX actions = argmax per individual Q-value.
    """
    num_agents = len(agents)

    @jax.jit
    def evaluate_env(rng, q_params, env_idx):
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, num_envs)
        obs, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(reset_rng)

        total_rewards = jnp.zeros((num_envs,), jnp.float32)
        total_soups = jnp.zeros((num_envs,), jnp.float32)

        def one_step(carry, _):
            env_state, obs, rewards, soups, rng = carry

            obs_b = batchify(obs, agents, num_agents * num_envs, not use_cnn)
            obs_b = obs_b.reshape(num_agents, num_envs, -1)

            # Greedy Q-values for all agents (IGM: argmax Q_total = argmax per agent)
            q_vals = jax.vmap(
                lambda p, o: network.apply(p, o, env_idx=env_idx), in_axes=(None, 0)
            )(q_params, obs_b)  # (A, num_envs, action_dim)

            actions_array = jnp.argmax(q_vals, axis=-1)  # (A, num_envs)
            env_act = unbatchify(actions_array, agents, num_envs, num_agents)
            env_act = {k: v.flatten() for k, v in env_act.items()}

            rng, sub = jax.random.split(rng)
            step_rng = jax.random.split(sub, num_envs)
            obs2, env_state2, reward, done, info = jax.vmap(
                lambda k, s, a: step_switch(k, s, a, jnp.int32(env_idx))
            )(step_rng, env_state, env_act)

            rewards = rewards + sum(reward[a] for a in agents)
            soups = soups + sum(info["soups"][a] for a in agents)
            return (env_state2, obs2, rewards, soups, rng), None

        (_, _, total_rewards, total_soups, _), _ = jax.lax.scan(
            one_step, (env_state, obs, total_rewards, total_soups, rng),
            xs=None, length=num_steps
        )
        return total_rewards.mean(), total_soups.mean()

    return evaluate_env


def make_q_importance_fn(reset_switch, step_switch, network, agents, use_cnn: bool,
                         max_episodes: int, max_steps: int,
                         norm_importance: bool, stride: int):
    """
    Importance estimation for Q-networks (used by EWC/MAS).
    Operates on Q-network params only; called with train_state.params["q"].
    """
    num_agents = len(agents)

    @jax.jit
    def q_importance(params, env_idx: jnp.int32, rng):
        importance0 = jax.tree.map(jnp.zeros_like, params)

        def one_episode(carry, _):
            rng, acc = carry
            rng, r = jax.random.split(rng)
            obs, state = reset_switch(r, env_idx)

            def one_step(carry, t):
                obs, state, acc, rng = carry
                rng, s1, s2 = jax.random.split(rng, 3)

                obs_b = batchify(obs, agents, num_agents, not use_cnn)
                obs_b = obs_b.reshape(num_agents, 1, -1)  # (A, 1, obs_dim)

                def q_l2_loss(p):
                    q = jax.vmap(
                        lambda pp, o: network.apply(pp, o, env_idx=env_idx),
                        in_axes=(None, 0)
                    )(p, obs_b)  # (A, 1, action_dim)
                    q = q.squeeze(1)  # (A, action_dim)
                    return 0.5 * jnp.sum(q * q) / q.shape[0]

                grads = jax.grad(q_l2_loss)(params)
                alpha = (t % stride == 0).astype(jnp.float32)
                g2 = jax.tree.map(lambda g: g * g * alpha, grads)
                acc = jax.tree.map(jnp.add, acc, g2)

                # Greedy step to advance state
                q_vals = jax.vmap(
                    lambda pp, o: network.apply(pp, o, env_idx=env_idx), in_axes=(None, 0)
                )(params, obs_b).squeeze(1)  # (A, action_dim)
                acts = jnp.argmax(q_vals, axis=-1)  # (A,)
                env_act = {a: acts[i:i + 1] for i, a in enumerate(agents)}
                obs2, state2, _, _, _ = step_switch(s2, state, env_act, env_idx)

                return (obs2, state2, acc, rng), None

            (_, _, acc, rng), _ = jax.lax.scan(
                one_step, (obs, state, acc, rng), xs=jnp.arange(max_steps)
            )
            return (rng, acc), None

        (_, importance), _ = jax.lax.scan(
            one_episode, (rng, importance0), xs=None, length=max_episodes
        )
        importance = jax.tree.map(
            lambda x: x / (max_episodes * max_steps + 1e-8), importance
        )
        if norm_importance:
            total_abs = jax.tree_util.tree_reduce(
                lambda a, x: a + jnp.sum(jnp.abs(x)), importance, 0.0
            )
            n_params = jax.tree_util.tree_reduce(
                lambda a, x: a + x.size, importance, 0
            )
            mean_abs = total_abs / (n_params + 1e-8)
            importance = jax.tree.map(lambda x: x / (mean_abs + 1e-8), importance)
        return importance

    return q_importance


def rollout_for_video_qmix(rng, config, train_state, env, network, env_idx=0, max_steps=300):
    """
    Record a single-environment rollout using greedy Q-values for visualization.
    Returns a list of raw env states.
    """
    rng, env_rng = jax.random.split(rng)
    obs, state = env.reset(env_rng)
    done = False
    step_count = 0
    states = [env.unwrap_env_state(state)]

    q_params = train_state.params["q"]
    while not done and step_count < max_steps:
        actions = {}
        for agent_id in env.agents:
            obs_v = obs[agent_id]
            obs_b = obs_v[None]  # (1, *obs_shape)
            if not config.use_cnn:
                obs_b = obs_b.reshape(1, -1)  # (1, flat_obs_dim)
            q_vals = network.apply(q_params, obs_b, env_idx=env_idx)
            actions[agent_id] = jnp.argmax(q_vals[0])

        rng, key_step = jax.random.split(rng)
        next_obs, next_state, reward, done_info, info = env.step(key_step, state, actions)
        done = done_info["__all__"]
        obs, state = next_obs, next_state
        step_count += 1
        states.append(env.unwrap_env_state(state))

    return states


############################
######  MAIN FUNCTION  #####
############################

def main():
    jax.config.update("jax_platform_name", "gpu")
    print("Device:", jax.devices())

    cfg = tyro.cli(Config)

    # Expand non_stationary flag
    if cfg.non_stationary:
        cfg.sticky_actions = True
        cfg.slippery_tiles = True
        cfg.random_pot_size = True
        cfg.random_cook_time = True

    # Validate reward settings
    if cfg.sparse_rewards and cfg.individual_rewards:
        raise ValueError(
            "Cannot enable both sparse_rewards and individual_rewards simultaneously."
        )

    if cfg.single_task_idx is not None:
        cfg.cl_method = "ft"
    if cfg.cl_method is None:
        raise ValueError(
            "cl_method is required (e.g., ewc, mas, l2, ft, agem)."
        )
    # Default regularization coefficients
    if cfg.reg_coef is None:
        cfg.reg_coef = {"ewc": 1e11, "mas": 1e9, "l2": 1e7}.get(cfg.cl_method.lower(), 0.0)

    method_map = dict(
        ewc=EWC(mode=cfg.importance_mode, decay=cfg.importance_decay),
        mas=MAS(mode=cfg.importance_mode, decay=cfg.importance_decay),
        l2=L2(),
        ft=FT(),
        agem=AGEM(),
    )
    cl = method_map[cfg.cl_method.lower()]

    difficulty = cfg.difficulty
    seq_length = cfg.seq_length
    strategy = cfg.strategy
    seed = cfg.seed

    # Create environment sequence
    envs = make_sequence(
        sequence_length=seq_length,
        strategy=strategy,
        env_id=cfg.env_name,
        seed=seed,
        num_agents=cfg.num_agents,
        max_steps=cfg.max_episode_steps,
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
        f"{cfg.alg_name}_{cfg.cl_method}_{difficulty}_{cfg.num_agents}agents"
        f"_{network_arch}_seq{seq_length}_{strategy}_seed_{seed}_{timestamp}"
    )
    exp_dir = os.path.join("runs", run_name)

    # Logging setup
    if cfg.use_wandb:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(
            project=cfg.project,
            config=asdict(cfg),
            mode=cfg.wandb_mode,
            tags=cfg.tags or [],
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

    # Wrap environments with LogWrapper, compute max soup
    env_names = []
    max_soup_vals = []
    for i, env in enumerate(envs):
        envs[i] = LogWrapper(env, replace_info=False)
        env_names.append(envs[i].layout_name)
        max_soup_vals.append(
            calculate_max_soup(envs[i].layout, envs[i].max_steps, n_agents=envs[i].num_agents)
        )
    max_soup_vals = jnp.asarray(max_soup_vals, dtype=jnp.float32)

    # Single-task baseline: trim to one env
    if cfg.single_task_idx is not None:
        idx = cfg.single_task_idx
        envs = [envs[idx]]
        env_names = [env_names[idx]]
        max_soup_vals = max_soup_vals[idx:idx + 1]
        cfg.seq_length = 1
        seq_length = 1

    # Derived config
    temp_env = envs[0]
    agents = temp_env.agents
    num_agents = temp_env.num_agents

    cfg.num_actors = num_agents * cfg.num_envs
    cfg.num_updates = int(cfg.steps_per_task // cfg.num_steps // cfg.num_envs)

    # Build CTRolloutManagers for training
    train_envs = [
        CTRolloutManager(env, batch_size=cfg.num_envs, preprocess_obs=False)
        for env in envs
    ]

    # Schedules
    total_grad_steps = cfg.update_epochs * cfg.num_minibatches * cfg.num_updates
    lr_scheduler = optax.linear_schedule(cfg.lr, 1e-10, total_grad_steps)
    lr = lr_scheduler if cfg.anneal_lr else cfg.lr
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.radam(learning_rate=lr),
    )
    eps_scheduler = optax.linear_schedule(
        cfg.eps_start, cfg.eps_finish, cfg.eps_decay * cfg.num_updates
    )

    # ── Network initialisation ────────────────────────────────────────────────
    rng = jax.random.PRNGKey(cfg.seed)
    rng, net_rng, mix_rng = jax.random.split(rng, 3)

    obs_dim = int(np.prod(temp_env.observation_space().shape))
    state_dim = num_agents * obs_dim  # global state = concat of all agent obs

    network = QNetwork(
        action_dim=train_envs[0].max_action_space,
        hidden_size=cfg.hidden_size,
        activation=cfg.activation,
        use_layer_norm=cfg.use_layer_norm,
        use_multihead=cfg.use_multihead,
        use_task_id=cfg.use_task_id,
        num_tasks=seq_length,
    )
    network.apply = jax.jit(network.apply)

    mixing_network = MixingNetwork(num_agents=num_agents, embed_dim=cfg.mixing_embed_dim)
    mixing_network.apply = jax.jit(mixing_network.apply)

    # Init Q-network params
    init_x = jnp.zeros((1, obs_dim))
    q_network_params = network.init(net_rng, init_x)

    # Init mixing network params
    dummy_qs = jnp.zeros((1, num_agents))
    dummy_state = jnp.zeros((1, state_dim))
    mix_params = mixing_network.init(mix_rng, dummy_qs, dummy_state)

    # Combined params: optimizer sees Q-network and mixing network jointly
    combined_params = {"q": q_network_params, "mix": mix_params}

    train_state = CustomTrainState.create(
        apply_fn=network.apply,
        params=combined_params,
        target_network_params=combined_params,
        tx=tx,
    )

    # Switches for single-env eval / importance (same pattern as ippo.py)
    reset_fns = tuple(env.reset for env in envs)
    step_fns = tuple(env.step for env in envs)

    def reset_switch(key, task_idx):
        return jax.lax.switch(task_idx, reset_fns, key)

    def step_switch(key, state, actions, task_idx):
        return jax.lax.switch(task_idx, step_fns, key, state, actions)

    # Evaluation function: greedy argmax Q-values (IGM property makes this correct for QMIX)
    evaluate_env = make_qmix_eval_fn(
        reset_switch, step_switch, network, agents,
        num_envs=cfg.eval_num_envs, num_steps=cfg.max_episode_steps, use_cnn=cfg.use_cnn
    )

    # Importance function: Q-network only (mixing network is not regularised)
    if cfg.cl_method.lower() in ("ewc", "mas"):
        importance_fn = make_q_importance_fn(
            reset_switch, step_switch, network, agents, cfg.use_cnn,
            cfg.importance_episodes, cfg.importance_steps,
            cfg.normalize_importance, cfg.importance_stride
        )
    else:
        importance_fn = cl.make_importance_fn(
            reset_switch, step_switch, network, agents, cfg.use_cnn,
            cfg.importance_episodes, cfg.importance_steps,
            cfg.normalize_importance, cfg.importance_stride
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Core training function (per environment)
    # ──────────────────────────────────────────────────────────────────────────

    N = cfg.num_steps * cfg.num_envs
    minibatch_size = N // cfg.num_minibatches

    @partial(jax.jit, static_argnums=(2, 4))
    def train_on_environment(rng, train_state, train_env, cl_state, env_idx):
        """Train on a single environment for cfg.num_updates update steps."""

        # Reset optimizer and update counter for this task
        new_opt_state = tx.init(train_state.params)
        train_state = train_state.replace(tx=tx, opt_state=new_opt_state, n_updates=0)

        def _update_step(runner_state, _):
            train_state, expl_state, rng = runner_state

            # ── COLLECTION PHASE ─────────────────────────────────────────────
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_action, rng_step = jax.random.split(rng, 3)

                obs_b = batchify(last_obs, agents, num_agents * cfg.num_envs, not cfg.use_cnn)
                obs_b = obs_b.reshape(num_agents, cfg.num_envs, -1)

                q_vals = jax.vmap(
                    lambda p, o: network.apply(p, o, env_idx=env_idx), in_axes=(None, 0)
                )(train_state.params["q"], obs_b)  # (A, num_envs, action_dim)

                avail_actions = train_env.get_valid_actions(env_state)
                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_action, num_agents)
                new_action = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, vdn_batchify(avail_actions, agents)
                )  # (A, num_envs)

                actions = vdn_unbatchify(new_action, agents)  # {agent: (num_envs,)}

                new_obs, new_env_state, rewards, dones, infos = train_env.batch_step(
                    rng_step, env_state, actions
                )

                # Reward shaping
                shaped_reward = infos.pop("shaped_reward")
                shaped_reward["__all__"] = vdn_batchify(shaped_reward, agents).sum(axis=0)
                rewards = jax.tree.map(
                    lambda x, y: x + y,
                    rewards,
                    shaped_reward,
                )

                timestep = QMIXTimestep(
                    obs={a: last_obs[a] for a in agents},
                    global_state=last_obs["__all__"],  # global state at current step
                    actions=actions,
                    avail_actions=avail_actions,
                    rewards=rewards,
                    dones=dones,
                )
                return (new_obs, new_env_state, rng), (timestep, infos)

            rng, _rng = jax.random.split(rng)
            (new_obs, new_env_state, _), (timesteps, infos) = jax.lax.scan(
                _step_env, (*expl_state, _rng), xs=None, length=cfg.num_steps
            )
            expl_state = (new_obs, new_env_state)

            train_state = train_state.replace(
                timesteps=train_state.timesteps + cfg.num_steps * cfg.num_envs
            )

            # ── ON-POLICY LEARNING PHASE ──────────────────────────────────────
            # Build (s_t, s_{t+1}) pairs from the trajectory.
            # For t < T-1: next_obs[t] = obs[t+1]; for t = T-1: new_obs.
            # global_state follows the same shift.
            next_obs = {
                a: jnp.concatenate(
                    [timesteps.obs[a][1:], new_obs[a][jnp.newaxis]], axis=0
                )
                for a in agents
            }  # (num_steps, num_envs, obs_dim)

            next_global_state = jnp.concatenate(
                [timesteps.global_state[1:], new_obs["__all__"][jnp.newaxis]], axis=0
            )  # (num_steps, num_envs, state_dim)

            # Flatten (num_steps, num_envs, ...) → (N, ...)
            obs_flat  = {a: timesteps.obs[a].reshape(N, -1) for a in agents}
            nxt_flat  = {a: next_obs[a].reshape(N, -1) for a in agents}
            act_flat  = {a: timesteps.actions[a].reshape(N) for a in agents}
            gs_flat   = timesteps.global_state.reshape(N, -1)
            ngs_flat  = next_global_state.reshape(N, -1)
            rew_flat  = timesteps.rewards["__all__"].reshape(N)
            don_flat  = timesteps.dones["__all__"].reshape(N).astype(jnp.float32)

            # Compute QMIX TD-targets once using the frozen target networks.
            tgt_q_params  = train_state.target_network_params["q"]
            tgt_mix_params = train_state.target_network_params["mix"]

            nxt_b = jnp.stack([nxt_flat[a] for a in agents])  # (A, N, obs_dim)
            q_next = jax.vmap(
                lambda p, o: network.apply(p, o, env_idx=env_idx), in_axes=(None, 0)
            )(tgt_q_params, nxt_b)  # (A, N, action_dim)
            q_next_max   = jnp.max(q_next, axis=-1)  # (A, N)
            q_next_max_T = q_next_max.T               # (N, A)

            q_total_next = mixing_network.apply(tgt_mix_params, q_next_max_T, ngs_flat)  # (N,)
            qmix_target  = rew_flat + (1 - don_flat) * cfg.gamma * q_total_next
            # shape: (N,) – stop_gradient implicit: uses target_network_params

            def _learn_minibatch(train_state, mb_indices):
                """One gradient step on a minibatch of on-policy transitions."""
                mb_obs  = {a: obs_flat[a][mb_indices] for a in agents}
                mb_acts = {a: act_flat[a][mb_indices] for a in agents}
                mb_gs   = gs_flat[mb_indices]   # (mb, state_dim)
                mb_tgt  = qmix_target[mb_indices]

                def _loss_fn(combined_params):
                    q_params  = combined_params["q"]
                    mix_params = combined_params["mix"]

                    obs_b = jnp.stack([mb_obs[a] for a in agents])  # (A, mb, obs_dim)
                    q_vals = jax.vmap(
                        lambda p, o: network.apply(p, o, env_idx=env_idx), in_axes=(None, 0)
                    )(q_params, obs_b)  # (A, mb, action_dim)

                    chosen_q = jnp.take_along_axis(
                        q_vals,
                        jnp.stack([mb_acts[a] for a in agents])[..., jnp.newaxis],
                        axis=-1,
                    ).squeeze(-1)  # (A, mb)
                    chosen_q_T = chosen_q.T  # (mb, A)

                    q_total = mixing_network.apply(mix_params, chosen_q_T, mb_gs)  # (mb,)
                    td_loss = jnp.mean((q_total - mb_tgt) ** 2)

                    # CL regularization on Q-network params only
                    cl_penalty = cl.penalty(q_params, cl_state, cfg.reg_coef)
                    total_loss = td_loss + cl_penalty
                    return total_loss, (td_loss, q_total.mean(), cl_penalty)

                (total_loss, (td_loss, qvals, cl_penalty)), grads = jax.value_and_grad(
                    _loss_fn, has_aux=True
                )(train_state.params)

                if cfg.cl_method.lower() == "agem":
                    # AGEM gradient projection over combined (Q + mixing) params
                    past_sizes = cl_state.sizes.at[env_idx].set(0)
                    samples_per_task = max(1, cfg.agem_sample_size // cl_state.max_tasks)
                    g_mem = jax.tree.map(jnp.zeros_like, grads)
                    for _t in range(cl_state.max_tasks):
                        _t_rng = jax.random.fold_in(rng, _t)
                        (
                            _t_obs, _t_global, _t_acts, _t_rews,
                            _t_nobs, _t_next_global, _t_dones
                        ) = sample_qmix_task_slot(cl_state, _t, samples_per_task, _t_rng)
                        _t_grads = compute_qmix_memory_gradient(
                            network, mixing_network,
                            train_state.params,
                            train_state.target_network_params,
                            cfg.gamma,
                            _t_obs, _t_global, _t_acts, _t_rews,
                            _t_nobs, _t_next_global, _t_dones,
                            env_idx=_t,
                        )
                        _mask = (past_sizes[_t] > 0).astype(jnp.float32)
                        _t_grads = jax.tree.map(lambda g, m=_mask: g * m, _t_grads)
                        g_mem = jax.tree.map(jnp.add, g_mem, _t_grads)
                    grads, _ = agem_project(grads, g_mem)

                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)
                return train_state, (total_loss, td_loss, qvals, cl_penalty)

            def _learn_epoch(carry, _):
                train_state, rng = carry
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, N).reshape(
                    cfg.num_minibatches, minibatch_size
                )
                train_state, losses = jax.lax.scan(_learn_minibatch, train_state, perm)
                return (train_state, rng), losses

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (total_loss, td_loss, qvals, cl_penalty) = jax.lax.scan(
                _learn_epoch, (train_state, _rng), xs=None, length=cfg.update_epochs
            )
            # Reduce (update_epochs, num_minibatches) → scalar
            total_loss  = total_loss.reshape(-1).mean()
            td_loss     = td_loss.reshape(-1).mean()
            qvals       = qvals.reshape(-1).mean()
            cl_penalty  = cl_penalty.reshape(-1).mean()

            # Update target networks every target_update_interval collection phases
            train_state = jax.lax.cond(
                train_state.n_updates % cfg.target_update_interval == 0,
                lambda ts: ts.replace(
                    target_network_params=optax.incremental_update(
                        ts.params, ts.target_network_params, cfg.tau
                    )
                ),
                lambda ts: ts,
                operand=train_state,
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            metrics = {
                "General/env_index": jnp.int32(env_idx),
                "General/env_step": train_state.n_updates * cfg.num_steps * cfg.num_envs,
                "General/steps_for_env": train_state.n_updates * cfg.num_steps * cfg.num_envs,
                "General/update_step": train_state.n_updates,
                "General/grad_steps": train_state.grad_steps,
                "General/epsilon": eps_scheduler(train_state.n_updates),
                "General/learning_rate": lr_scheduler(
                    train_state.grad_steps) if cfg.anneal_lr else cfg.lr,
                "Losses/total_loss": total_loss,
                "Losses/td_loss": td_loss,
                "Losses/reg_loss": cl_penalty,
                "Values/qvals": qvals,
                "Rewards/step_reward": timesteps.rewards["__all__"].mean(),
            }
            # Soup: per-episode averages matching ippo logging style
            soups_info = infos.pop("soups", {})
            soups_tea = jnp.stack([soups_info[a] for a in agents], axis=-1)  # (T, E, A)
            soups_per_env = soups_tea.sum(axis=(0, 2))  # (E,)
            don_te = timesteps.dones["__all__"]  # (T, E)
            episodes_per_env = don_te.sum(axis=0)  # (E,)
            mask = episodes_per_env > 0
            true_avg = jnp.where(mask, soups_per_env / jnp.maximum(episodes_per_env, 1), 0.0)
            num_finished = jnp.maximum(mask.sum(), 1)
            max_per_episode = max_soup_vals[env_idx]
            metrics["Soup/total"] = true_avg.sum() / num_finished
            metrics["Soup/scaled"] = jnp.where(
                max_per_episode > 0, (true_avg / max_per_episode).sum() / num_finished, 0.0
            )
            for ai, agent in enumerate(agents):
                soups_te = soups_tea[:, :, ai].sum(axis=0)  # (E,)
                per_agent = jnp.where(mask, soups_te / jnp.maximum(episodes_per_env, 1), 0.0)
                metrics[f"Soup/{agent}"] = per_agent.sum() / num_finished
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            def evaluate_and_log(rng, update_step):
                rng, eval_rng = jax.random.split(rng)

                def log_metrics(metrics, update_step):
                    if cfg.evaluation:
                        # Pass Q-network params only to eval (mixing network not needed)
                        avg_rewards, avg_soups = evaluate_all_envs(
                            eval_rng, train_state.params["q"], seq_length, evaluate_env
                        )
                        metrics = add_eval_metrics(
                            avg_rewards, avg_soups, env_names, max_soup_vals, metrics
                        )

                    def callback(m, step, env_ctr):
                        real_step = int((env_ctr - 1) * cfg.num_updates + step)
                        for k, v in m.items():
                            writer.add_scalar(k, float(v), real_step)
                        if cfg.use_wandb:
                            wandb.log({k: float(v) for k, v in m.items()}, step=real_step)

                    jax.debug.callback(callback, metrics, update_step, jnp.int32(env_idx + 1))
                    return None

                jax.lax.cond(
                    (update_step % cfg.log_interval) == 0,
                    log_metrics,
                    lambda m, s: None,
                    metrics,
                    update_step,
                )

            evaluate_and_log(rng=rng, update_step=train_state.n_updates)

            runner_state = (train_state, expl_state, rng)
            return runner_state, metrics

        # Initial reset for this task
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = train_env.batch_reset(reset_rng)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, (obs, env_state), _rng)
        runner_state, _ = jax.lax.scan(
            _update_step, runner_state, xs=None, length=cfg.num_updates
        )

        final_train_state = runner_state[0]
        final_expl_state  = runner_state[1]

        # AGEM: collect one final on-policy batch with the trained policy and
        # store it as the memory for this task.
        if cfg.cl_method.lower() == "agem":
            rng, mem_rng = jax.random.split(rng)

            def _mem_step(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                obs_b = batchify(last_obs, agents, num_agents * cfg.num_envs, not cfg.use_cnn)
                obs_b = obs_b.reshape(num_agents, cfg.num_envs, -1)

                q_vals = jax.vmap(
                    lambda p, o: network.apply(p, o, env_idx=env_idx), in_axes=(None, 0)
                )(final_train_state.params["q"], obs_b)

                avail_actions = train_env.get_valid_actions(env_state)
                _rngs = jax.random.split(rng_a, num_agents)
                new_action = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, cfg.eps_finish, vdn_batchify(avail_actions, agents)
                )
                actions = vdn_unbatchify(new_action, agents)

                new_obs, new_env_state, rewards, dones, infos = train_env.batch_step(
                    rng_s, env_state, actions
                )
                shaped_reward = infos.pop("shaped_reward")
                shaped_reward["__all__"] = vdn_batchify(shaped_reward, agents).sum(axis=0)
                rewards = jax.tree.map(lambda x, y: x + y, rewards, shaped_reward)

                timestep = QMIXTimestep(
                    obs={a: last_obs[a] for a in agents},
                    global_state=last_obs["__all__"],
                    actions=actions,
                    avail_actions=avail_actions,
                    rewards=rewards,
                    dones=dones,
                )
                return (new_obs, new_env_state, rng), timestep

            (new_obs_mem, _, _), mem_ts = jax.lax.scan(
                _mem_step, (*final_expl_state, mem_rng), xs=None, length=cfg.num_steps
            )

            next_obs_mem = {
                a: jnp.concatenate(
                    [mem_ts.obs[a][1:], new_obs_mem[a][jnp.newaxis]], axis=0
                )
                for a in agents
            }
            next_gs_mem = jnp.concatenate(
                [mem_ts.global_state[1:], new_obs_mem["__all__"][jnp.newaxis]], axis=0
            )

            N_mem = cfg.num_steps * cfg.num_envs
            obs_m  = {a: mem_ts.obs[a].reshape(N_mem, -1) for a in agents}
            nxt_m  = {a: next_obs_mem[a].reshape(N_mem, -1) for a in agents}
            act_m  = {a: mem_ts.actions[a].reshape(N_mem) for a in agents}
            gs_m   = mem_ts.global_state.reshape(N_mem, -1)
            ngs_m  = next_gs_mem.reshape(N_mem, -1)
            rew_m  = mem_ts.rewards["__all__"].reshape(N_mem)
            don_m  = mem_ts.dones["__all__"].reshape(N_mem).astype(jnp.float32)

            rng, samp_rng = jax.random.split(rng)
            samp_idx = jax.random.choice(samp_rng, N_mem, (cfg.agem_sample_size,), replace=False)

            _obs_b  = jnp.stack([obs_m[a][samp_idx] for a in agents]).transpose(1, 0, 2)
            _nxt_b  = jnp.stack([nxt_m[a][samp_idx] for a in agents]).transpose(1, 0, 2)
            _act_b  = jnp.stack([act_m[a][samp_idx] for a in agents]).T
            _gs     = gs_m[samp_idx]
            _ngs    = ngs_m[samp_idx]
            _rew    = rew_m[samp_idx]
            _don    = don_m[samp_idx]

            cl_state = update_qmix_agem_memory(
                cl_state, env_idx,
                _obs_b, _gs, _act_b, _rew, _nxt_b, _ngs, _don,
            )

        return rng, final_train_state, cl_state

    # ──────────────────────────────────────────────────────────────────────────

    def save_params(path, train_state, env_kwargs=None, layout_name=None, config=None):
        """Saves model parameters and optional metadata to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(flax.serialization.to_bytes({"params": train_state.params}))

        def _convert(obj):
            if isinstance(obj, flax.core.frozen_dict.FrozenDict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(i) for i in obj]
            if isinstance(obj, jax.Array):
                arr = np.array(obj)
                return arr.item() if arr.size == 1 else arr.tolist()
            return obj

        meta = {}
        if env_kwargs is not None:
            meta["env_kwargs"] = _convert(env_kwargs)
        if layout_name is not None:
            meta["layout_name"] = layout_name
        if config is not None:
            meta.update({
                "use_cnn": config.use_cnn,
                "num_tasks": seq_length,
                "use_multihead": config.use_multihead,
                "use_task_id": config.use_task_id,
                "regularize_heads": config.regularize_heads,
                "use_layer_norm": config.use_layer_norm,
                "activation": config.activation,
                "strategy": config.strategy,
                "seed": config.seed,
                "mixing_embed_dim": config.mixing_embed_dim,
            })
        if meta:
            with open(f"{path}_config.json", "w") as f:
                json.dump(_convert(meta), f, indent=2)
        print("model saved to", path)

    def loop_over_envs(rng, train_state, cl_state):
        rng, *env_rngs = jax.random.split(rng, seq_length + 1)

        visualizer = None
        for task_idx, (env_rng, train_env, env) in enumerate(
                zip(env_rngs, train_envs, envs)
        ):
            print(f"Training on task {task_idx + 1}/{seq_length}: {env.layout_name}")

            rng, train_state, cl_state = train_on_environment(
                env_rng, train_state, train_env, cl_state, task_idx
            )

            # Update CL state with Q-network params only (mixing network not regularised)
            # No-op for AGEM — memory was updated inside train_on_environment
            importance = importance_fn(train_state.params["q"], task_idx, rng)
            cl_state = cl.update_state(cl_state, train_state.params["q"], importance)

            # Optional video recording
            if cfg.record_video:
                if visualizer is None:
                    visualizer = create_visualizer(num_agents, cfg.env_name)
                states = rollout_for_video_qmix(
                    rng, cfg, train_state, env, network, task_idx, cfg.video_length
                )
                file_path = f"{exp_dir}/task_{task_idx}_{env.layout_name}.mp4"
                visualizer.animate(states, out_path=file_path, task_idx=task_idx, env=env)

            # Save model
            repo_root = Path(__file__).resolve().parent.parent
            path = (
                f"{repo_root}/checkpoints/overcooked/{cfg.cl_method}"
                f"/{run_name}/model_env_{task_idx + 1}"
            )
            save_params(path, train_state, env_kwargs=env.layout,
                        layout_name=env.layout_name, config=cfg)

            if cfg.single_task_idx is not None:
                break

    # ── Run ──────────────────────────────────────────────────────────────────
    if cfg.cl_method.lower() == "agem":
        cl_state = init_qmix_agem_memory(
            max_size=cfg.agem_memory_size,
            num_agents=num_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
            max_tasks=seq_length,
        )
    else:
        # CL state tracks Q-network params only
        cl_state = init_cl_state(q_network_params, False, cfg.regularize_heads)

    rng, train_rng = jax.random.split(rng)
    loop_over_envs(train_rng, train_state, cl_state)


if __name__ == "__main__":
    print("Running main...")
    main()
