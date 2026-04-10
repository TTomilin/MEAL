import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence, Optional, List, Literal

import flax
import numpy as np
import optax
import tyro
import wandb
from flax.core.frozen_dict import unfreeze
from flax.training.train_state import TrainState
from jax._src.flatten_util import ravel_pytree

from experiments.continual.agem import (AGEM, init_agem_memory, sample_task_slot,
                                        agem_project, update_agem_memory)
from experiments.continual.base import RegCLMethod
from experiments.continual.er_ace import ERACE
from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.l2 import L2
from experiments.continual.mas import MAS
from experiments.continual.packnet import Packnet
from experiments.evaluation import evaluate_all_envs, make_eval_fn
from experiments.model.decoupled_mlp import Actor, Critic
from experiments.utils import *
from meal import make_sequence
from meal.env.utils.max_soup_calculator import calculate_max_soup
from meal.wrappers.logging import LogWrapper


@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING / PPO PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    alg_name: Literal["ippo", "mappo"] = "mappo"
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
    sparse_rewards: bool = False  # Only shared reward for soup delivery
    individual_rewards: bool = False  # Only respective agent gets reward for their actions

    # ═══════════════════════════════════════════════════════════════════════════
    # NETWORK ARCHITECTURE PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    activation: str = "relu"
    use_cnn: bool = False
    use_layer_norm: bool = True
    big_network: bool = False

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTINUAL LEARNING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    cl_method: Optional[str] = None
    reg_coef: Optional[float] = None
    use_task_id: bool = True
    use_agent_id: bool = True  # MAPPO-specific: append agent one-hot to actor obs
    use_multihead: bool = True
    shared_backbone: bool = False
    normalize_importance: bool = False
    regularize_critic: bool = False
    regularize_heads: bool = False
    reset_optimizer: bool = True

    # Regularization method specific parameters
    importance_episodes: int = 5
    importance_stride: int = 5  # compute importance once every N steps
    importance_steps: int = 500
    importance_mode: str = "online"  # "online", "last", or "multi" — for EWC & MAS
    importance_decay: float = 0.9  # Only for online EWC & MAS

    # AGEM specific parameters
    agem_memory_size: int = 100000
    agem_sample_size: int = 1024
    agem_gradient_scale: float = 1.0
    er_ace_coef: float = 1.0

    # Packnet specific parameters
    train_epochs: int = 8
    finetune_epochs: int = 2
    finetune_timesteps: float = 1e7
    re_init_pruned_weights: bool = False

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
    log_interval: int = 75
    renderer_version: str = "v1"
    eval_deterministic: bool = False

    # ═══════════════════════════════════════════════════════════════════════════
    # LOGGING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    use_wandb: bool = False
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
    finetune_updates: int = 0
    minibatch_size: int = 0


############################
######  MAIN FUNCTION  #####
############################


def create_global_state_for_critic(obs_dict, agent_list, num_envs, use_cnn=False):
    """
    Create global state for MAPPO critic by concatenating all agents' observations.

    Returns:
        - MLP: (num_envs, num_agents * flattened_obs_dim)
        - CNN: (num_envs, height, width, num_agents * channels)
    """
    agent_obs = jnp.stack([obs_dict[agent] for agent in agent_list])

    if use_cnn:
        agent_obs = jnp.transpose(agent_obs, (1, 0, 2, 3, 4))
        global_state = jnp.concatenate([agent_obs[:, i] for i in range(agent_obs.shape[1])], axis=-1)
    else:
        agent_obs = agent_obs.reshape(agent_obs.shape[0], agent_obs.shape[1], -1)
        agent_obs = jnp.transpose(agent_obs, (1, 0, 2))
        global_state = agent_obs.reshape(num_envs, -1)

    return global_state


def main():
    jax.config.update("jax_platform_name", "gpu")
    print("Device: ", jax.devices())

    cfg = tyro.cli(Config)

    # If non-stationary mode is enabled, force all 4 env knobs on
    if cfg.non_stationary:
        cfg.sticky_actions = True
        cfg.slippery_tiles = True
        cfg.random_pot_size = True
        cfg.random_cook_time = True

    # Validate reward settings
    if cfg.sparse_rewards and cfg.individual_rewards:
        raise ValueError(
            "Cannot enable both sparse_rewards and individual_rewards simultaneously.")

    if cfg.single_task_idx is not None:
        cfg.cl_method = "ft"
    if cfg.cl_method is None:
        raise ValueError(
            "cl_method is required. Please specify a continual learning method (e.g., ewc, mas, l2, ft, agem).")

    difficulty = cfg.difficulty
    seq_length = cfg.seq_length
    strategy = cfg.strategy
    seed = cfg.seed

    # Set default regularization coefficient
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
        er_ace=ERACE(memory_size=cfg.agem_memory_size, sample_size=cfg.agem_sample_size),
                      packnet=Packnet(seq_length=cfg.seq_length, prune_instructions=0.4,
                      train_finetune_split=(cfg.train_epochs, cfg.finetune_epochs),
                      prunable_layers=[nn.Dense, nn.Conv],
                      re_init_pruned_weights=cfg.re_init_pruned_weights)
    )

    cl = method_map[cfg.cl_method.lower()]

    # Create environment sequence
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
    run_name = (f'{cfg.alg_name}_{cfg.cl_method}_{difficulty}_{cfg.num_agents}agents_'
                f'{network_arch}_seq{seq_length}_{strategy}_seed_{seed}_{timestamp}')
    exp_dir = os.path.join("runs", run_name)

    # Initialize WandB
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

    # Set up Tensorboard
    writer = SummaryWriter(exp_dir)
    rows = []
    for key, value in vars(cfg).items():
        value_str = str(value).replace("\n", "<br>").replace("|", "\\|")
        rows.append(f"|{key}|{value_str}|")
    writer.add_text("hyperparameters", f"|param|value|\n|-|-|\n" + "\n".join(rows))

    # Collect env metadata
    env_names = []
    max_soup_vals = []
    for i, env in enumerate(envs):
        env_w = LogWrapper(env, replace_info=False)
        env_names.append(env_w.layout_name)
        max_soup_vals.append(
            calculate_max_soup(env_w.layout, env_w.max_steps, n_agents=env_w.num_agents)
        )

    max_soup_vals = jnp.asarray(max_soup_vals, dtype=jnp.float32)

    temp_env = envs[0]
    num_agents = temp_env.num_agents
    agents = temp_env.agents

    cfg.num_actors = num_agents * cfg.num_envs
    cfg.num_updates = int(cfg.steps_per_task // cfg.num_steps // cfg.num_envs)
    cfg.minibatch_size = (cfg.num_actors * cfg.num_steps) // cfg.num_minibatches

    def linear_schedule(count):
        frac = 1.0 - (count // (cfg.num_minibatches * cfg.update_epochs)) / cfg.num_updates
        return cfg.lr * frac

    # ── Build reset/step switches (same pattern as ippo) ─────────────────────
    reset_fns = tuple(env.reset for env in envs)
    step_fns = tuple(env.step for env in envs)

    def reset_switch(key, task_idx):
        return jax.lax.switch(task_idx, reset_fns, key)

    def step_switch(key, state, actions, task_idx):
        return jax.lax.switch(task_idx, step_fns, key, state, actions)

    # ── Build MAPPO actor / critic networks ───────────────────────────────────
    obs_dim = temp_env.observation_space().shape
    if not cfg.use_cnn:
        local_obs_dim = int(np.prod(obs_dim))
        global_obs_dim = local_obs_dim * num_agents
    else:
        local_obs_dim = obs_dim
        global_obs_dim = (obs_dim[0], obs_dim[1], obs_dim[2] * num_agents)

    if cfg.cl_method == "packnet" and cfg.use_cnn:
        raise ValueError("Packnet currently doesn't support CNN.")

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

    rng = jax.random.PRNGKey(seed)
    rng, actor_rng, critic_rng = jax.random.split(rng, 3)

    if cfg.use_cnn:
        actor_init_x = jnp.zeros((1, *local_obs_dim))
        critic_init_x = jnp.zeros((1, *global_obs_dim))
    else:
        actor_init_x = jnp.zeros((1, local_obs_dim))
        critic_init_x = jnp.zeros((1, global_obs_dim))

    actor_params = actor_network.init(actor_rng, actor_init_x, env_idx=0)
    critic_params = critic_network.init(critic_rng, critic_init_x, env_idx=0)
    network_params = {'actor': actor_params, 'critic': critic_params}

    # Wrapper used by make_eval_fn and make_importance_fn (both expect 3-tuple)
    class DecoupledNetworkWrapper:
        action_dim = temp_env.action_space().n

        def apply(self, params, obs, *, env_idx=0):
            pi, dormant_ratio = actor_network.apply(params['actor'], obs, env_idx=env_idx)
            dummy_value = jnp.zeros(obs.shape[0])
            return pi, dummy_value, dormant_ratio

    network = DecoupledNetworkWrapper()

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=linear_schedule if cfg.anneal_lr else cfg.lr, eps=1e-5),
    )

    train_state = TrainState.create(
        apply_fn=lambda params, obs, **kw: None,  # not used; actor/critic called directly
        params=network_params,
        tx=tx,
    )

    evaluate_env = make_eval_fn(cl, reset_switch, step_switch, network, agents, seq_length,
                                cfg.num_steps, cfg.use_cnn, cfg.eval_deterministic, cfg.seed)

    importance_fn = cl.make_importance_fn(
        reset_switch, step_switch, network, agents, cfg.use_cnn,
        cfg.importance_episodes, cfg.importance_steps,
        cfg.normalize_importance, cfg.importance_stride,
    )

    # ─────────────────────────────────────────────────────────────────────────
    @jax.jit
    def train_on_environment(rng, train_state, cl_state, env_idx):
        """Train MAPPO on a single task (env_idx)."""

        if cfg.reset_optimizer:
            new_opt_state = train_state.tx.init(train_state.params)
            train_state = train_state.replace(tx=tx, opt_state=new_opt_state)

        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, cfg.num_envs)
        obsv, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(reset_rng)

        reward_shaping_horizon = cfg.steps_per_task / 2
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1., end_value=0., transition_steps=reward_shaping_horizon
        )

        def _update_step(runner_state, _):
            # ── COLLECT TRAJECTORIES ──────────────────────────────────────────
            def _env_step(runner_state, _):
                train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state
                rng, _rng = jax.random.split(rng)

                obs_batch = batchify(last_obs, agents, cfg.num_actors, not cfg.use_cnn)
                global_state = create_global_state_for_critic(
                    last_obs, agents, cfg.num_envs, cfg.use_cnn
                )

                # Actor uses local obs; critic uses global state
                pi, _actor_dormant = actor_network.apply(
                    train_state.params['actor'], obs_batch, env_idx=env_idx
                )
                value_per_env, _critic_dormant = critic_network.apply(
                    train_state.params['critic'], global_state, env_idx=env_idx
                )
                # Tile value/global_state: blocked layout to match batchify
                # batchify gives [agent0_env0..N, agent1_env0..N] (blocked, not interleaved)
                # jnp.tile gives [env0..N, env0..N] which aligns correctly
                value = jnp.tile(value_per_env, len(agents))
                global_state_batch = jnp.tile(global_state, (len(agents), 1))

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

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
                    total_delivery = sum(reward[a] for a in agents)
                    shared = {a: total_delivery for a in agents}
                    reward = jax.tree_util.tree_map(
                        lambda x, y: x + y * rew_shaping_anneal(current_timestep),
                        shared, info["shaped_reward"],
                    )

                transition = Transition_MAPPO(
                    batchify(done, agents, cfg.num_actors, not cfg.use_cnn).squeeze(),
                    action,
                    value,
                    batchify(reward, agents, cfg.num_actors).squeeze(),
                    log_prob,
                    obs_batch,
                    global_state_batch,
                )
                steps_for_env = steps_for_env + cfg.num_envs
                runner_state = (train_state, env_state, obsv, update_step, steps_for_env, rng, cl_state)
                return runner_state, (transition, info)

            runner_state, (traj_batch, info) = jax.lax.scan(
                _env_step, runner_state, xs=None, length=cfg.num_steps
            )
            train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state

            last_obs_batch = batchify(last_obs, agents, cfg.num_actors, not cfg.use_cnn)
            last_global_state = create_global_state_for_critic(last_obs, agents, cfg.num_envs, cfg.use_cnn)
            last_val_per_env, _ = critic_network.apply(
                train_state.params['critic'], last_global_state, env_idx=env_idx
            )
            last_val = jnp.tile(last_val_per_env, len(agents))

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

            # ── UPDATE NETWORK ────────────────────────────────────────────────
            def _update_epoch(update_state, _):
                def _update_minbatch(carry, batch_info):
                    train_state, cl_state, rng = carry
                    rng, agem_rng = jax.random.split(rng)
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        local_obs = traj_batch.obs
                        global_state = traj_batch.global_state

                        pi, _ = actor_network.apply(params['actor'], local_obs, env_idx=env_idx)
                        value, _ = critic_network.apply(params['critic'], global_state, env_idx=env_idx)
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -cfg.clip_eps, cfg.clip_eps
                        )
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(value - targets),
                            jnp.square(value_pred_clipped - targets),
                        ).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * gae,
                        ).mean()
                        entropy = pi.entropy().mean()

                        cl_penalty = cl.penalty(params, cl_state, cfg.reg_coef) if isinstance(cl, RegCLMethod) else 0.0
                        total_loss = (loss_actor + cfg.vf_coef * value_loss
                                      - cfg.ent_coef * entropy + cl_penalty)
                        return total_loss, (value_loss, loss_actor, entropy, cl_penalty)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)

                    agem_stats = {}

                    if cfg.cl_method.lower() == "agem" and cl_state is not None:
                        # Zero out the current task's slot so we don't replay it
                        past_sizes = cl_state.sizes.at[env_idx].set(0)

                        def apply_agem_projection(agem_rng, past_sizes):
                            max_tasks = cl_state.obs.shape[0]
                            samples_per_task = max(cfg.agem_sample_size // max_tasks, 1)

                            grads_mem = None
                            ppo_stats_sum = {
                                "agem/ppo_total_loss": jnp.array(0.0),
                                "agem/ppo_value_loss": jnp.array(0.0),
                                "agem/ppo_actor_loss": jnp.array(0.0),
                                "agem/ppo_entropy": jnp.array(0.0),
                            }

                            # Actor-only BC loss for memory replay.
                            # The centralized critic uses global state which is not stored in
                            # AGEM memory (only local obs are stored), so we replay only
                            # the actor via behavioural cloning to prevent policy drift.
                            for t in range(max_tasks):
                                agem_rng, task_rng = jax.random.split(agem_rng)
                                t_obs, t_actions, t_logp, t_advs, t_targets, t_values = sample_task_slot(
                                    cl_state, t, samples_per_task, task_rng
                                )

                                def actor_bc_loss_fn(params, obs=t_obs, acts=t_actions, t=t):
                                    pi_m, _ = actor_network.apply(params['actor'], obs, env_idx=t)
                                    actor_loss = -jnp.mean(pi_m.log_prob(acts))
                                    entropy_m = jnp.mean(pi_m.entropy())
                                    return actor_loss - cfg.ent_coef * entropy_m, (actor_loss, entropy_m)

                                (t_total, (t_aloss, t_entropy)), t_grads = jax.value_and_grad(
                                    actor_bc_loss_fn, has_aux=True
                                )(train_state.params)

                                t_stats = {
                                    "agem/ppo_total_loss": t_total,
                                    "agem/ppo_actor_loss": t_aloss,
                                    "agem/ppo_entropy": t_entropy,
                                    "agem/ppo_value_loss": jnp.array(0.0),
                                }

                                mask = (past_sizes[t] > 0).astype(jnp.float32)
                                t_grads = jax.tree_util.tree_map(lambda g: g * mask, t_grads)
                                grads_mem = (t_grads if grads_mem is None
                                             else jax.tree_util.tree_map(lambda a, b: a + b, grads_mem, t_grads))
                                for k in ppo_stats_sum:
                                    ppo_stats_sum[k] = ppo_stats_sum[k] + t_stats[k] * mask

                            n_active = jnp.sum((past_sizes > 0).astype(jnp.float32)) + 1e-8
                            ppo_stats = {k: v / n_active for k, v in ppo_stats_sum.items()}

                            projected_grads, proj_stats = agem_project(grads, grads_mem)
                            combined_stats = {**ppo_stats, **proj_stats}
                            combined_stats["agem/mem_grad_norm_raw"] = jnp.linalg.norm(
                                ravel_pytree(grads_mem)[0]
                            )
                            total_used = jnp.sum(cl_state.sizes)
                            total_capacity = cl_state.obs.shape[0] * cl_state.max_size_per_task
                            combined_stats["agem/memory_fullness_pct"] = (total_used / total_capacity) * 100.0
                            return projected_grads, combined_stats

                        def no_agem_projection():
                            empty_stats = {
                                "agem/agem_alpha": jnp.array(0.0),
                                "agem/agem_dot_g": jnp.array(0.0),
                                "agem/agem_final_grad_norm": jnp.array(0.0),
                                "agem/agem_is_proj": jnp.array(False),
                                "agem/agem_mem_grad_norm": jnp.array(0.0),
                                "agem/agem_ppo_grad_norm": jnp.array(0.0),
                                "agem/agem_projected_grad_norm": jnp.array(0.0),
                                "agem/mem_grad_norm_raw": jnp.array(0.0),
                                "agem/memory_fullness_pct": jnp.array(0.0),
                                "agem/ppo_actor_loss": jnp.array(0.0),
                                "agem/ppo_entropy": jnp.array(0.0),
                                "agem/ppo_total_loss": jnp.array(0.0),
                                "agem/ppo_value_loss": jnp.array(0.0),
                            }
                            return grads, empty_stats

                        final_grads, agem_stats = jax.lax.cond(
                            jnp.sum(past_sizes) > 0,
                            lambda: apply_agem_projection(agem_rng, past_sizes),
                            lambda: no_agem_projection(),
                        )
                        train_state = train_state.apply_gradients(grads=final_grads)
                    elif cfg.cl_method.lower() == "er_ace" and cl_state is not None:
                        past_sizes = cl_state.sizes.at[env_idx].set(0)
                        max_tasks = cl_state.obs.shape[0]
                        samples_per_task = max(cfg.agem_sample_size // max_tasks, 1)

                        er_ace_grads = None
                        for t in range(max_tasks):
                            agem_rng, task_rng = jax.random.split(agem_rng)
                            t_obs, t_actions, _, _, _, _ = sample_task_slot(cl_state, t, samples_per_task, task_rng)
                            t_obs = jax.lax.stop_gradient(t_obs)
                            t_actions = jax.lax.stop_gradient(t_actions)

                            def actor_bc_loss(params, obs=t_obs, acts=t_actions, task=t):
                                pi_m, _ = actor_network.apply(params['actor'], obs, env_idx=task)
                                return -jnp.mean(pi_m.log_prob(acts))

                            t_grads = jax.grad(actor_bc_loss)(train_state.params)
                            mask = (past_sizes[t] > 0).astype(jnp.float32)
                            t_grads = jax.tree_util.tree_map(lambda g: g * mask, t_grads)
                            er_ace_grads = (
                                t_grads if er_ace_grads is None
                                else jax.tree_util.tree_map(lambda a, b: a + b, er_ace_grads, t_grads)
                            )

                        has_past = jnp.sum(past_sizes) > 0
                        er_ace_grads = jax.lax.cond(
                            has_past,
                            lambda: er_ace_grads,
                            lambda: jax.tree_util.tree_map(jnp.zeros_like, er_ace_grads),
                        )
                        grads = jax.tree_util.tree_map(
                            lambda g, eg: g + cfg.er_ace_coef * eg, grads, er_ace_grads
                        )
                        train_state = train_state.apply_gradients(grads=grads)
                        agem_stats = {"er_ace/total_past_samples": jnp.sum(past_sizes).astype(jnp.float32)}
                    else:
                        # if using packnet, mask before applying gradient:
                        if cfg.cl_method == "packnet":
                            grads = cl.mask_gradients(cl_state, grads)
                        train_state = train_state.apply_gradients(grads=grads)

                    return (train_state, cl_state, rng), (total_loss, grads, agem_stats)

                train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state

                batch_size = cfg.minibatch_size * cfg.num_minibatches
                assert batch_size == cfg.num_steps * cfg.num_actors

                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, batch_size)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [cfg.num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )

                (train_state, cl_state, rng), loss_information = jax.lax.scan(
                    _update_minbatch,
                    init=(train_state, cl_state, rng),
                    xs=minibatches,
                )

                total_loss, grads, agem_stats = loss_information
                loss_dict = {"total_loss": total_loss}
                if cfg.cl_method.lower() in ("agem", "er_ace"):
                    loss_dict["agem_stats"] = agem_stats

                update_state = (train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state)
                return update_state, loss_dict

            update_state = (train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, xs=None, length=cfg.update_epochs
            )
            train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state

            current_timestep = update_step * cfg.num_steps * cfg.num_envs
            metrics = jax.tree_util.tree_map(lambda x: x.mean(), info)

            # AGEM / ER-ACE memory update
            if cfg.cl_method.lower() in ("agem", "er_ace") and cl_state is not None:
                cl_state, rng = update_agem_memory(
                    cfg.agem_sample_size, env_idx, advantages, cl_state, rng, targets, traj_batch
                )

            update_step += 1

            metrics["General/env_index"] = env_idx
            metrics["General/update_step"] = update_step
            metrics["General/steps_for_env"] = steps_for_env
            metrics["General/env_step"] = update_step * cfg.num_steps * cfg.num_envs
            if cfg.anneal_lr:
                metrics["General/learning_rate"] = linear_schedule(
                    update_step * cfg.num_minibatches * cfg.update_epochs
                )
            else:
                metrics["General/learning_rate"] = cfg.lr
            metrics["General/reward_shaping_anneal"] = rew_shaping_anneal(current_timestep)

            # Losses
            loss_dict = loss_info
            total_loss = loss_dict["total_loss"]
            value_loss, loss_actor, entropy, reg_loss = total_loss[1]
            total_loss = total_loss[0]
            metrics["Losses/total_loss"] = total_loss.mean()
            metrics["Losses/value_loss"] = value_loss.mean()
            metrics["Losses/actor_loss"] = loss_actor.mean()
            metrics["Losses/entropy"] = entropy.mean()
            metrics["Losses/reg_loss"] = reg_loss.mean()

            if "agem_stats" in loss_dict:
                for k, v in loss_dict["agem_stats"].items():
                    if v.size > 0:
                        metrics[k] = v.mean()

            # Soup metrics — true per-episode average using done flags (matches ippo)
            T, E = cfg.num_steps, cfg.num_envs
            A = num_agents
            max_per_episode = max_soup_vals[env_idx]

            soups_tea = jnp.stack([info["soups"][a] for a in agents], axis=-1)
            soups_per_env = soups_tea.sum(axis=(0, 2))

            done_tea = traj_batch.done.reshape(T, E, A)
            done_te = done_tea[..., 0]
            episodes_per_env = done_te.sum(axis=0)

            mask = episodes_per_env > 0
            true_avg = jnp.where(mask, soups_per_env / jnp.maximum(episodes_per_env, 1), 0.0)
            num_finished = jnp.maximum(mask.sum(), 1)
            metrics["Soup/total"] = true_avg.sum() / num_finished
            metrics["Soup/scaled"] = jnp.where(
                max_per_episode > 0,
                (true_avg / max_per_episode).sum() / num_finished,
                0.0,
            )
            for ai, agent in enumerate(agents):
                soups_te = soups_tea[:, :, ai].sum(axis=0)
                per_agent = jnp.where(mask, soups_te / jnp.maximum(episodes_per_env, 1), 0.0)
                metrics[f"Soup/{agent}"] = per_agent.sum() / num_finished

            metrics.pop('soups', None)

            # Rewards
            for agent in agents:
                metrics[f"General/shaped_reward_{agent}"] = metrics["shaped_reward"][agent]
                metrics[f"General/shaped_reward_annealed_{agent}"] = (
                        metrics[f"General/shaped_reward_{agent}"] * rew_shaping_anneal(current_timestep)
                )
            metrics.pop('shaped_reward', None)

            metrics["Advantage_Targets/advantages"] = advantages.mean()
            metrics["Advantage_Targets/targets"] = targets.mean()

            def evaluate_and_log(rng, update_step):
                rng, eval_rng = jax.random.split(rng)

                def log_metrics(metrics, update_step):
                    if cfg.evaluation:
                        avg_rewards, avg_soups, avg_het = evaluate_all_envs(
                            cl_state, eval_rng, train_state.params, seq_length, evaluate_env
                        )
                        metrics = add_eval_metrics(avg_rewards, avg_soups, env_names, max_soup_vals, metrics)
                        metrics = add_het_metrics(avg_het, env_names, metrics)

                    def callback(args):
                        metrics, update_step, env_counter = args
                        real_step = (env_counter - 1) * cfg.num_updates + update_step
                        for key, value in metrics.items():
                            writer.add_scalar(key, value, real_step)

                    jax.experimental.io_callback(callback, None, (metrics, update_step, env_idx + 1))
                    return None

                def do_not_log(metrics, update_step):
                    return None

                jax.lax.cond(
                    (update_step % cfg.log_interval) == 0,
                    log_metrics, do_not_log, metrics, update_step,
                )

            evaluate_and_log(rng=rng, update_step=update_step)

            runner_state = (train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state)
            return runner_state, metrics

        rng, train_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, 0, train_rng, cl_state)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, xs=None, length=cfg.num_updates
        )

        if cfg.cl_method.lower() == "packnet":
            # Unpack runner state
            train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state

            # Prune the model and update the parameters
            train_state, cl_state = cl.on_train_end(train_state, cl_state)

            # create new runner state for fine-tuning:
            rng, finetune_rng = jax.random.split(rng)
            runner_state = (train_state, env_state, last_obs, update_step, steps_for_env, finetune_rng, cl_state)

            # run fine-tuning
            runner_state, metrics = jax.lax.scan(
                f=_update_step,
                init=runner_state,
                xs=None,
                length=cfg.finetune_updates
            )

            train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state
            # handle the end of the finetune phase
            train_state, cl_state = cl.on_finetune_end(train_state, cl_state)

            # add cl_state (packnet_state in this case) to new runner state
            runner_state = (train_state, env_state, last_obs, update_step, steps_for_env, finetune_rng, cl_state)

        return runner_state, metrics

    # ─────────────────────────────────────────────────────────────────────────
    def loop_over_envs(rng, train_state, cl_state, envs):
        rng, *env_rngs = jax.random.split(rng, seq_length + 1)

        visualizer = None
        for task_idx, (task_rng, env) in enumerate(zip(env_rngs, envs)):
            print(f"Training on environment: {task_idx} - {env.layout_name}")
            runner_state, metrics = train_on_environment(task_rng, train_state, cl_state, task_idx)
            train_state = runner_state[0]
            cl_state = runner_state[6]

            # Continual Learning importance update
            importance = importance_fn(train_state.params, task_idx, task_rng)
            cl_state = cl.update_state(cl_state, train_state.params, importance)

            # Video recording
            if cfg.record_video:
                if visualizer is None:
                    visualizer = create_visualizer(num_agents, cfg.env_name, cfg.renderer_version)
                env_name = env.layout_name
                start_time = time.time()
                states = rollout_for_video(task_rng, cfg, train_state, env, network, task_idx, cfg.video_length)
                print(f"Rollout for video took {time.time() - start_time:.2f} seconds.")
                start_time = time.time()
                file_path = f"{exp_dir}/task_{task_idx}_{env_name}.mp4"
                visualizer.animate(states, out_path=file_path, task_idx=task_idx, env=env)
                print(f"Animating video took {time.time() - start_time:.2f} seconds.")

            # Save checkpoint
            repo_root = Path(__file__).resolve().parent.parent
            path = (f"{repo_root}/checkpoints/overcooked/{cfg.cl_method}/"
                    f"{run_name}/model_env_{task_idx + 1}")
            save_params(path, train_state, env_kwargs=env.layout,
                        layout_name=env.layout_name, config=cfg)

            if cfg.single_task_idx is not None:
                break

    def save_params(path, train_state, env_kwargs=None, layout_name=None, config=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            f.write(flax.serialization.to_bytes({"params": train_state.params}))

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
                config_dict = {
                    "use_cnn": config.use_cnn,
                    "num_tasks": seq_length,
                    "use_multihead": config.use_multihead,
                    "shared_backbone": config.shared_backbone,
                    "big_network": config.big_network,
                    "use_task_id": config.use_task_id,
                    "use_agent_id": config.use_agent_id,
                    "regularize_heads": config.regularize_heads,
                    "use_layer_norm": config.use_layer_norm,
                    "activation": config.activation,
                    "strategy": config.strategy,
                    "seed": config.seed,
                }
                config_data.update(convert_frozen_dict(config_dict))

            config_data = convert_frozen_dict(config_data)
            config_path = f"{path}_config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

        print('model saved to', path)

    # ── Run ───────────────────────────────────────────────────────────────────
    rng, train_rng = jax.random.split(rng)
    cl_state = init_cl_state(train_state.params, cfg.regularize_critic, cfg.regularize_heads, cl, cfg)

    if cfg.cl_method.lower() in ("agem", "er_ace"):
        obs_dim_agem = temp_env.observation_space().shape
        if not cfg.use_cnn:
            obs_dim_agem = (int(np.prod(obs_dim_agem)),)
        cl_state = init_agem_memory(cfg.agem_memory_size, obs_dim_agem, max_tasks=seq_length)

    loop_over_envs(train_rng, train_state, cl_state, envs)


if __name__ == "__main__":
    print("Running main...")
    main()
