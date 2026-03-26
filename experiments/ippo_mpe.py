"""
IPPO training script for the MPE SimpleSpread continual learning benchmark.

Usage:
    conda run -n meal python experiments/ippo_mpe.py \\
        --cl_method ft --seq_length 10 --num_agents 3 --num_landmarks 3 \\
        --strategy random --seed 42

Task diversity is driven by ``local_ratio`` variation across tasks:
    0.0  → pure global reward (team must cover all landmarks together)
    1.0  → pure local reward  (each agent only penalised for its own collision)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Sequence

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
from tensorboardX import SummaryWriter

from experiments.continual.agem import (
    AGEM,
    agem_project,
    compute_memory_gradient,
    init_agem_memory,
    sample_memory,
    update_agem_memory,
)
from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.l2 import L2
from experiments.continual.mas import MAS
from experiments.evaluation_mpe import evaluate_all_envs, make_eval_fn
from experiments.model.cnn import ActorCritic as CNNActorCritic
from experiments.model.mlp import ActorCritic as MLPActorCritic
from experiments.utils import *
from meal.env.mpe import MPESpreadEnv, make_mpe_sequence
from meal.wrappers.logging import LogWrapper


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING / PPO PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════
    alg_name: Literal["ippo"] = "ippo"
    lr: float = 3e-4
    anneal_lr: bool = False
    num_envs: int = 512
    num_steps: int = 100        # episode length (must match max_steps)
    steps_per_task: float = 5e7
    update_epochs: int = 4
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ═══════════════════════════════════════════════════════════════════════
    # NETWORK ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════════
    activation: str = "relu"
    use_cnn: bool = False
    use_layer_norm: bool = True
    big_network: bool = False

    # ═══════════════════════════════════════════════════════════════════════
    # CONTINUAL LEARNING
    # ═══════════════════════════════════════════════════════════════════════
    cl_method: Optional[str] = None
    reg_coef: Optional[float] = None
    use_task_id: bool = True
    use_multihead: bool = True
    shared_backbone: bool = False
    normalize_importance: bool = False
    regularize_critic: bool = False
    regularize_heads: bool = False
    reset_optimizer: bool = True

    importance_episodes: int = 5
    importance_stride: int = 5
    importance_steps: int = 500
    importance_mode: str = "online"
    importance_decay: float = 0.9

    agem_memory_size: int = 100000
    agem_sample_size: int = 1024
    agem_gradient_scale: float = 1.0

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT
    # ═══════════════════════════════════════════════════════════════════════
    num_agents: int = 3
    num_landmarks: int = 3
    num_obstacles: int = 4
    max_steps: int = 100         # episode length; keep in sync with num_steps
    local_ratio: float = 0.5     # 0=fully global reward, 1=fully local
    seq_length: int = 10
    single_task_idx: Optional[int] = None

    # ═══════════════════════════════════════════════════════════════════════
    # EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    evaluation: bool = True
    record_video: bool = False
    video_length: int = 100
    log_interval: int = 5

    # ═══════════════════════════════════════════════════════════════════════
    # LOGGING
    # ═══════════════════════════════════════════════════════════════════════
    use_wandb: bool = True
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    entity: Optional[str] = ""
    project: str = "MEAL"
    tags: List[str] = field(default_factory=list)

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT
    # ═══════════════════════════════════════════════════════════════════════
    seed: int = 42
    num_seeds: int = 1

    # ═══════════════════════════════════════════════════════════════════════
    # RUNTIME COMPUTED (set in main)
    # ═══════════════════════════════════════════════════════════════════════
    num_actors: int = 0
    num_updates: int = 0
    minibatch_size: int = 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    jax.config.update("jax_platform_name", "gpu")
    print("Device:", jax.devices())

    cfg = tyro.cli(Config)

    if cfg.single_task_idx is not None:
        cfg.cl_method = "ft"
    if cfg.cl_method is None:
        raise ValueError("cl_method is required. Choose: ewc, mas, l2, ft, agem")

    if cfg.reg_coef is None:
        defaults = {"ewc": 1e11, "mas": 1e9, "l2": 1e7}
        cfg.reg_coef = defaults.get(cfg.cl_method.lower())

    seq_length = cfg.seq_length
    seed = cfg.seed

    method_map = dict(
        ewc=EWC(mode=cfg.importance_mode, decay=cfg.importance_decay),
        mas=MAS(mode=cfg.importance_mode, decay=cfg.importance_decay),
        l2=L2(),
        ft=FT(),
        agem=AGEM(memory_size=cfg.agem_memory_size, sample_size=cfg.agem_sample_size),
    )
    cl = method_map[cfg.cl_method.lower()]

    envs = make_mpe_sequence(
        sequence_length=seq_length,
        seed=seed,
        num_agents=cfg.num_agents,
        num_landmarks=cfg.num_landmarks,
        num_obstacles=cfg.num_obstacles,
        max_steps=cfg.max_steps,
        local_ratio=cfg.local_ratio,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    network_tag = "cnn" if cfg.use_cnn else "mlp"
    run_name = (
        f"ippo_mpe_{cfg.cl_method}_{cfg.num_agents}a_{cfg.num_landmarks}l_{cfg.num_obstacles}k"
        f"_{network_tag}_seq{seq_length}_seed{seed}_{timestamp}"
    )
    exp_dir = os.path.join("runs", run_name)

    if cfg.use_wandb:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(
            project=cfg.project,
            config=asdict(cfg),
            sync_tensorboard=True,
            mode=cfg.wandb_mode,
            tags=cfg.tags or [],
            group=cfg.cl_method.upper(),
            name=run_name,
            id=run_name,
        )

    writer = SummaryWriter(exp_dir)
    rows = [f"|{k}|{str(v).replace(chr(10), '<br>')}|" for k, v in vars(cfg).items()]
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(rows))

    # Wrap with LogWrapper
    env_names = []
    for i, env in enumerate(envs):
        envs[i] = LogWrapper(env, replace_info=False)
        env_names.append(envs[i].map_id)

    temp_env = envs[0]
    num_agents = temp_env.num_agents
    agents = temp_env.agents

    cfg.num_actors = num_agents * cfg.num_envs
    cfg.num_updates = int(cfg.steps_per_task // cfg.num_steps // cfg.num_envs)
    cfg.minibatch_size = (cfg.num_actors * cfg.num_steps) // cfg.num_minibatches

    def linear_schedule(count):
        frac = 1.0 - (count // (cfg.num_minibatches * cfg.update_epochs)) / cfg.num_updates
        return cfg.lr * frac

    ac_cls = CNNActorCritic if cfg.use_cnn else MLPActorCritic
    network = ac_cls(
        temp_env.action_space().n,
        cfg.activation,
        seq_length,
        cfg.use_multihead,
        cfg.shared_backbone,
        cfg.big_network,
        cfg.use_task_id,
        cfg.regularize_heads,
        cfg.use_layer_norm,
    )

    rng = jax.random.PRNGKey(cfg.seed)
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, cfg.num_envs)
    temp_obs, _ = jax.vmap(temp_env.reset, in_axes=(0,))(reset_rngs)
    temp_obs_batch = batchify(temp_obs, agents, cfg.num_actors, not cfg.use_cnn)
    obs_dim = temp_obs_batch.shape[1]

    rng, network_rng = jax.random.split(rng)
    init_x = jnp.zeros((1, obs_dim))
    network_params = network.init(network_rng, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(
            learning_rate=linear_schedule if cfg.anneal_lr else cfg.lr,
            eps=1e-5,
        ),
    )
    network.apply = jax.jit(network.apply)
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    reset_fns = tuple(env.reset for env in envs)
    step_fns = tuple(env.step for env in envs)

    def reset_switch(key, task_idx):
        return jax.lax.switch(task_idx, reset_fns, key)

    def step_switch(key, state, actions, task_idx):
        return jax.lax.switch(task_idx, step_fns, key, state, actions)

    evaluate_env = make_eval_fn(
        reset_switch, step_switch, network, agents,
        seq_length, cfg.num_steps, cfg.use_cnn,
    )
    importance_fn = cl.make_importance_fn(
        reset_switch, step_switch, network, agents, cfg.use_cnn,
        cfg.importance_episodes, cfg.importance_steps,
        cfg.normalize_importance, cfg.importance_stride,
    )

    # ------------------------------------------------------------------
    # Per-task training (mirrors ippo_jaxnav.py structure exactly)
    # ------------------------------------------------------------------

    @jax.jit
    def train_on_environment(rng, train_state, cl_state, env_idx):
        if cfg.reset_optimizer:
            new_opt_state = train_state.tx.init(train_state.params)
            train_state = train_state.replace(tx=tx, opt_state=new_opt_state)

        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, cfg.num_envs)
        obsv, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(reset_rng)

        def _update_step(runner_state, _):

            def _env_step(runner_state, _):
                train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, agents, cfg.num_actors, not cfg.use_cnn)
                pi, value, _ = network.apply(train_state.params, obs_batch, env_idx=env_idx)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                env_act = unbatchify(action, agents, cfg.num_envs, num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, cfg.num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    lambda k, s, a: step_switch(k, s, a, jnp.int32(env_idx))
                )(rng_step, env_state, env_act)

                transition = Transition(
                    batchify(done, agents, cfg.num_actors, not cfg.use_cnn).squeeze(),
                    action,
                    value,
                    batchify(reward, agents, cfg.num_actors).squeeze(),
                    log_prob,
                    obs_batch,
                )
                steps_for_env = steps_for_env + cfg.num_envs
                runner_state = (train_state, env_state, obsv, update_step, steps_for_env, rng, cl_state)
                return runner_state, (transition, info)

            runner_state, (traj_batch, info) = jax.lax.scan(
                _env_step, runner_state, None, length=cfg.num_steps
            )
            train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state

            last_obs_batch = batchify(last_obs, agents, cfg.num_actors, not cfg.use_cnn)
            _, last_val, _ = network.apply(train_state.params, last_obs_batch, env_idx=env_idx)

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

            def _update_epoch(update_state, _):
                def _update_minbatch(carry, batch_info):
                    train_state, cl_state = carry
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value, _ = network.apply(params, traj_batch.obs, env_idx=env_idx)
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -cfg.clip_eps, cfg.clip_eps
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor_unclipped = ratio * gae
                        loss_actor_clipped = (
                            jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor_unclipped, loss_actor_clipped).mean()
                        entropy = pi.entropy().mean()

                        cl_penalty = cl.penalty(params, cl_state, cfg.reg_coef)
                        total_loss = (
                            loss_actor
                            + cfg.vf_coef * value_loss
                            - cfg.ent_coef * entropy
                            + cl_penalty
                        )
                        return total_loss, (value_loss, loss_actor, entropy, cl_penalty)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)

                    agem_stats = {}

                    def apply_agem_projection():
                        rng_1, sample_rng = jax.random.split(rng)
                        mem_obs, mem_actions, mem_log_probs, mem_advs, mem_targets, mem_values = sample_memory(
                            cl_state, cfg.agem_sample_size, sample_rng
                        )
                        grads_mem, grads_stats = compute_memory_gradient(
                            network, train_state.params,
                            cfg.clip_eps, cfg.vf_coef, cfg.ent_coef,
                            mem_obs, mem_actions, mem_advs, mem_log_probs,
                            mem_targets, mem_values,
                            env_idx=env_idx,
                        )
                        g_ppo, _ = ravel_pytree(grads)
                        g_mem, _ = ravel_pytree(grads_mem)
                        norm_ppo = jnp.linalg.norm(g_ppo) + 1e-12
                        norm_mem = jnp.linalg.norm(g_mem) + 1e-12
                        scale = norm_ppo / norm_mem * cfg.agem_gradient_scale
                        grads_mem_scaled = jax.tree_util.tree_map(lambda g: g * scale, grads_mem)
                        projected_grads, proj_stats = agem_project(grads, grads_mem_scaled)
                        combined_stats = {**grads_stats, **proj_stats}
                        combined_stats["agem/mem_grad_norm_scaled"] = jnp.linalg.norm(
                            ravel_pytree(grads_mem_scaled)[0]
                        )
                        total_used = jnp.sum(cl_state.sizes)
                        total_capacity = cl_state.max_tasks * cl_state.max_size_per_task
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
                            "agem/mem_grad_norm_scaled": jnp.array(0.0),
                            "agem/memory_fullness_pct": jnp.array(0.0),
                            "agem/ppo_actor_loss": jnp.array(0.0),
                            "agem/ppo_entropy": jnp.array(0.0),
                            "agem/ppo_total_loss": jnp.array(0.0),
                            "agem/ppo_value_loss": jnp.array(0.0),
                        }
                        return grads, empty_stats

                    if cfg.cl_method.lower() == "agem" and cl_state is not None:
                        grads, agem_stats = jax.lax.cond(
                            jnp.sum(cl_state.sizes) > 0,
                            lambda: apply_agem_projection(),
                            lambda: no_agem_projection(),
                        )

                    loss_information = total_loss, grads, agem_stats
                    train_state = train_state.apply_gradients(grads=grads)
                    return (train_state, cl_state), loss_information

                train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state

                batch_size = cfg.minibatch_size * cfg.num_minibatches
                assert batch_size == cfg.num_steps * cfg.num_actors
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, batch_size)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [cfg.num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )

                (train_state, cl_state), loss_information = jax.lax.scan(
                    _update_minbatch, (train_state, cl_state), minibatches
                )

                total_loss, grads, agem_stats = loss_information
                loss_dict = {"total_loss": total_loss}
                if cfg.cl_method.lower() == "agem":
                    loss_dict["agem_stats"] = agem_stats

                update_state = (train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state)
                return update_state, loss_dict

            update_state = (train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, length=cfg.update_epochs
            )
            train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state

            current_timestep = update_step * cfg.num_steps * cfg.num_envs
            metrics = jax.tree_util.tree_map(lambda x: x.mean(), info)

            if cfg.cl_method.lower() == "agem" and cl_state is not None:
                cl_state, rng = update_agem_memory(
                    cfg.agem_sample_size, env_idx, advantages, cl_state, rng, targets, traj_batch
                )

            update_step += 1
            metrics["General/env_index"] = env_idx
            metrics["General/update_step"] = update_step
            metrics["General/steps_for_env"] = steps_for_env
            metrics["General/env_step"] = update_step * cfg.num_steps * cfg.num_envs
            metrics["General/learning_rate"] = cfg.lr

            loss_dict = loss_info
            total_loss = loss_dict["total_loss"]
            value_loss, loss_actor, entropy, reg_loss = total_loss[1]
            total_loss_scalar = total_loss[0]
            metrics["Losses/total_loss"] = total_loss_scalar.mean()
            metrics["Losses/value_loss"] = value_loss.mean()
            metrics["Losses/actor_loss"] = loss_actor.mean()
            metrics["Losses/entropy"] = entropy.mean()
            metrics["Losses/reg_loss"] = reg_loss.mean()

            if "agem_stats" in loss_dict:
                for k, v in loss_dict["agem_stats"].items():
                    if v.size > 0:
                        metrics[k] = v.mean()

            metrics["Advantage_Targets/advantages"] = advantages.mean()
            metrics["Advantage_Targets/targets"] = targets.mean()

            obs_batch = batchify(last_obs, agents, cfg.num_actors, not cfg.use_cnn)
            _, _, current_dormant_ratio = network.apply(train_state.params, obs_batch, env_idx=env_idx)
            metrics["Neural_Activity/dormant_ratio"] = current_dormant_ratio

            metrics.pop("terminated", None)

            def evaluate_and_log(rng, update_step):
                rng, eval_rng = jax.random.split(rng)

                def log_metrics(metrics, update_step):
                    if cfg.evaluation:
                        avg_rewards, avg_coverage, avg_covered = evaluate_all_envs(
                            eval_rng, train_state.params, seq_length, evaluate_env
                        )
                        for i, env_name in enumerate(env_names):
                            metrics[f"Evaluation/Returns/{i}_{env_name}"] = avg_rewards[i]
                            metrics[f"Evaluation/Coverage/{i}_{env_name}"] = avg_coverage[i]
                            metrics[f"Evaluation/NumCovered/{i}_{env_name}"] = avg_covered[i]

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
                    log_metrics,
                    do_not_log,
                    metrics,
                    update_step,
                )

            evaluate_and_log(rng=rng, update_step=update_step)

            runner_state = (train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state)
            return runner_state, metrics

        rng, train_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, 0, train_rng, cl_state)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=cfg.num_updates
        )
        return runner_state, metrics

    # ------------------------------------------------------------------
    # CL outer loop
    # ------------------------------------------------------------------

    def save_params(path, train_state, layout_name=None, config=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(flax.serialization.to_bytes({"params": train_state.params}))
        if layout_name is not None or config is not None:
            config_data = {"layout_name": layout_name}
            if config is not None:
                config_data.update({
                    "use_cnn": config.use_cnn,
                    "num_tasks": seq_length,
                    "use_multihead": config.use_multihead,
                    "seed": config.seed,
                    "strategy": config.strategy,
                })
            with open(f"{path}_config.json", "w") as f:
                json.dump(config_data, f, indent=2)
        print("Model saved to", path)

    def loop_over_envs(rng, train_state, cl_state, envs):
        rng, *env_rngs = jax.random.split(rng, seq_length + 1)
        for task_idx, (task_rng, env) in enumerate(zip(env_rngs, envs)):
            print(f"Training on task {task_idx}: {env.map_id}")
            runner_state, _ = train_on_environment(task_rng, train_state, cl_state, task_idx)
            train_state = runner_state[0]
            cl_state = runner_state[6]

            importance = importance_fn(train_state.params, task_idx, task_rng)
            cl_state = cl.update_state(cl_state, train_state.params, importance)

            repo_root = Path(__file__).resolve().parent.parent
            path = (
                f"{repo_root}/checkpoints/mpe/{cfg.cl_method}"
                f"/{run_name}/model_env_{task_idx + 1}"
            )
            save_params(path, train_state, layout_name=env.map_id, config=cfg)

            if cfg.single_task_idx is not None:
                break

    rng, train_rng = jax.random.split(rng)
    cl_state = init_cl_state(train_state.params, cfg.regularize_critic, cfg.regularize_heads)

    if cfg.cl_method.lower() == "agem":
        obs_shape = temp_env.observation_space().shape
        if not cfg.use_cnn:
            obs_shape = (int(np.prod(obs_shape)),)
        cl_state = init_agem_memory(cfg.agem_memory_size, obs_shape, max_tasks=seq_length)

    loop_over_envs(train_rng, train_state, cl_state, envs)


if __name__ == "__main__":
    print("Running ippo_mpe...")
    main()
