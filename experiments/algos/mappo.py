import json
import os
from dataclasses import dataclass
from typing import Literal

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from experiments.algo_common import convert_frozen_dict, build_reset_step_switch
from experiments.algos.on_policy import OnPolicyAlgo, OnPolicyConfig
from experiments.continual.agem import AGEM, init_agem_memory, update_agem_memory
from experiments.continual.base import RegCLMethod
from experiments.continual.er_ace import ERACE
from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.l2 import L2
from experiments.continual.mas import MAS
from experiments.continual.packnet import Packnet
from experiments.model.decoupled_mlp import Actor, Critic
from experiments.utils import Transition_MAPPO, batchify, init_cl_state, unbatchify


@dataclass
class MAPPOConfig(OnPolicyConfig):
    alg_name: Literal["ippo", "mappo"] = "mappo"
    use_agent_id: bool = True  # MAPPO-specific: append agent one-hot to actor obs
    log_interval: int = 75
    big_network = False
    shared_backbone: bool = False
    regularize_critic: bool = False


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


class MAPPO(OnPolicyAlgo):
    def build_cl_method(self):
        cfg = self.cfg
        method_map = dict(
            ewc=EWC(mode=cfg.importance_mode, decay=cfg.importance_decay),
            mas=MAS(mode=cfg.importance_mode, decay=cfg.importance_decay),
            l2=L2(),
            ft=FT(),
            agem=AGEM(memory_size=cfg.agem_memory_size, sample_size=cfg.agem_sample_size),
            er_ace=ERACE(memory_size=cfg.agem_memory_size, sample_size=cfg.agem_sample_size),
            packnet=Packnet(seq_length=cfg.seq_length, prune_instructions=0.4,
                            train_finetune_split=(cfg.train_epochs, cfg.finetune_epochs),
                            prunable_layers=[nn.Dense, nn.Conv]),
        )
        return method_map[cfg.cl_method.lower()]

    def linear_schedule(self, count):
        cfg = self.cfg
        frac = 1.0 - (count // (cfg.num_minibatches * cfg.update_epochs)) / cfg.num_updates
        return cfg.lr * frac

    def init_network(self):
        cfg = self.cfg
        num_agents = self.num_agents
        temp_env = self.temp_env

        cfg.num_actors = num_agents * cfg.num_envs
        cfg.num_updates = int(cfg.steps_per_task // cfg.num_steps // cfg.num_envs)
        cfg.finetune_updates = cfg.finetune_timesteps // cfg.num_steps // cfg.num_envs
        cfg.minibatch_size = (cfg.num_actors * cfg.num_steps) // cfg.num_minibatches

        self.reset_switch, self.step_switch = build_reset_step_switch(self.envs)

        obs_dim = self.env_adapter.observation_shape(temp_env, self.agents)
        if not cfg.use_cnn:
            local_obs_dim = int(np.prod(obs_dim))
            global_obs_dim = local_obs_dim * num_agents
        else:
            local_obs_dim = obs_dim
            global_obs_dim = (obs_dim[0], obs_dim[1], obs_dim[2] * num_agents)

        if cfg.cl_method == "packnet" and cfg.use_cnn:
            raise ValueError("Packnet currently doesn't support CNN.")

        actor_network = Actor(
            action_dim=temp_env.action_space(self.agents[0]).n,
            activation=cfg.activation,
            num_tasks=cfg.seq_length,
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
            num_tasks=cfg.seq_length,
            use_multihead=cfg.use_multihead,
            use_task_id=cfg.use_task_id,
            use_cnn=cfg.use_cnn,
            use_layer_norm=cfg.use_layer_norm,
        )

        rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

        if cfg.use_cnn:
            actor_init_x = jnp.zeros((1, *local_obs_dim))
            critic_init_x = jnp.zeros((1, *global_obs_dim))
        else:
            actor_init_x = jnp.zeros((1, local_obs_dim))
            critic_init_x = jnp.zeros((1, global_obs_dim))

        actor_params = actor_network.init(actor_rng, actor_init_x, env_idx=0)
        critic_params = critic_network.init(critic_rng, critic_init_x, env_idx=0)
        network_params = {'actor': actor_params, 'critic': critic_params}

        action_dim = temp_env.action_space(self.agents[0]).n

        # Wrapper used by make_eval_fn / make_importance_fn / compute_memory_gradient /
        # compute_er_ace_gradient (all expect a `.apply(params, obs, *, env_idx) -> (pi,
        # value, dormant)` 3-tuple).
        class DecoupledNetworkWrapper:
            def __init__(self):
                self.action_dim = action_dim

            def apply(self, params, obs, *, env_idx=0):
                pi, dormant_ratio = actor_network.apply(params['actor'], obs, env_idx=env_idx)
                dummy_value = jnp.zeros(obs.shape[0])
                return pi, dummy_value, dormant_ratio

        network = DecoupledNetworkWrapper()

        tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=self.linear_schedule if cfg.anneal_lr else cfg.lr, eps=1e-5),
        )

        train_state = TrainState.create(
            apply_fn=lambda params, obs, **kw: None,  # not used; actor/critic called directly
            params=network_params,
            tx=tx,
        )

        self.rng = rng
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.network = network
        self.tx = tx
        self.train_state = train_state

        self.evaluate_env = self.env_adapter.make_eval_fn(
            self.cl, self.reset_switch, self.step_switch, network, self.agents, cfg.seq_length,
            cfg.num_steps, cfg.use_cnn, cfg.eval_deterministic, cfg.seed
        )
        self.importance_fn = self.cl.make_importance_fn(
            self.reset_switch, self.step_switch, network, self.agents, cfg.use_cnn,
            cfg.importance_episodes, cfg.importance_steps, cfg.normalize_importance,
            cfg.importance_stride,
        )

    def build_train_on_environment(self):
        cfg = self.cfg
        actor_network = self.actor_network
        critic_network = self.critic_network
        network = self.network
        tx = self.tx
        cl = self.cl
        agents = self.agents
        num_agents = self.num_agents
        reset_switch, step_switch = self.reset_switch, self.step_switch

        @jax.jit
        def train_on_environment(rng, train_state, cl_state, env_idx):
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
                def _env_step(runner_state, _):
                    train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state
                    rng, _rng = jax.random.split(rng)

                    obs_batch = batchify(last_obs, agents, cfg.num_actors, not cfg.use_cnn)
                    global_state = create_global_state_for_critic(last_obs, agents, cfg.num_envs, cfg.use_cnn)

                    pi, _actor_dormant = actor_network.apply(
                        train_state.params['actor'], obs_batch, env_idx=env_idx
                    )
                    value_per_env, _critic_dormant = critic_network.apply(
                        train_state.params['critic'], global_state, env_idx=env_idx
                    )
                    value = jnp.tile(value_per_env, len(agents))
                    global_state_batch = jnp.tile(global_state, (len(agents),) + (1,) * (global_state.ndim - 1))

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

                    reward = self.env_adapter.compute_reward(
                        reward, info, agents, cfg, current_timestep, rew_shaping_anneal
                    )

                    transition = Transition_MAPPO(
                        batchify(done, agents, cfg.num_actors, not cfg.use_cnn).squeeze(),
                        action, value,
                        batchify(reward, agents, cfg.num_actors).squeeze(),
                        log_prob, obs_batch, global_state_batch,
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

                advantages, targets = self.calculate_gae(traj_batch, last_val)

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

                            cl_penalty = cl.penalty(params, cl_state, cfg.reg_coef) if isinstance(cl,
                                                                                                  RegCLMethod) else 0.0
                            total_loss = (loss_actor + cfg.vf_coef * value_loss
                                          - cfg.ent_coef * entropy + cl_penalty)
                            return total_loss, (value_loss, loss_actor, entropy, cl_penalty)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)

                        train_state, agem_rng, agem_stats = self.apply_cl_gradient_step(
                            network, 0.0, cl_state, env_idx, train_state, grads, agem_rng,
                            is_packnet=(cfg.cl_method == "packnet"),
                        )

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
                        _update_minbatch, init=(train_state, cl_state, rng), xs=minibatches,
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
                    metrics["General/learning_rate"] = self.linear_schedule(
                        update_step * cfg.num_minibatches * cfg.update_epochs
                    )
                else:
                    metrics["General/learning_rate"] = cfg.lr
                metrics["General/reward_shaping_anneal"] = rew_shaping_anneal(current_timestep)

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

                metrics = self.compute_env_step_metrics(
                    metrics, info, traj_batch, env_idx, current_timestep, rew_shaping_anneal
                )

                metrics["Advantage_Targets/advantages"] = advantages.mean()
                metrics["Advantage_Targets/targets"] = targets.mean()

                evaluate_and_log = self.build_evaluate_and_log(
                    cl_state=cl_state, evaluate_env=self.evaluate_env,
                    get_params=lambda: train_state.params,
                )
                evaluate_and_log(rng, update_step, metrics, env_idx)

                runner_state = (train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state)
                return runner_state, metrics

            rng, train_rng = jax.random.split(rng)
            runner_state = (train_state, env_state, obsv, 0, 0, train_rng, cl_state)
            runner_state, metrics = self.run_train_then_finetune(_update_step, runner_state)
            return runner_state, metrics

        return train_on_environment

    def save_params(self, path, train_state, env_kwargs=None, layout_name=None, config=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(flax.serialization.to_bytes({"params": train_state.params}))

        if env_kwargs is not None or layout_name is not None or config is not None:
            env_kwargs = convert_frozen_dict(env_kwargs)
            config_data = {"env_kwargs": env_kwargs, "layout_name": layout_name}
            if config is not None:
                config_dict = {
                    "use_cnn": config.use_cnn,
                    "num_tasks": self.cfg.seq_length,
                    "use_multihead": config.use_multihead,
                    "shared_backbone": config.shared_backbone,
                    "big_network": config.big_network,
                    "use_task_id": config.use_task_id,
                    "use_agent_id": config.use_agent_id,
                    "use_layer_norm": config.use_layer_norm,
                    "activation": config.activation,
                    "strategy": getattr(config.env, "strategy", None),
                    "seed": config.seed,
                }
                config_data.update(convert_frozen_dict(config_dict))
            config_data = convert_frozen_dict(config_data)
            with open(f"{path}_config.json", "w") as f:
                json.dump(config_data, f, indent=2)
        print('model saved to', path)

    def run(self):
        self.setup_envs()
        self.init_network()
        self.setup_logging()

        cl_state = init_cl_state(self.train_state.params, self.cfg.regularize_critic,
                                 not self.cfg.use_multihead, self.cl, self.cfg)
        if self.cfg.cl_method.lower() in ("agem", "er_ace"):
            obs_dim_agem = self.env_adapter.observation_shape(self.temp_env, self.agents)
            if not self.cfg.use_cnn:
                obs_dim_agem = (int(np.prod(obs_dim_agem)),)
            cl_state = init_agem_memory(self.cfg.agem_memory_size, obs_dim_agem, max_tasks=self.cfg.seq_length)

        train_on_environment = self.build_train_on_environment()

        rng, train_rng = jax.random.split(self.rng)
        self.loop_over_envs(
            train_rng, self.train_state, cl_state, train_on_environment,
            self.save_params, self.importance_fn, self.network,
        )
