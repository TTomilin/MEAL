"""HAPPO on top of OnPolicyAlgo.

HAPPO overrides more of the OnPolicyAlgo template than IPPO/MAPPO do: it carries two
TrainStates (actor_ts, critic_ts) instead of one, so `loop_over_envs` and `save_params`
are HAPPO-specific rather than the shared OnPolicyAlgo versions. Its per-agent AGEM/
ER-ACE/packnet gradient application goes through the same `OnPolicyAlgo.apply_cl_gradient_step`
IPPO/MAPPO use.
"""
import json
import os
from dataclasses import dataclass
from typing import Literal, NamedTuple

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
from experiments.utils import batchify, init_cl_state, unbatchify


class Transition_HAPPO(NamedTuple):
    """Per-step transition for HAPPO. Fields with (num_actors,) shape cover all agents
    in all envs, batchified in the same order as IPPO. global_state has shape
    (num_envs, global_dim) -- one row per *environment*, not per actor."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    global_state: jnp.ndarray


@dataclass
class HAPPOConfig(OnPolicyConfig):
    alg_name: Literal["happo"] = "happo"
    use_agent_id: bool = False  # Optionally condition actor on agent index


def create_global_state_for_critic(obs_dict, agent_list, num_envs: int, use_cnn: bool = False):
    """Concatenate all agents' observations into a single global state for the
    centralized critic. MLP: (num_envs, sum_of_flat_obs_dims). CNN: (num_envs, H, W,
    num_agents * C)."""
    if use_cnn:
        agent_obs = [obs_dict[a] for a in agent_list]
        return jnp.concatenate(agent_obs, axis=-1)
    else:
        agent_obs = [obs_dict[a].reshape(num_envs, -1) for a in agent_list]
        return jnp.concatenate(agent_obs, axis=-1)


class HAPPOActorWrapper:
    """Wraps the Actor network to expose a 3-value apply signature
    (pi, value_placeholder, dormant_ratio) required by make_eval_fn,
    compute_memory_gradient, compute_er_ace_gradient, and CL importance functions."""

    def __init__(self, actor_net: Actor):
        self._net = actor_net
        self.action_dim = actor_net.action_dim

    def apply(self, params, obs, *, env_idx=0):
        pi, dormant = self._net.apply(params, obs, env_idx=env_idx)
        value_placeholder = jnp.zeros(obs.shape[0])
        return pi, value_placeholder, dormant


class HAPPO(OnPolicyAlgo):
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
        # HAPPO updates the actor num_agents times per minibatch (one per agent).
        cfg = self.cfg
        frac = 1.0 - (count // (cfg.num_minibatches * cfg.update_epochs * self.num_agents)) / cfg.num_updates
        return cfg.lr * frac

    def init_network(self):
        cfg = self.cfg
        num_agents = self.num_agents
        temp_env = self.temp_env

        cfg.num_actors = num_agents * cfg.num_envs
        cfg.num_updates = int(cfg.steps_per_task // cfg.num_steps // cfg.num_envs)
        cfg.finetune_updates = cfg.finetune_timesteps // cfg.num_steps // cfg.num_envs
        # Per-agent minibatch size (total / num_agents / num_minibatches)
        cfg.minibatch_size = (cfg.num_envs * cfg.num_steps) // cfg.num_minibatches

        self.reset_switch, self.step_switch = build_reset_step_switch(self.envs)

        obs_shape = self.env_adapter.observation_shape(temp_env, self.agents)
        if cfg.use_cnn:
            local_obs_dim = obs_shape
            global_obs_dim = (obs_shape[0], obs_shape[1], obs_shape[2] * num_agents)
        else:
            local_obs_dim_flat = int(np.prod(obs_shape))
            global_obs_dim_flat = local_obs_dim_flat * num_agents

        actor_network = Actor(
            action_dim=temp_env.action_space(self.agents[0]).n,
            activation=cfg.activation,
            num_tasks=cfg.seq_length,
            use_multihead=cfg.use_multihead,
            use_task_id=cfg.use_task_id,
            use_cnn=cfg.use_cnn,
            use_layer_norm=cfg.use_layer_norm,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
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
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
        )

        rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

        if cfg.use_cnn:
            actor_init_x = jnp.zeros((1, *local_obs_dim))
            critic_init_x = jnp.zeros((1, *global_obs_dim))
        else:
            actor_init_x = jnp.zeros((1, local_obs_dim_flat))
            critic_init_x = jnp.zeros((1, global_obs_dim_flat))

        actor_params = actor_network.init(actor_rng, actor_init_x, env_idx=0)
        critic_params = critic_network.init(critic_rng, critic_init_x, env_idx=0)

        actor_network.apply = jax.jit(actor_network.apply)
        critic_network.apply = jax.jit(critic_network.apply)

        actor_wrapper = HAPPOActorWrapper(actor_network)

        actor_tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=self.linear_schedule if cfg.anneal_lr else cfg.lr, eps=1e-5),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=self.linear_schedule if cfg.anneal_lr else cfg.lr, eps=1e-5),
        )

        actor_ts = TrainState.create(apply_fn=actor_network.apply, params=actor_params, tx=actor_tx)
        critic_ts = TrainState.create(apply_fn=critic_network.apply, params=critic_params, tx=critic_tx)

        self.rng = rng
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_wrapper = actor_wrapper
        self.actor_tx = actor_tx
        self.critic_tx = critic_tx
        self.actor_ts = actor_ts
        self.critic_ts = critic_ts

        self.evaluate_env = self.env_adapter.make_eval_fn(
            self.cl, self.reset_switch, self.step_switch, actor_wrapper, self.agents,
            cfg.num_envs, cfg.num_steps, cfg.use_cnn, cfg.eval_deterministic, cfg.seed,
        )
        self.importance_fn = self.cl.make_importance_fn(
            self.reset_switch, self.step_switch, actor_wrapper, self.agents, cfg.use_cnn,
            cfg.importance_episodes, cfg.importance_steps, cfg.normalize_importance,
            cfg.importance_stride,
        )

    def build_train_on_environment(self):
        cfg = self.cfg
        actor_network = self.actor_network
        critic_network = self.critic_network
        actor_wrapper = self.actor_wrapper
        actor_tx = self.actor_tx
        critic_tx = self.critic_tx
        cl = self.cl
        agents = self.agents
        num_agents = self.num_agents
        reset_switch, step_switch = self.reset_switch, self.step_switch

        @jax.jit
        def train_on_environment(rng, actor_ts, critic_ts, cl_state, env_idx):
            if cfg.reset_optimizer:
                actor_ts = actor_ts.replace(tx=actor_tx, opt_state=actor_tx.init(actor_ts.params))
                critic_ts = critic_ts.replace(tx=critic_tx, opt_state=critic_tx.init(critic_ts.params))

            rng, env_rng = jax.random.split(rng)
            reset_rng = jax.random.split(env_rng, cfg.num_envs)
            obsv, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(reset_rng)

            reward_shaping_horizon = cfg.steps_per_task / 2
            rew_shaping_anneal = optax.linear_schedule(
                init_value=1.0, end_value=1.0, transition_steps=reward_shaping_horizon,
            )

            def _update_step(runner_state, _):
                def _env_step(runner_state, _):
                    actor_ts, critic_ts, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state
                    rng, _rng = jax.random.split(rng)

                    obs_batch = batchify(last_obs, agents, cfg.num_actors, not cfg.use_cnn)
                    pi, _, _ = actor_wrapper.apply(actor_ts.params, obs_batch, env_idx=env_idx)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)

                    global_state = create_global_state_for_critic(last_obs, agents, cfg.num_envs, cfg.use_cnn)
                    value_per_env, _ = critic_network.apply(critic_ts.params, global_state, env_idx=env_idx)
                    value = jnp.tile(value_per_env, num_agents)

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

                    transition = Transition_HAPPO(
                        done=batchify(done, agents, cfg.num_actors, not cfg.use_cnn).squeeze(),
                        action=action, value=value,
                        reward=batchify(reward, agents, cfg.num_actors).squeeze(),
                        log_prob=log_prob, obs=obs_batch, global_state=global_state,
                    )

                    steps_for_env = steps_for_env + cfg.num_envs
                    runner_state = (actor_ts, critic_ts, env_state, obsv, update_step, steps_for_env, rng, cl_state)
                    return runner_state, (transition, info)

                runner_state, (traj_batch, info) = jax.lax.scan(
                    _env_step, runner_state, xs=None, length=cfg.num_steps
                )
                actor_ts, critic_ts, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state

                last_global_state = create_global_state_for_critic(last_obs, agents, cfg.num_envs, cfg.use_cnn)
                last_val_per_env, _ = critic_network.apply(critic_ts.params, last_global_state, env_idx=env_idx)
                last_val = jnp.tile(last_val_per_env, num_agents)

                advantages, targets = self.calculate_gae(traj_batch, last_val)

                def _update_epoch(update_state, _):
                    actor_ts, critic_ts, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state

                    per_agent_bs = cfg.num_steps * cfg.num_envs
                    mb_size = per_agent_bs // cfg.num_minibatches

                    def slice_agent(x, i):
                        s, e = i * cfg.num_envs, (i + 1) * cfg.num_envs
                        sliced = x[:, s:e]
                        return sliced.reshape(per_agent_bs, *sliced.shape[2:])

                    agent_obs = jnp.stack([slice_agent(traj_batch.obs, i) for i in range(num_agents)])
                    agent_acts = jnp.stack([slice_agent(traj_batch.action, i) for i in range(num_agents)])
                    agent_logps = jnp.stack([slice_agent(traj_batch.log_prob, i) for i in range(num_agents)])
                    agent_vals = jnp.stack([slice_agent(traj_batch.value, i) for i in range(num_agents)])
                    agent_advs = jnp.stack([slice_agent(advantages, i) for i in range(num_agents)])
                    agent_tgts = jnp.stack([slice_agent(targets, i) for i in range(num_agents)])

                    critic_gs = traj_batch.global_state.reshape(per_agent_bs, *traj_batch.global_state.shape[2:])
                    critic_vals = traj_batch.value[:, :cfg.num_envs].reshape(per_agent_bs)
                    critic_tgts = targets[:, :cfg.num_envs].reshape(per_agent_bs)

                    rng, _rng = jax.random.split(rng)
                    perm = jax.random.permutation(_rng, per_agent_bs)

                    agent_obs = jnp.take(agent_obs, perm, axis=1)
                    agent_acts = jnp.take(agent_acts, perm, axis=1)
                    agent_logps = jnp.take(agent_logps, perm, axis=1)
                    agent_vals = jnp.take(agent_vals, perm, axis=1)
                    agent_advs = jnp.take(agent_advs, perm, axis=1)
                    agent_tgts = jnp.take(agent_tgts, perm, axis=1)
                    critic_gs = jnp.take(critic_gs, perm, axis=0)
                    critic_vals = jnp.take(critic_vals, perm, axis=0)
                    critic_tgts = jnp.take(critic_tgts, perm, axis=0)

                    def make_agent_mbs(x):
                        n_a = x.shape[0]
                        rest = x.shape[2:]
                        x_mb = x.reshape(n_a, cfg.num_minibatches, mb_size, *rest)
                        return jnp.swapaxes(x_mb, 0, 1)

                    def make_critic_mbs(x):
                        return x.reshape(cfg.num_minibatches, mb_size, *x.shape[1:])

                    def make_critic_mbs_1d(x):
                        return x.reshape(cfg.num_minibatches, mb_size)

                    xs = (
                        make_agent_mbs(agent_obs), make_agent_mbs(agent_acts), make_agent_mbs(agent_logps),
                        make_agent_mbs(agent_vals), make_agent_mbs(agent_advs), make_agent_mbs(agent_tgts),
                        make_critic_mbs(critic_gs), make_critic_mbs_1d(critic_vals), make_critic_mbs_1d(critic_tgts),
                    )

                    def _update_minbatch(carry, xs_mb):
                        actor_ts, critic_ts, cl_state, rng = carry
                        rng, agem_rng = jax.random.split(rng)

                        (agent_obs_mb, agent_acts_mb, agent_logps_mb,
                         agent_vals_mb, agent_advs_mb, agent_tgts_mb,
                         critic_gs_mb, critic_vals_mb, critic_tgts_mb) = xs_mb

                        def critic_loss_fn(critic_params):
                            value, _ = critic_network.apply(critic_params, critic_gs_mb, env_idx=env_idx)
                            v_clipped = critic_vals_mb + (value - critic_vals_mb).clip(-cfg.clip_eps, cfg.clip_eps)
                            vl = jnp.maximum(
                                jnp.square(value - critic_tgts_mb),
                                jnp.square(v_clipped - critic_tgts_mb),
                            )
                            return 0.5 * vl.mean(), vl.mean()

                        (critic_loss_val, _), critic_grads = jax.value_and_grad(
                            critic_loss_fn, has_aux=True
                        )(critic_ts.params)
                        critic_ts = critic_ts.apply_gradients(grads=critic_grads)

                        M = jnp.ones(agent_obs_mb.shape[1])

                        total_actor_loss = jnp.array(0.0)
                        total_entropy = jnp.array(0.0)
                        total_cl_penalty = jnp.array(0.0)
                        agem_stats = {}

                        for i in range(num_agents):
                            obs_i = agent_obs_mb[i]
                            act_i = agent_acts_mb[i]
                            logp_i = agent_logps_mb[i]
                            adv_i = agent_advs_mb[i]

                            M_i = jax.lax.stop_gradient(M)

                            def happo_actor_loss(actor_params):
                                pi, _ = actor_network.apply(actor_params, obs_i, env_idx=env_idx)
                                log_prob = pi.log_prob(act_i)
                                ratio = jnp.exp(log_prob - logp_i)

                                adv_norm = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)
                                happo_adv = M_i * adv_norm

                                loss_unclipped = ratio * happo_adv
                                loss_clipped = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * happo_adv
                                actor_loss = -jnp.minimum(loss_unclipped, loss_clipped).mean()
                                entropy = pi.entropy().mean()
                                cl_penalty = cl.penalty(actor_params, cl_state, cfg.reg_coef) \
                                    if isinstance(cl, RegCLMethod) else 0.0

                                total = actor_loss - cfg.ent_coef * entropy + cl_penalty
                                return total, (actor_loss, entropy, cl_penalty)

                            (total_i, (al_i, e_i, cp_i)), grads_i = jax.value_and_grad(
                                happo_actor_loss, has_aux=True
                            )(actor_ts.params)

                            actor_ts, agem_rng, agem_stats = self.apply_cl_gradient_step(
                                actor_wrapper, 0.0, cl_state, env_idx, actor_ts, grads_i, agem_rng,
                                is_packnet=(cfg.cl_method.lower() == "packnet"),
                            )

                            # Update M-factor for the next agent
                            pi_new, _ = actor_network.apply(
                                jax.lax.stop_gradient(actor_ts.params), obs_i, env_idx=env_idx,
                            )
                            log_prob_new = pi_new.log_prob(act_i)
                            ratio_new = jax.lax.stop_gradient(jnp.exp(log_prob_new - logp_i))
                            M = M * jnp.clip(ratio_new, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)

                            total_actor_loss = total_actor_loss + al_i
                            total_entropy = total_entropy + e_i
                            total_cl_penalty = total_cl_penalty + cp_i

                        avg_actor_loss = total_actor_loss / num_agents
                        avg_entropy = total_entropy / num_agents
                        avg_cl_penalty = total_cl_penalty / num_agents

                        actor_grad_norm = optax.global_norm(grads_i)
                        critic_grad_norm = optax.global_norm(critic_grads)

                        loss_information = (
                            (critic_loss_val + avg_actor_loss,
                             (critic_loss_val, avg_actor_loss, avg_entropy, avg_cl_penalty)),
                            (actor_grad_norm, critic_grad_norm),
                            agem_stats,
                        )
                        return (actor_ts, critic_ts, cl_state, rng), loss_information

                    (actor_ts, critic_ts, cl_state, rng), loss_information = jax.lax.scan(
                        _update_minbatch, (actor_ts, critic_ts, cl_state, rng), xs,
                    )

                    total_loss, grad_norms, agem_stats = loss_information
                    loss_dict = {"total_loss": total_loss, "grad_norms": grad_norms}
                    if cfg.cl_method.lower() in ("agem", "er_ace"):
                        loss_dict["agem_stats"] = agem_stats

                    update_state = (actor_ts, critic_ts, traj_batch, advantages, targets, steps_for_env, rng, cl_state)
                    return update_state, loss_dict

                update_state = (actor_ts, critic_ts, traj_batch, advantages, targets, steps_for_env, rng, cl_state)
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, xs=None, length=cfg.update_epochs
                )
                actor_ts, critic_ts, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state

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
                metrics["General/learning_rate"] = (
                    self.linear_schedule(
                        (update_step - 1) * cfg.num_minibatches * cfg.update_epochs * num_agents
                    ) if cfg.anneal_lr else cfg.lr
                )
                metrics["General/reward_shaping_anneal"] = rew_shaping_anneal(current_timestep)

                loss_dict = loss_info
                total_loss = loss_dict["total_loss"]
                critic_loss_val, avg_actor_loss, avg_entropy, avg_cl_penalty = total_loss[1]
                total_loss_scalar = total_loss[0]
                actor_grad_norm, critic_grad_norm = loss_dict["grad_norms"]

                metrics["Losses/total_loss"] = total_loss_scalar.mean()
                metrics["Losses/critic_loss"] = critic_loss_val.mean()
                metrics["Losses/actor_loss"] = avg_actor_loss.mean()
                metrics["Losses/entropy"] = avg_entropy.mean()
                metrics["Losses/reg_loss"] = avg_cl_penalty.mean()
                metrics["Gradients/actor_grad_norm"] = actor_grad_norm.mean()
                metrics["Gradients/critic_grad_norm"] = critic_grad_norm.mean()

                if "agem_stats" in loss_dict:
                    for k, v in loss_dict["agem_stats"].items():
                        if v.size > 0:
                            metrics[k] = v.mean()

                metrics = self.compute_env_step_metrics(
                    metrics, info, traj_batch, env_idx, current_timestep, rew_shaping_anneal
                )

                metrics["Advantage_Targets/advantages"] = advantages.mean()
                metrics["Advantage_Targets/advantages_std"] = advantages.std()
                metrics["Advantage_Targets/targets"] = targets.mean()

                obs_batch_last = batchify(last_obs, agents, cfg.num_actors, not cfg.use_cnn)
                _, _, actor_dormant = actor_wrapper.apply(actor_ts.params, obs_batch_last, env_idx=env_idx)
                metrics["Neural_Activity/actor_dormant_ratio"] = actor_dormant

                evaluate_and_log = self.build_evaluate_and_log(
                    cl_state=cl_state, evaluate_env=self.evaluate_env,
                    get_params=lambda: actor_ts.params,
                )
                evaluate_and_log(rng, update_step, metrics, env_idx)

                runner_state = (actor_ts, critic_ts, env_state, last_obs, update_step, steps_for_env, rng, cl_state)
                return runner_state, metrics

            rng, train_rng = jax.random.split(rng)
            runner_state = (actor_ts, critic_ts, env_state, obsv, 0, 0, train_rng, cl_state)
            runner_state, metrics = self.run_train_then_finetune(_update_step, runner_state)
            return runner_state, metrics

        return train_on_environment

    def save_params(self, path, actor_ts, critic_ts, env_kwargs=None, layout_name=None, config=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path + "_actor", "wb") as f:
            f.write(flax.serialization.to_bytes({"params": actor_ts.params}))
        with open(path + "_critic", "wb") as f:
            f.write(flax.serialization.to_bytes({"params": critic_ts.params}))

        if env_kwargs is not None or layout_name is not None or config is not None:
            env_kwargs = convert_frozen_dict(env_kwargs)
            config_data = {"env_kwargs": env_kwargs, "layout_name": layout_name}
            if config is not None:
                config_dict = convert_frozen_dict({
                    "use_cnn": config.use_cnn,
                    "num_tasks": self.cfg.seq_length,
                    "use_multihead": config.use_multihead,
                    "use_task_id": config.use_task_id,
                    "use_layer_norm": config.use_layer_norm,
                    "activation": config.activation,
                    "strategy": getattr(config.env, "strategy", None),
                    "seed": config.seed,
                })
                config_data.update(config_dict)
            config_data = convert_frozen_dict(config_data)
            with open(path + "_config.json", "w") as f:
                json.dump(config_data, f, indent=2)
        print("Model saved to", path)

    def loop_over_envs(self, rng, actor_ts, critic_ts, cl_state, train_on_environment):
        cfg = self.cfg
        adapter = self.env_adapter
        rng, *env_rngs = jax.random.split(rng, cfg.seq_length + 1)
        recorder = adapter.build_visualizer(cfg) if cfg.record_video else None

        for task_idx, (task_rng, env) in enumerate(zip(env_rngs, self.envs)):
            env_name = adapter.env_display_name(env)
            print(f"Training on environment: {task_idx} - {env_name}")
            runner_state, metrics = train_on_environment(task_rng, actor_ts, critic_ts, cl_state, task_idx)
            actor_ts = runner_state[0]
            critic_ts = runner_state[1]
            cl_state = runner_state[7]

            importance = self.importance_fn(actor_ts.params, task_idx, task_rng)
            cl_state = self.cl.update_state(cl_state, actor_ts.params, importance)

            if recorder is not None:
                # HAPPO has no single "network" -- actor_wrapper matches the same
                # (params, obs, *, env_idx) -> (pi, value, dormant) signature the
                # adapter's recorder expects, exactly like actor_ts matches "train_state".
                recorder(task_rng, actor_ts, self.actor_wrapper, env, task_idx, self.exp_dir)

            path = self.checkpoint_path(task_idx)
            self.save_params(path, actor_ts, critic_ts, env_kwargs=getattr(env, "layout", None),
                             layout_name=env_name, config=cfg)

            if cfg.single_task_idx is not None:
                break

    def run(self):
        self.setup_envs()
        self.init_network()
        self.setup_logging()

        if self.cfg.cl_method.lower() in ("agem", "er_ace"):
            obs_dim_for_mem = self.env_adapter.observation_shape(self.envs[0], self.agents)
            if not self.cfg.use_cnn:
                obs_dim_for_mem = (int(np.prod(obs_dim_for_mem)),)
            cl_state = init_agem_memory(self.cfg.agem_memory_size, obs_dim_for_mem, max_tasks=self.cfg.seq_length)
        else:
            # CL state tracks actor params only (no critic regularisation).
            cl_state = init_cl_state(self.actor_ts.params, regularize_critic=False,
                                     regularize_heads=not self.cfg.use_multihead, cl=self.cl, cfg=self.cfg)

        train_on_environment = self.build_train_on_environment()

        rng, train_rng = jax.random.split(self.rng)
        self.loop_over_envs(train_rng, self.actor_ts, self.critic_ts, cl_state, train_on_environment)
