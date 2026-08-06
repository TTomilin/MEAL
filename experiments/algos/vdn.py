from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.algo_common import build_reset_step_switch
from experiments.algos.off_policy import OffPolicyAlgo, OffPolicyConfig
from experiments.continual.agem import (
    AGEM, init_vdn_agem_memory, sample_vdn_task_slot,
    compute_vdn_memory_gradient, update_vdn_agem_memory, agem_project,
)
from experiments.continual.base import RegCLMethod
from experiments.continual.er_ace import ERACE
from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.l2 import L2
from experiments.continual.mas import MAS
from experiments.continual.packnet import Packnet
from experiments.evaluation import evaluate_all_envs
from experiments.model.q_mlp import QNetwork
from experiments.utils import add_eval_metrics, batchify, create_visualizer, init_cl_state, unbatchify
from experiments.utils_vdn import (
    CustomTrainState, Timestep, eps_greedy_exploration,
    batchify as vdn_batchify, unbatchify as vdn_unbatchify,
)
from meal.wrappers.jaxmarl import CTRolloutManager


@dataclass
class VDNConfig(OffPolicyConfig):
    alg_name: str = "vdn"


def make_vdn_eval_fn(reset_switch, step_switch, network, agents, num_envs: int,
                     num_steps: int, use_cnn: bool, eval_deterministic: bool, seed: int):
    """Returns a JITted evaluate_env(rng, params, env_idx) -> (avg_reward, avg_soups)
    compatible with evaluate_all_envs(). Uses greedy (argmax) Q-values."""
    num_agents = len(agents)

    @jax.jit
    def evaluate_env(cl_state, rng, params, env_idx):
        if eval_deterministic:
            rng = jax.random.PRNGKey(env_idx + seed)
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, num_envs)
        obs, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(reset_rng)

        total_rewards = jnp.zeros((num_envs,), jnp.float32)
        total_soups = jnp.zeros((num_envs,), jnp.float32)

        def one_step(carry, _):
            env_state, obs, rewards, soups, rng = carry

            obs_b = batchify(obs, agents, num_agents * num_envs, not use_cnn)
            obs_b = obs_b.reshape((num_agents, num_envs) + obs_b.shape[1:])

            q_vals = jax.vmap(
                lambda p, o: network.apply(p, o, env_idx=env_idx), in_axes=(None, 0)
            )(params, obs_b)

            actions_array = jnp.argmax(q_vals, axis=-1)
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
    """Importance estimation for Q-networks (used by EWC/MAS). Computes gradient of the
    squared Q-value norm (MAS-style output sensitivity)."""
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

                obs_b = jnp.stack([obs[a] for a in agents])[:, jnp.newaxis]

                def q_l2_loss(p):
                    q = jax.vmap(
                        lambda pp, o: network.apply(pp, o, env_idx=env_idx),
                        in_axes=(None, 0)
                    )(p, obs_b)
                    q = q.squeeze(1)
                    return 0.5 * jnp.sum(q * q) / q.shape[0]

                grads = jax.grad(q_l2_loss)(params)
                alpha = (t % stride == 0).astype(jnp.float32)
                g2 = jax.tree.map(lambda g: g * g * alpha, grads)
                acc = jax.tree.map(jnp.add, acc, g2)

                q_vals = jax.vmap(
                    lambda pp, o: network.apply(pp, o, env_idx=env_idx), in_axes=(None, 0)
                )(params, obs_b).squeeze(1)
                acts = jnp.argmax(q_vals, axis=-1)
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


def rollout_for_video_vdn(rng, config, train_state, env, network, env_idx=0, max_steps=300):
    """Record a single-environment rollout using greedy Q-values for visualization."""
    rng, env_rng = jax.random.split(rng)
    obs, state = env.reset(env_rng)
    done = False
    step_count = 0
    states = [env.unwrap_env_state(state)]

    while not done and step_count < max_steps:
        actions = {}
        for agent_id in env.agents:
            obs_v = obs[agent_id]
            obs_b = obs_v[None]
            if not config.use_cnn:
                obs_b = obs_b.reshape(1, -1)
            q_vals = network.apply(train_state.params, obs_b, env_idx=env_idx)
            actions[agent_id] = jnp.argmax(q_vals[0])

        rng, key_step = jax.random.split(rng)
        next_obs, next_state, reward, done_info, info = env.step(key_step, state, actions)
        done = done_info["__all__"]
        obs, state = next_obs, next_state
        step_count += 1
        states.append(env.unwrap_env_state(state))

    return states


class VDN(OffPolicyAlgo):
    def _method_map(self):
        cfg = self.cfg
        return dict(
            ewc=EWC(mode=cfg.importance_mode, decay=cfg.importance_decay),
            mas=MAS(mode=cfg.importance_mode, decay=cfg.importance_decay),
            l2=L2(),
            ft=FT(),
            agem=AGEM(),
            er_ace=ERACE(memory_size=cfg.agem_memory_size, sample_size=cfg.agem_sample_size),
            packnet=Packnet(seq_length=cfg.seq_length, prune_instructions=0.4,
                            train_finetune_split=(cfg.train_epochs, cfg.finetune_epochs),
                            prunable_layers=[]),
        )

    def init_network(self):
        cfg = self.cfg
        temp_env = self.temp_env

        self.train_envs = [
            CTRolloutManager(env, batch_size=cfg.num_envs, preprocess_obs=False)
            for env in self.envs
        ]

        total_grad_steps = cfg.update_epochs * cfg.num_minibatches * cfg.num_updates
        lr_scheduler = optax.linear_schedule(cfg.lr, cfg.lr_end, total_grad_steps)
        lr = lr_scheduler if cfg.anneal_lr else cfg.lr
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.radam(learning_rate=lr),
        )
        eps_scheduler = optax.linear_schedule(
            cfg.eps_start, cfg.eps_finish, cfg.eps_decay * cfg.num_updates
        )
        reward_shaping_horizon = cfg.reward_shaping_horizon
        if reward_shaping_horizon is None:
            reward_shaping_horizon = cfg.steps_per_task / 2
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1., end_value=0.,
            transition_steps=reward_shaping_horizon / (cfg.num_steps * cfg.num_envs),
        )

        rng, net_rng = jax.random.split(self.rng)

        network = QNetwork(
            action_dim=self.train_envs[0].max_action_space,
            hidden_size=cfg.hidden_size,
            activation=cfg.activation,
            use_layer_norm=cfg.use_layer_norm,
            use_multihead=cfg.use_multihead,
            use_task_id=cfg.use_task_id,
            num_tasks=cfg.seq_length,
            encoder_type="cnn" if cfg.use_cnn else "mlp",
        )
        network.apply = jax.jit(network.apply)

        obs_shape = self.env_adapter.observation_shape(temp_env, self.agents)
        obs_dim = int(np.prod(obs_shape))
        init_x = jnp.zeros((1,) + obs_shape) if cfg.use_cnn else jnp.zeros((1, obs_dim))
        network_params = network.init(net_rng, init_x)

        train_state = CustomTrainState.create(
            apply_fn=network.apply, params=network_params,
            target_network_params=network_params, tx=tx,
        )

        self.rng = rng
        self.network = network
        self.tx = tx
        self.lr_scheduler = lr_scheduler
        self.eps_scheduler = eps_scheduler
        self.rew_shaping_anneal = rew_shaping_anneal
        self.train_state = train_state
        self.reset_switch, self.step_switch = build_reset_step_switch(self.envs)

        self.evaluate_env = make_vdn_eval_fn(
            self.reset_switch, self.step_switch, network, self.agents,
            num_envs=cfg.num_envs, num_steps=self.env_adapter.max_steps, use_cnn=cfg.use_cnn,
            eval_deterministic=cfg.eval_deterministic, seed=cfg.seed
        )

        if cfg.cl_method.lower() in ("ewc", "mas"):
            self.importance_fn = make_q_importance_fn(
                self.reset_switch, self.step_switch, network, self.agents, cfg.use_cnn,
                cfg.importance_episodes, cfg.importance_steps,
                cfg.normalize_importance, cfg.importance_stride
            )
        else:
            self.importance_fn = self.cl.make_importance_fn(
                self.reset_switch, self.step_switch, network, self.agents, cfg.use_cnn,
                cfg.importance_episodes, cfg.importance_steps,
                cfg.normalize_importance, cfg.importance_stride
            )

    def build_train_on_environment(self):
        cfg = self.cfg
        network = self.network
        tx = self.tx
        cl = self.cl
        agents = self.agents
        num_agents = self.num_agents
        eps_scheduler = self.eps_scheduler
        lr_scheduler = self.lr_scheduler
        rew_shaping_anneal = self.rew_shaping_anneal

        N = cfg.num_steps * cfg.num_envs
        minibatch_size = N // cfg.num_minibatches

        @partial(jax.jit, static_argnums=(2, 4))
        def train_on_environment(rng, train_state, train_env, cl_state, env_idx):
            new_opt_state = tx.init(train_state.params)
            train_state = train_state.replace(tx=tx, opt_state=new_opt_state, n_updates=0, grad_steps=0)

            def _update_step(runner_state, _):
                train_state, expl_state, rng = runner_state

                def _step_env(carry, _):
                    last_obs, env_state, rng = carry
                    rng, rng_action, rng_step = jax.random.split(rng, 3)

                    obs_b = batchify(last_obs, agents, num_agents * cfg.num_envs, not cfg.use_cnn)
                    obs_b = obs_b.reshape((num_agents, cfg.num_envs) + obs_b.shape[1:])

                    q_vals = jax.vmap(
                        lambda p, o: network.apply(p, o, env_idx=env_idx), in_axes=(None, 0)
                    )(train_state.params, obs_b)

                    avail_actions = train_env.get_valid_actions(env_state)
                    eps = eps_scheduler(train_state.n_updates)
                    _rngs = jax.random.split(rng_action, num_agents)
                    new_action = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                        _rngs, q_vals, eps, vdn_batchify(avail_actions, agents)
                    )

                    actions = vdn_unbatchify(new_action, agents)

                    new_obs, new_env_state, rewards, dones, infos = train_env.batch_step(
                        rng_step, env_state, actions
                    )

                    shaped_reward = self.env_adapter.get_shaped_reward(infos, agents)
                    if shaped_reward is not None:
                        shaped_reward["__all__"] = vdn_batchify(shaped_reward, agents).sum(axis=0)
                        anneal = rew_shaping_anneal(train_state.n_updates)
                        shaped_reward = jax.tree.map(lambda y: y * anneal, shaped_reward)
                        rewards = jax.tree.map(lambda x, y: x + y, rewards, shaped_reward)

                    timestep = Timestep(
                        obs={a: last_obs[a] for a in agents},
                        actions=actions, avail_actions=avail_actions,
                        rewards=rewards, dones=dones,
                    )
                    return (new_obs, new_env_state, rng), (timestep, infos, shaped_reward)

                rng, _rng = jax.random.split(rng)
                (new_obs, new_env_state, _), (timesteps, infos, shaped_rewards) = jax.lax.scan(
                    _step_env, (*expl_state, _rng), xs=None, length=cfg.num_steps
                )
                expl_state = (new_obs, new_env_state)

                train_state = train_state.replace(
                    timesteps=train_state.timesteps + cfg.num_steps * cfg.num_envs
                )

                next_obs = {
                    a: jnp.concatenate([timesteps.obs[a][1:], new_obs[a][jnp.newaxis]], axis=0)
                    for a in agents
                }

                obs_flat = {a: timesteps.obs[a].reshape((N,) + timesteps.obs[a].shape[2:]) for a in agents}
                nxt_flat = {a: next_obs[a].reshape((N,) + next_obs[a].shape[2:]) for a in agents}
                act_flat = {a: timesteps.actions[a].reshape(N) for a in agents}
                rew_flat = timesteps.rewards["__all__"].reshape(N)
                don_flat = timesteps.dones["__all__"].reshape(N).astype(jnp.float32)

                nxt_b = jnp.stack([nxt_flat[a] for a in agents])
                q_next = jax.vmap(
                    lambda p, o: network.apply(p, o, env_idx=env_idx), in_axes=(None, 0)
                )(train_state.target_network_params, nxt_b)
                q_next_max = jnp.max(q_next, axis=-1)
                vdn_target = rew_flat + (1 - don_flat) * cfg.gamma * jnp.sum(q_next_max, axis=0)

                def _learn_minibatch(train_state, mb_indices):
                    mb_obs = {a: obs_flat[a][mb_indices] for a in agents}
                    mb_acts = {a: act_flat[a][mb_indices] for a in agents}
                    mb_tgt = vdn_target[mb_indices]

                    def _loss_fn(params):
                        obs_b = jnp.stack([mb_obs[a] for a in agents])
                        q_vals = jax.vmap(
                            lambda p, o: network.apply(p, o, env_idx=env_idx), in_axes=(None, 0)
                        )(params, obs_b)

                        chosen_q = jnp.take_along_axis(
                            q_vals, jnp.stack([mb_acts[a] for a in agents])[..., jnp.newaxis], axis=-1,
                        ).squeeze(-1)

                        chosen_q_sum = jnp.sum(chosen_q, axis=0)
                        td_loss = jnp.mean((chosen_q_sum - mb_tgt) ** 2)
                        cl_penalty = cl.penalty(params, cl_state, cfg.reg_coef) if isinstance(cl, RegCLMethod) else 0.0
                        total_loss = td_loss + cl_penalty
                        return total_loss, (td_loss, chosen_q_sum.mean(), cl_penalty)

                    (total_loss, (td_loss, qvals, cl_penalty)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)

                    if cfg.cl_method.lower() == "agem":
                        past_sizes = cl_state.sizes.at[env_idx].set(0)
                        _max_tasks = cl_state.obs.shape[0]
                        samples_per_task = max(1, cfg.agem_sample_size // _max_tasks)
                        g_mem = jax.tree.map(jnp.zeros_like, grads)
                        for _t in range(_max_tasks):
                            _t_rng = jax.random.fold_in(rng, _t)
                            _t_obs, _t_acts, _t_rews, _t_nobs, _t_dones = sample_vdn_task_slot(
                                cl_state, _t, samples_per_task, _t_rng
                            )
                            _t_grads = compute_vdn_memory_gradient(
                                network, train_state.params, train_state.target_network_params,
                                cfg.gamma, _t_obs, _t_acts, _t_rews, _t_nobs, _t_dones, env_idx=_t,
                            )
                            _mask = (past_sizes[_t] > 0).astype(jnp.float32)
                            _t_grads = jax.tree.map(lambda g, m=_mask: g * m, _t_grads)
                            g_mem = jax.tree.map(jnp.add, g_mem, _t_grads)

                        def _zero_q_head(path, g):
                            return jnp.zeros_like(g) if "q_head" in "/".join(str(p) for p in path) else g

                        g_mem = jax.tree_util.tree_map_with_path(_zero_q_head, g_mem)
                        grads, _ = agem_project(grads, g_mem)

                    elif cfg.cl_method.lower() == "er_ace":
                        past_sizes = cl_state.sizes.at[env_idx].set(0)
                        _max_tasks = cl_state.obs.shape[0]
                        samples_per_task = max(1, cfg.agem_sample_size // _max_tasks)
                        g_mem = jax.tree.map(jnp.zeros_like, grads)
                        for _t in range(_max_tasks):
                            _t_rng = jax.random.fold_in(rng, _t)
                            _t_obs, _t_acts, _t_rews, _t_nobs, _t_dones = sample_vdn_task_slot(
                                cl_state, _t, samples_per_task, _t_rng
                            )
                            _t_grads = compute_vdn_memory_gradient(
                                network, train_state.params, train_state.target_network_params,
                                cfg.gamma, _t_obs, _t_acts, _t_rews, _t_nobs, _t_dones, env_idx=_t,
                            )
                            _mask = (past_sizes[_t] > 0).astype(jnp.float32)
                            _t_grads = jax.tree.map(lambda g, m=_mask: g * m, _t_grads)
                            g_mem = jax.tree.map(jnp.add, g_mem, _t_grads)
                        grads = jax.tree.map(lambda g, gm: g + cfg.er_ace_coef * gm, grads, g_mem)

                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)
                    return train_state, (total_loss, td_loss, qvals, cl_penalty)

                def _learn_epoch(carry, _):
                    train_state, rng = carry
                    rng, perm_rng = jax.random.split(rng)
                    perm = jax.random.permutation(perm_rng, N).reshape(cfg.num_minibatches, minibatch_size)
                    train_state, losses = jax.lax.scan(_learn_minibatch, train_state, perm)
                    return (train_state, rng), losses

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (total_loss, td_loss, qvals, cl_penalty) = jax.lax.scan(
                    _learn_epoch, (train_state, _rng), xs=None, length=cfg.update_epochs
                )
                total_loss = total_loss.reshape(-1).mean()
                td_loss = td_loss.reshape(-1).mean()
                qvals = qvals.reshape(-1).mean()
                cl_penalty = cl_penalty.reshape(-1).mean()

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
                    "General/learning_rate": lr_scheduler(train_state.grad_steps) if cfg.anneal_lr else cfg.lr,
                    "Losses/total_loss": total_loss,
                    "Losses/td_loss": td_loss,
                    "Losses/reg_loss": cl_penalty,
                    "Values/qvals": qvals,
                    "Rewards/step_reward": timesteps.rewards["__all__"].mean(),
                }
                # Soup/shaped-reward metrics are Overcooked-specific (rely on
                # info["soups"] / info["shaped_reward"] existing) -- only computed when
                # the active env adapter actually provides them (Overcooked). Other envs
                # still train correctly; they just don't get these particular summary
                # metrics logged (off-policy per-env metrics remain Overcooked-only for
                # now, see EnvAdapter.get_shaped_reward's docstring).
                if hasattr(self.env_adapter, "max_soup_vals"):
                    soups_info = infos.pop("soups", {})
                    soups_tea = jnp.stack([soups_info[a] for a in agents], axis=-1)
                    soups_per_env = soups_tea.sum(axis=(0, 2))
                    don_te = timesteps.dones["__all__"]
                    episodes_per_env = don_te.sum(axis=0)
                    mask = episodes_per_env > 0
                    true_avg = jnp.where(mask, soups_per_env / jnp.maximum(episodes_per_env, 1), 0.0)
                    num_finished = jnp.maximum(mask.sum(), 1)
                    max_per_episode = self.env_adapter.max_soup_vals[env_idx]
                    metrics["Soup/total"] = true_avg.sum() / num_finished
                    metrics["Soup/scaled"] = jnp.where(
                        max_per_episode > 0, (true_avg / max_per_episode).sum() / num_finished, 0.0
                    )
                    for ai, agent in enumerate(agents):
                        soups_te = soups_tea[:, :, ai].sum(axis=0)
                        per_agent = jnp.where(mask, soups_te / jnp.maximum(episodes_per_env, 1), 0.0)
                        metrics[f"Soup/{agent}"] = per_agent.sum() / num_finished
                    for agent in agents:
                        metrics[f"General/shaped_reward_{agent}"] = shaped_rewards[agent].mean()
                metrics.update(jax.tree.map(lambda x: x.mean(), infos))

                def evaluate_and_log(rng, update_step):
                    rng, eval_rng = jax.random.split(rng)

                    def log_metrics(metrics, update_step):
                        if cfg.evaluation:
                            avg_rewards, avg_soups = evaluate_all_envs(
                                cl_state, eval_rng, train_state.params, cfg.seq_length, self.evaluate_env
                            )
                            metrics_ = self.add_eval_metrics(avg_rewards, avg_soups, metrics)
                        else:
                            metrics_ = metrics

                        def callback(args):
                            m, step, env_ctr = args
                            real_step = int((env_ctr - 1) * cfg.num_updates + step)
                            for k, v in m.items():
                                self.writer.add_scalar(k, float(v), real_step)

                        jax.experimental.io_callback(callback, None, (metrics_, update_step, jnp.int32(env_idx + 1)))
                        return None

                    def do_not_log(m, s):
                        return None

                    jax.lax.cond(
                        (update_step % cfg.log_interval) == 0, log_metrics, do_not_log, metrics, update_step,
                    )

                evaluate_and_log(rng=rng, update_step=train_state.n_updates)

                runner_state = (train_state, expl_state, rng)
                return runner_state, metrics

            rng, reset_rng = jax.random.split(rng)
            obs, env_state = train_env.batch_reset(reset_rng)

            rng, _rng = jax.random.split(rng)
            runner_state = (train_state, (obs, env_state), _rng)
            runner_state, _ = jax.lax.scan(
                _update_step, runner_state, xs=None, length=cfg.num_updates
            )

            final_train_state = runner_state[0]
            final_expl_state = runner_state[1]

            if cfg.cl_method.lower() in ("agem", "er_ace"):
                rng, mem_rng = jax.random.split(rng)

                def _mem_step(carry, _):
                    last_obs, env_state, rng = carry
                    rng, rng_a, rng_s = jax.random.split(rng, 3)

                    obs_b = batchify(last_obs, agents, num_agents * cfg.num_envs, not cfg.use_cnn)
                    obs_b = obs_b.reshape((num_agents, cfg.num_envs) + obs_b.shape[1:])

                    q_vals = jax.vmap(
                        lambda p, o: network.apply(p, o, env_idx=env_idx), in_axes=(None, 0)
                    )(final_train_state.params, obs_b)

                    avail_actions = train_env.get_valid_actions(env_state)
                    _rngs = jax.random.split(rng_a, num_agents)
                    new_action = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                        _rngs, q_vals, cfg.eps_finish, vdn_batchify(avail_actions, agents)
                    )
                    actions = vdn_unbatchify(new_action, agents)

                    new_obs, new_env_state, rewards, dones, infos = train_env.batch_step(
                        rng_s, env_state, actions
                    )
                    shaped_reward = self.env_adapter.get_shaped_reward(infos, agents)
                    if shaped_reward is not None:
                        shaped_reward["__all__"] = vdn_batchify(shaped_reward, agents).sum(axis=0)
                        # Memory collection runs post-training at the fixed final epsilon
                        # (cfg.eps_finish above), so use the fully-annealed shaping weight too.
                        anneal = rew_shaping_anneal(final_train_state.n_updates)
                        shaped_reward = jax.tree.map(lambda y: y * anneal, shaped_reward)
                        rewards = jax.tree.map(lambda x, y: x + y, rewards, shaped_reward)

                    timestep = Timestep(
                        obs={a: last_obs[a] for a in agents},
                        actions=actions, avail_actions=avail_actions,
                        rewards=rewards, dones=dones,
                    )
                    return (new_obs, new_env_state, rng), timestep

                (new_obs_mem, _, _), mem_ts = jax.lax.scan(
                    _mem_step, (*final_expl_state, mem_rng), xs=None, length=cfg.num_steps
                )

                next_obs_mem = {
                    a: jnp.concatenate([mem_ts.obs[a][1:], new_obs_mem[a][jnp.newaxis]], axis=0)
                    for a in agents
                }

                N_mem = cfg.num_steps * cfg.num_envs
                obs_m = {a: mem_ts.obs[a].reshape(N_mem, -1) for a in agents}
                nxt_m = {a: next_obs_mem[a].reshape(N_mem, -1) for a in agents}
                act_m = {a: mem_ts.actions[a].reshape(N_mem) for a in agents}
                rew_m = mem_ts.rewards["__all__"].reshape(N_mem)
                don_m = mem_ts.dones["__all__"].reshape(N_mem).astype(jnp.float32)

                rng, samp_rng = jax.random.split(rng)
                samp_idx = jax.random.choice(samp_rng, N_mem, (cfg.agem_sample_size,), replace=False)

                _obs_b = jnp.stack([obs_m[a][samp_idx] for a in agents]).transpose(1, 0, 2)
                _nxt_b = jnp.stack([nxt_m[a][samp_idx] for a in agents]).transpose(1, 0, 2)
                _act_b = jnp.stack([act_m[a][samp_idx] for a in agents]).T
                _rew = rew_m[samp_idx]
                _don = don_m[samp_idx]

                cl_state = update_vdn_agem_memory(cl_state, env_idx, _obs_b, _act_b, _rew, _nxt_b, _don)

            return rng, final_train_state, cl_state

        return train_on_environment

    def add_eval_metrics(self, avg_rewards, avg_soups, metrics):
        return add_eval_metrics(avg_rewards, avg_soups, self.env_names, self.env_adapter.max_soup_vals, metrics)

    def loop_over_envs(self, rng, train_state, cl_state, train_on_environment):
        cfg = self.cfg
        rng, *env_rngs = jax.random.split(rng, cfg.seq_length + 1)

        visualizer = None
        for task_idx, (env_rng, train_env, env) in enumerate(
                zip(env_rngs, self.train_envs, self.envs)
        ):
            env_name = self.env_adapter.env_display_name(env)
            print(f"Training on task {task_idx + 1}/{cfg.seq_length}: {env_name}")

            rng, train_state, cl_state = train_on_environment(env_rng, train_state, train_env, cl_state, task_idx)

            importance = self.importance_fn(train_state.params, task_idx, rng)
            cl_state = self.cl.update_state(cl_state, train_state.params, importance)

            if cfg.record_video:
                if visualizer is None:
                    visualizer = create_visualizer(self.num_agents, getattr(cfg.env, "env_name", "overcooked"))
                states = rollout_for_video_vdn(rng, cfg, train_state, env, self.network, task_idx, cfg.video_length)
                file_path = f"{self.exp_dir}/task_{task_idx}_{env_name}.mp4"
                visualizer.animate(states, out_path=file_path, task_idx=task_idx, env=env)

            path = self.checkpoint_path(task_idx)
            self.save_params(path, train_state, env_kwargs=getattr(env, "layout", None), layout_name=env_name,
                             config=cfg)

    def run(self):
        self.setup_envs()
        self.init_network()
        self.setup_logging()

        if self.cfg.cl_method.lower() in ("agem", "er_ace"):
            obs_dim = int(np.prod(self.env_adapter.observation_shape(self.temp_env, self.agents)))
            cl_state = init_vdn_agem_memory(
                max_size=self.cfg.agem_memory_size, num_agents=self.num_agents,
                obs_dim=obs_dim, max_tasks=self.cfg.seq_length,
            )
        else:
            cl_state = init_cl_state(self.train_state.params, False, not self.cfg.use_multihead, self.cl, self.cfg)

        train_on_environment = self.build_train_on_environment()

        rng, train_rng = jax.random.split(self.rng)
        self.loop_over_envs(train_rng, self.train_state, cl_state, train_on_environment)
