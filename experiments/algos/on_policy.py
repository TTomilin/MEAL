"""OnPolicyAlgo: shared pieces for IPPO / MAPPO / HAPPO.

What's genuinely different per-algorithm (network forward pass, Transition
shape, loss function) stays in each concrete class's own `build_train_on_environment`.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax._src.flatten_util import ravel_pytree

from experiments.algo_common import apply_packnet_mask_or_plain, run_packnet_train_then_finetune
from experiments.algos.base import AbstractAlgo, BaseConfig
from experiments.continual.agem import compute_memory_gradient, agem_project, sample_task_slot
from experiments.continual.er_ace import compute_er_ace_gradient


@dataclass
class OnPolicyConfig(BaseConfig):
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING / PPO PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
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
    reset_optimizer: bool = True

    # ═══════════════════════════════════════════════════════════════════════════
    # NETWORK SIZE PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    hidden_size: int = 128
    num_layers: int = 2

    # ═══════════════════════════════════════════════════════════════════════════
    # RUNTIME COMPUTED PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    finetune_updates: int = 0
    minibatch_size: int = 0


class OnPolicyAlgo(AbstractAlgo):
    def calculate_gae(self, traj_batch, last_val):
        cfg = self.cfg

        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + cfg.gamma * next_value * (1 - done) - value
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            f=_get_advantages,
            init=(jnp.zeros_like(last_val), last_val),
            xs=traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    def apply_cl_gradient_step(self, network, vf_coef, cl_state, env_idx, train_state, grads, agem_rng, is_packnet):
        """Unified AGEM projection / ER-ACE addition / packnet-mask-or-plain apply.

        `network` must have `.apply(params, obs, *, env_idx) -> (pi, value, dormant)`.
        `vf_coef` controls compute_memory_gradient's critic term for AGEM memory replay:
        pass the algorithm's real cfg.vf_coef when `network`'s value output is meaningful
        (IPPO's joint network), or 0.0 when it's a placeholder (MAPPO/HAPPO's decoupled
        critics -- AGEM memory only stores local obs, not the global state their real
        critics need). `is_packnet` lets callers preserve their own cl_method-casing check
        (ippo.py/mappo.py used `==`, happo.py used `.lower() ==`).

        Returns (new_train_state, new_agem_rng, agem_stats). `agem_rng` is split exactly
        once per call (regardless of which branch runs) so repeated calls in the same
        Python scope -- e.g. HAPPO's per-agent loop -- never silently reuse a stale key.
        """
        cfg = self.cfg
        agem_rng, sub_rng = jax.random.split(agem_rng)
        agem_stats = {}

        if cfg.cl_method.lower() == "agem" and cl_state is not None:
            past_sizes = cl_state.sizes.at[env_idx].set(0)

            def apply_agem_projection(rng, past_sizes):
                max_tasks = cl_state.obs.shape[0]
                samples_per_task = max(cfg.agem_sample_size // max_tasks, 1)
                grads_mem = None
                ppo_stats_sum = {
                    "agem/ppo_total_loss": jnp.array(0.0),
                    "agem/ppo_value_loss": jnp.array(0.0),
                    "agem/ppo_actor_loss": jnp.array(0.0),
                    "agem/ppo_entropy": jnp.array(0.0),
                }
                for t in range(max_tasks):
                    rng, task_rng = jax.random.split(rng)
                    t_obs, t_actions, t_logp, t_advs, t_targets, t_values = sample_task_slot(
                        cl_state, t, samples_per_task, task_rng
                    )
                    t_grads, t_stats = compute_memory_gradient(
                        network, train_state.params,
                        cfg.clip_eps, vf_coef, cfg.ent_coef,
                        t_obs, t_actions, t_advs, t_logp,
                        t_targets, t_values,
                        env_idx=t,
                    )
                    mask = (past_sizes[t] > 0).astype(jnp.float32)
                    t_grads = jax.tree_util.tree_map(lambda g: g * mask, t_grads)
                    grads_mem = t_grads if grads_mem is None else jax.tree_util.tree_map(
                        lambda a, b: a + b, grads_mem, t_grads
                    )
                    for k in ppo_stats_sum:
                        ppo_stats_sum[k] = ppo_stats_sum[k] + t_stats[k] * mask

                n_active = jnp.sum((past_sizes > 0).astype(jnp.float32)) + 1e-8
                ppo_stats = {k: v / n_active for k, v in ppo_stats_sum.items()}

                projected_grads, proj_stats = agem_project(grads, grads_mem)
                combined_stats = {**ppo_stats, **proj_stats}
                mem_norm = jnp.linalg.norm(ravel_pytree(grads_mem)[0])
                combined_stats["agem/mem_grad_norm_raw"] = mem_norm
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
                lambda: apply_agem_projection(sub_rng, past_sizes),
                lambda: no_agem_projection(),
            )
            train_state = train_state.apply_gradients(grads=final_grads)

        elif cfg.cl_method.lower() == "er_ace" and cl_state is not None:
            past_sizes = cl_state.sizes.at[env_idx].set(0)
            er_ace_grads, er_ace_stats = compute_er_ace_gradient(
                network, train_state.params, cl_state,
                cfg.agem_sample_size, sub_rng, past_sizes,
            )
            grads = jax.tree_util.tree_map(
                lambda g, eg: g + cfg.er_ace_coef * eg, grads, er_ace_grads
            )
            train_state = train_state.apply_gradients(grads=grads)
            agem_stats = er_ace_stats

        else:
            train_state = apply_packnet_mask_or_plain(is_packnet, self.cl, cl_state, grads, train_state)

        return train_state, agem_rng, agem_stats

    def compute_env_step_metrics(self, metrics, info, traj_batch, env_idx, current_timestep, rew_shaping_anneal):
        """`metrics` must already be `jax.tree_util.tree_map(lambda x: x.mean(), info)`
        -- see EnvAdapter.compute_step_metrics's docstring for the exact convention."""
        return self.env_adapter.compute_step_metrics(
            metrics, self.cfg, info, traj_batch, env_idx, self.num_agents, self.agents,
            current_timestep, rew_shaping_anneal,
        )

    def run_train_then_finetune(self, update_step_fn, runner_state):
        return run_packnet_train_then_finetune(
            self.cfg, self.cl, update_step_fn, runner_state, self.cfg.finetune_updates
        )

    def build_evaluate_and_log(self, cl_state, evaluate_env, get_params):
        """Returns evaluate_and_log(rng, update_step, metrics, env_idx) -> None, matching
        ippo.py / mappo.py's / happo.py's equivalents (all three go through
        `self.env_adapter.evaluate_all_envs`/`add_eval_metrics`)."""
        cfg = self.cfg
        writer = self.writer
        env_names = self.env_names
        seq_length = cfg.seq_length
        adapter = self.env_adapter

        def evaluate_and_log(rng, update_step, metrics, env_idx):
            rng, eval_rng = jax.random.split(rng)

            def log_metrics(metrics, update_step):
                if cfg.evaluation:
                    eval_result = adapter.evaluate_all_envs(
                        cl_state, eval_rng, get_params(), seq_length, evaluate_env
                    )
                    metrics_ = adapter.add_eval_metrics(metrics, eval_result, env_names)
                else:
                    metrics_ = metrics

                def callback(args):
                    m, step, env_counter = args
                    real_step = (env_counter - 1) * cfg.num_updates + step
                    for key, value in m.items():
                        writer.add_scalar(key, value, real_step)

                jax.experimental.io_callback(callback, None, (metrics_, update_step, env_idx + 1))
                return None

            def do_not_log(metrics, update_step):
                return None

            jax.lax.cond((update_step % cfg.log_interval) == 0, log_metrics, do_not_log, metrics, update_step)

        return evaluate_and_log

    def loop_over_envs(self, rng, train_state, cl_state, train_on_environment, save_params, importance_fn,
                       network_for_video):
        """Generic outer per-task loop for algorithms with a single TrainState (IPPO,
        MAPPO). HAPPO overrides this (dual actor/critic TrainState)."""
        cfg = self.cfg
        adapter = self.env_adapter
        rng, *env_rngs = jax.random.split(rng, cfg.seq_length + 1)
        recorder = adapter.build_visualizer(cfg) if cfg.record_video else None

        for task_idx, (task_rng, env) in enumerate(zip(env_rngs, self.envs)):
            if cfg.single_task_idx is not None and task_idx != cfg.single_task_idx:
                continue
            env_name = adapter.env_display_name(env)
            print(f"Training on environment: {task_idx} - {env_name}")
            runner_state, metrics = train_on_environment(task_rng, train_state, cl_state, task_idx)
            train_state = runner_state[0]
            cl_state = runner_state[-1]

            importance = importance_fn(train_state.params, task_idx, task_rng)
            cl_state = self.cl.update_state(cl_state, train_state.params, importance)

            if recorder is not None:
                recorder(task_rng, train_state, network_for_video, env, task_idx, self.exp_dir)

            path = self.checkpoint_path(task_idx)
            save_params(path, train_state, env_kwargs=getattr(env, "layout", None), layout_name=env_name, config=cfg)

            if cfg.single_task_idx is not None:
                break

        return train_state, cl_state
