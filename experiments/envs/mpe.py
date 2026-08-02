from dataclasses import dataclass

import jax
import jax.numpy as jnp

from experiments.continual.packnet import Packnet
from experiments.envs.base import EnvAdapter
from experiments.utils import batchify, unbatchify
from meal.env.mpe import make_mpe_sequence
from meal.wrappers.logging import LogWrapper


@dataclass
class MPEEnvConfig:
    num_agents: int = 3
    num_landmarks: int = 3
    num_obstacles: int = 4
    max_steps: int = 100  # episode length; keep in sync with num_steps
    local_ratio: float = 0.5  # 0=fully global reward, 1=fully local


class MPEAdapter(EnvAdapter):
    checkpoint_subdir = "mpe"

    def build_sequence(self, cfg):
        env_cfg = cfg.env
        envs = make_mpe_sequence(
            sequence_length=cfg.seq_length,
            seed=cfg.seed,
            num_agents=env_cfg.num_agents,
            num_landmarks=env_cfg.num_landmarks,
            num_obstacles=env_cfg.num_obstacles,
            max_steps=env_cfg.max_steps,
            local_ratio=env_cfg.local_ratio,
        )
        for i, env in enumerate(envs):
            envs[i] = LogWrapper(env, replace_info=False)
        return envs

    def env_display_name(self, env) -> str:
        return env.map_id

    def run_name_tag(self, cfg) -> str:
        env_cfg = cfg.env
        return f"{env_cfg.num_agents}a_{env_cfg.num_landmarks}l_{env_cfg.num_obstacles}k"

    def make_eval_fn(self, cl, reset_switch, step_switch, network, agents, num_envs,
                     num_steps, use_cnn, eval_deterministic, seed):
        @jax.jit
        def evaluate_env(cl_state, rng, params, env_idx):
            if eval_deterministic:
                rng = jax.random.PRNGKey(env_idx + seed)
            rng, env_rng = jax.random.split(rng)
            reset_rng = jax.random.split(env_rng, num_envs)
            obs, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(reset_rng)

            total_rewards = jnp.zeros((num_envs,), jnp.float32)
            total_coverage_fraction = jnp.zeros((num_envs,), jnp.float32)
            total_num_covered = jnp.zeros((num_envs,), jnp.float32)

            mask = None
            if isinstance(cl, Packnet):
                mask = cl.get_eval_mask(env_idx, cl_state)

            def one_step(carry, _):
                env_state, obs, rewards, coverage_fraction, num_covered, rng = carry

                obs_batch = batchify(obs, agents, len(agents) * num_envs, not use_cnn)
                if isinstance(cl, Packnet):
                    masked_params = cl.apply_eval_mask(params, mask)
                    pi, _, _ = network.apply(masked_params, obs_batch, env_idx=env_idx)
                else:
                    pi, _, _ = network.apply(params, obs_batch, env_idx=env_idx)
                action = pi.mode()

                env_act = unbatchify(action, agents, num_envs, len(agents))
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, sub = jax.random.split(rng)
                step_rng = jax.random.split(sub, num_envs)
                obs2, env_state2, reward, done, info = jax.vmap(
                    lambda k, s, a: step_switch(k, s, a, jnp.int32(env_idx))
                )(step_rng, env_state, env_act)

                rewards += sum(reward[a] for a in agents)
                coverage_fraction += info.get("coverage_fraction", jnp.zeros((num_envs,)))
                num_covered += info.get("num_covered", jnp.zeros((num_envs,)))

                return (env_state2, obs2, rewards, coverage_fraction, num_covered, rng), None

            (_, _, total_rewards, total_coverage_fraction, total_num_covered, _), _ = jax.lax.scan(
                one_step,
                (env_state, obs, total_rewards, total_coverage_fraction, total_num_covered, rng),
                xs=None,
                length=num_steps,
            )

            return (
                total_rewards.mean(),
                total_coverage_fraction.mean() / num_steps,
                total_num_covered.mean() / num_steps,
            )

        return evaluate_env

    def evaluate_all_envs(self, cl_state, rng, params, num_tasks, evaluate_env):
        env_indices = jnp.arange(num_tasks, dtype=jnp.int32)
        rngs = jax.random.split(rng, num_tasks)
        eval_vmapped = jax.vmap(evaluate_env, in_axes=(None, 0, None, 0))
        return eval_vmapped(cl_state, rngs, params, env_indices)

    def add_eval_metrics(self, metrics, eval_result, env_names):
        avg_rewards, avg_coverage_fraction, avg_num_covered = eval_result
        for i, env_name in enumerate(env_names):
            metrics[f"Evaluation/Returns/{i}_{env_name}"] = avg_rewards[i]
            metrics[f"Evaluation/CoverageFraction/{i}_{env_name}"] = avg_coverage_fraction[i]
            metrics[f"Evaluation/NumCovered/{i}_{env_name}"] = avg_num_covered[i]
        return metrics

    def compute_step_metrics(self, metrics, cfg, info, traj_batch, env_idx, num_agents, agents,
                             current_timestep, rew_shaping_anneal):
        # `metrics` arrives already populated with a blanket tree_map(mean, info) by the
        # caller (matches Overcooked's compute_soup_metrics convention); `info` is still
        # the raw, unaveraged per-step data. coverage_fraction must be averaged only at
        # episode boundaries (mirrors eval logic), so the naive blanket-mean value is
        # wrong and gets discarded/recomputed here.
        coverage_fraction_traj = info.get("coverage_fraction", None)  # (T, E)
        episode_done_traj = info.get("episode_done", None)  # (T, E)
        metrics.pop("coverage_fraction", None)
        metrics.pop("episode_done", None)
        if coverage_fraction_traj is not None and episode_done_traj is not None:
            n_eps = episode_done_traj.sum()
            metrics["coverage_fraction"] = jnp.where(
                n_eps > 0,
                (coverage_fraction_traj * episode_done_traj).sum() / n_eps,
                0.0,
            )
        metrics.pop("terminated", None)
        return metrics
