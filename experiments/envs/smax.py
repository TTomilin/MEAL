from dataclasses import dataclass

import jax
import jax.numpy as jnp

from experiments.continual.packnet import Packnet
from experiments.envs.base import EnvAdapter
from experiments.utils import batchify, unbatchify
from meal.env.smax import make_smax_sequence
from meal.wrappers.logging import LogWrapper


@dataclass
class SMAXEnvConfig:
    num_allies: int = 5
    num_enemies: int = 5
    max_steps: int = 100  # episode length; keep in sync with num_steps
    enemy_shoots: bool = True
    partial_observability: bool = False  # not yet implemented for SMAX


class SMAXAdapter(EnvAdapter):
    checkpoint_subdir = "smax"

    def build_sequence(self, cfg):
        env_cfg = cfg.env
        if env_cfg.partial_observability:
            raise NotImplementedError("Partial observability is not implemented for SMAX.")
        envs = make_smax_sequence(
            sequence_length=cfg.seq_length,
            seed=cfg.seed,
            num_allies=env_cfg.num_allies,
            num_enemies=env_cfg.num_enemies,
            max_steps=env_cfg.max_steps,
            enemy_shoots=env_cfg.enemy_shoots,
        )
        for i, env in enumerate(envs):
            envs[i] = LogWrapper(env, replace_info=False)
        return envs

    def env_display_name(self, env) -> str:
        return env.map_id

    def observation_shape(self, env, agents):
        return env.observation_space(agents[0]).shape

    def run_name_tag(self, cfg) -> str:
        env_cfg = cfg.env
        return f"{env_cfg.num_allies}v{env_cfg.num_enemies}"

    def make_eval_fn(self, cl, reset_switch, step_switch, network, agents, num_envs,
                     num_steps, use_cnn, eval_deterministic, seed):
        @jax.jit
        def evaluate_env(cl_state, rng, params, env_idx):
            if eval_deterministic:
                rng = jax.random.PRNGKey(env_idx + seed)
            rng, env_rng = jax.random.split(rng)
            reset_rng = jax.random.split(env_rng, num_envs)
            obs, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(reset_rng)

            total_returns = jnp.zeros((num_envs,), jnp.float32)
            total_kill_fracs = jnp.zeros((num_envs,), jnp.float32)
            num_episodes = jnp.zeros((num_envs,), jnp.float32)

            mask = None
            if isinstance(cl, Packnet):
                mask = cl.get_eval_mask(env_idx, cl_state)

            def one_step(carry, _):
                env_state, obs, returns, kill_fracs, episodes, rng = carry

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

                episode_ended = done.get("__all__", jnp.zeros((num_envs,), jnp.bool_))

                returns += sum(reward[a] for a in agents)
                kf = info.get("kill_fraction", jnp.zeros((num_envs,), jnp.float32))
                kill_fracs += jnp.where(episode_ended, kf, 0.0)
                episodes += episode_ended.astype(jnp.float32)

                return (env_state2, obs2, returns, kill_fracs, episodes, rng), None

            (_, _, total_returns, total_kill_fracs, num_episodes, _), _ = jax.lax.scan(
                one_step,
                (env_state, obs, total_returns, total_kill_fracs, num_episodes, rng),
                xs=None,
                length=num_steps,
            )

            avg_kill_fraction = total_kill_fracs / jnp.maximum(num_episodes, 1.0)
            return total_returns.mean(), avg_kill_fraction.mean()

        return evaluate_env

    def evaluate_all_envs(self, cl_state, rng, params, num_tasks, evaluate_env):
        env_indices = jnp.arange(num_tasks, dtype=jnp.int32)
        rngs = jax.random.split(rng, num_tasks)
        eval_vmapped = jax.vmap(evaluate_env, in_axes=(None, 0, None, 0))
        return eval_vmapped(cl_state, rngs, params, env_indices)

    def add_eval_metrics(self, metrics, eval_result, env_names):
        avg_returns, avg_kill_fraction = eval_result
        for i, env_name in enumerate(env_names):
            metrics[f"Evaluation/Returns/{i}_{env_name}"] = avg_returns[i]
            metrics[f"Evaluation/KillFraction/{i}_{env_name}"] = avg_kill_fraction[i]
        return metrics

    def compute_step_metrics(self, metrics, cfg, info, traj_batch, env_idx, num_agents, agents,
                             current_timestep, rew_shaping_anneal):
        # kill_fraction must be averaged only at episode boundaries (mirrors eval logic).
        # Averaging over all timesteps gives a misleading low value because kill_fraction
        # starts at 0 each episode and grows as enemies die.
        kill_fraction_traj = info.get("kill_fraction", None)  # (T, E)
        episode_done_traj = info.get("episode_done", None)  # (T, E)
        metrics.pop("kill_fraction", None)
        metrics.pop("episode_done", None)
        if kill_fraction_traj is not None and episode_done_traj is not None:
            n_eps = episode_done_traj.sum()
            metrics["kill_fraction"] = jnp.where(
                n_eps > 0,
                (kill_fraction_traj * episode_done_traj).sum() / n_eps,
                0.0,
            )
        metrics.pop("terminated", None)
        return metrics
