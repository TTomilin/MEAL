from dataclasses import dataclass

import jax
import jax.numpy as jnp
import wandb

from experiments.continual.packnet import Packnet
from experiments.envs.base import EnvAdapter
from experiments.utils import batchify, unbatchify
from meal.env.jaxnav import JaxNav, JaxNavVisualizer
from meal.wrappers.logging import LogWrapper


@dataclass
class JaxNavEnvConfig:
    map_dim: int = 7
    partial_observability: bool = False  # not yet implemented for JaxNav


def make_jaxnav_sequence(sequence_length: int, seed: int, num_agents: int, max_steps: int, map_dim: int):
    """JaxNav CL sequence: same dynamics, random Grid-Rand-Poly maps per task."""
    key = jax.random.PRNGKey(seed)
    envs = []
    for _ in range(sequence_length):
        env = JaxNav(
            num_agents=num_agents,
            act_type="Discrete",
            max_steps=max_steps,
            map_id="Grid-Rand-Poly",
            map_params={"map_size": (map_dim, map_dim)},
        )
        key, k_layout = jax.random.split(key)
        env.map_obj._fixed_map = env.map_obj.sample_map(k_layout)
        envs.append(env)
    return envs


def rollout_jaxnav_trained(rng, cfg, train_state, network, env, task_idx, num_steps, env_adapter):
    """Trained-policy rollout for JaxNav video, matching the output format
    JaxNavVisualizer expects. Deterministic pi.mode() action selection, matching
    JaxNavAdapter.make_eval_fn's own action-selection convention."""
    rng, reset_key = jax.random.split(rng)

    obs, state = env.reset(reset_key)
    obs_seq = [obs]
    state_seq = [state]
    reward_seq = []

    num_agents = env.num_agents
    agents = env.agents
    done_frames = jnp.full((num_agents,), num_steps - 1, dtype=jnp.int32)
    expected_shape = env_adapter.observation_shape(env, agents)

    for t in range(num_steps):
        actions = {}
        for agent_id in agents:
            obs_v = obs[agent_id]
            obs_b = jnp.expand_dims(obs_v, axis=0) if obs_v.ndim == len(expected_shape) else obs_v
            if not cfg.use_cnn:
                obs_b = jnp.reshape(obs_b, (obs_b.shape[0], -1))
            pi, _, _ = network.apply(train_state.params, obs_b, env_idx=task_idx)
            actions[agent_id] = jnp.squeeze(pi.mode(), axis=0)

        rng, step_key = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_key, state, actions)

        obs_seq.append(obs)
        state_seq.append(state)

        if isinstance(reward, dict):
            r = 0.0
            for v in reward.values():
                r += float(v)
        else:
            r = float(jnp.sum(jnp.asarray(reward)))
        reward_seq.append(r)

        if isinstance(done, dict):
            if "__all__" in done:
                ep_done_scalar = bool(done["__all__"])
                done_arr = jnp.full((env.num_agents,), ep_done_scalar)
            else:
                done_arr = jnp.array([bool(done[a]) for a in env.agents], dtype=jnp.bool_)
        else:
            done_arr = jnp.asarray(done, jnp.bool_)
            if done_arr.ndim == 0:
                done_arr = jnp.full((env.num_agents,), done_arr)

        mask_new = (done_arr & (done_frames == (num_steps - 1)))
        done_frames = jnp.where(mask_new, jnp.full_like(done_frames, t), done_frames)

    return obs_seq, state_seq, reward_seq, done_frames


class JaxNavAdapter(EnvAdapter):
    checkpoint_subdir = "jaxnav"

    def build_sequence(self, cfg):
        env_cfg = cfg.env
        if env_cfg.partial_observability:
            raise NotImplementedError("Partial observability is not implemented for JaxNav.")
        self.max_steps = cfg.max_episode_steps if cfg.max_episode_steps is not None else 400
        envs = make_jaxnav_sequence(
            sequence_length=cfg.seq_length,
            seed=cfg.seed,
            num_agents=cfg.num_agents if cfg.num_agents is not None else 2,
            max_steps=self.max_steps,
            map_dim=env_cfg.map_dim,
        )
        for i, env in enumerate(envs):
            envs[i] = LogWrapper(env, replace_info=False)
        return envs

    def env_display_name(self, env) -> str:
        return env.map_id

    def run_name_tag(self, cfg) -> str:
        return f"mapdim{cfg.env.map_dim}"

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
            total_goals_reached = jnp.zeros((num_envs,), jnp.float32)
            total_episodes = jnp.zeros((num_envs,), jnp.float32)

            mask = None
            if isinstance(cl, Packnet):
                mask = cl.get_eval_mask(env_idx, cl_state)

            def one_step(carry, _):
                env_state, obs, rewards, goals_reached, episodes, rng = carry

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
                goals_reached += info["GoalR"]
                episodes += info["NumC"]

                return (env_state2, obs2, rewards, goals_reached, episodes, rng), None

            (env_state, obs, total_rewards, total_goals_reached, total_episodes, rng), _ = jax.lax.scan(
                one_step,
                (env_state, obs, total_rewards, total_goals_reached, total_episodes, rng),
                xs=None,
                length=num_steps,
            )

            success_per_env = jnp.where(total_episodes > 0, total_goals_reached / total_episodes, 0.0)
            return total_rewards.mean(), success_per_env.mean()

        return evaluate_env

    def evaluate_all_envs(self, cl_state, rng, params, num_tasks, evaluate_env):
        env_indices = jnp.arange(num_tasks, dtype=jnp.int32)
        rngs = jax.random.split(rng, num_tasks)
        eval_vmapped = jax.vmap(evaluate_env, in_axes=(None, 0, None, 0))
        return eval_vmapped(cl_state, rngs, params, env_indices)

    def add_eval_metrics(self, metrics, eval_result, env_names):
        avg_rewards, avg_success = eval_result
        for i, env_name in enumerate(env_names):
            metrics[f"Evaluation/Returns/{i}_{env_name}"] = avg_rewards[i]
            metrics[f"Evaluation/Success/{i}_{env_name}"] = avg_success[i]
        return metrics

    def compute_step_metrics(self, metrics, cfg, info, traj_batch, env_idx, num_agents, agents,
                             current_timestep, rew_shaping_anneal):
        metrics.pop("terminated", None)
        return metrics

    def build_visualizer(self, cfg):
        def record_video(rng, train_state, network, env, task_idx, exp_dir):
            raw_env = env._env if isinstance(env, LogWrapper) else env
            obs_seq, state_seq, reward_seq, done_frames = rollout_jaxnav_trained(
                rng, cfg, train_state, network, raw_env, task_idx, cfg.video_length, self,
            )
            task_name = f"task_{task_idx}_{raw_env.map_id}"
            visualizer = JaxNavVisualizer(
                env=raw_env, obs_seq=obs_seq, state_seq=state_seq,
                reward_seq=reward_seq, done_frames=done_frames, title_text=task_name,
            )
            file_path = f"{exp_dir}/{task_name}.mp4"
            visualizer.animate(save_fname=file_path, view=False)
            if wandb.run is not None:
                wandb.log({f"task_{task_idx}": wandb.Video(file_path, format="mp4")})

        return record_video
