from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from experiments.algo_common import compute_reward_metrics, compute_soup_metrics
from experiments.envs.base import EnvAdapter
from experiments.evaluation import evaluate_all_envs as _evaluate_all_envs, make_eval_fn as _make_eval_fn
from experiments.utils import add_eval_metrics, add_het_metrics, create_visualizer, rollout_for_video
from meal import make_sequence
from meal.env.overcooked.max_soup_calculator import calculate_max_soup
from meal.wrappers.logging import LogWrapper


@dataclass
class OvercookedEnvConfig:
    env_name: str = "overcooked"  # internal env id; derived from partial_observability below
    difficulty: Optional[str] = "easy"
    curriculum: bool = False  # split the sequence equally across easy/medium/hard, ascending; overrides difficulty
    random_reset: bool = False
    complementary_restrictions: bool = False  # one agent can't pick up onions, other can't pick up plates
    separated_agents: bool = False  # only accept layouts where agents occupy different connected regions
    partial_observability: bool = False  # agents see only a local window instead of the full grid

    # Non-stationarity environment parameters
    sticky_actions: bool = False
    slippery_tiles: bool = False
    random_pot_size: bool = False
    random_cook_time: bool = False
    non_stationary: bool = False  # shortcut: enable all 4 non-stationarity knobs above

    # Reward distribution settings
    sparse_rewards: bool = False  # only shared reward for soup delivery
    individual_rewards: bool = False  # only respective agent gets reward for their actions

    def __post_init__(self):
        if self.partial_observability:
            self.env_name = "overcooked_po"
        if self.non_stationary:
            self.sticky_actions = True
            self.slippery_tiles = True
            self.random_pot_size = True
            self.random_cook_time = True
        if self.sparse_rewards and self.individual_rewards:
            raise ValueError(
                "Cannot enable both sparse_rewards and individual_rewards simultaneously. "
                "Please choose only one reward setting."
            )


class OvercookedAdapter(EnvAdapter):
    checkpoint_subdir = "overcooked"

    def build_sequence(self, cfg):
        env_cfg = cfg.env
        self.max_steps = cfg.max_episode_steps if cfg.max_episode_steps is not None else 400
        envs = make_sequence(
            sequence_length=cfg.seq_length,
            env_id=env_cfg.env_name,
            strategy="curriculum" if env_cfg.curriculum else "generate",
            seed=cfg.seed,
            num_agents=cfg.num_agents if cfg.num_agents is not None else 2,
            max_steps=self.max_steps,
            random_reset=env_cfg.random_reset,
            difficulty=env_cfg.difficulty,
            complementary_restrictions=env_cfg.complementary_restrictions,
            separated_agents=env_cfg.separated_agents,
            sticky_actions=env_cfg.sticky_actions,
            slippery_tiles=env_cfg.slippery_tiles,
            random_pot_size=env_cfg.random_pot_size,
            random_cook_time=env_cfg.random_cook_time,
        )

        max_soup_vals = []
        for i, env in enumerate(envs):
            envs[i] = LogWrapper(env, replace_info=False)
            max_soup_vals.append(
                calculate_max_soup(envs[i].layout, envs[i].max_steps, n_agents=envs[i].num_agents)
            )
        self.max_soup_vals = jnp.asarray(max_soup_vals, dtype=jnp.float32)
        return envs

    def env_display_name(self, env) -> str:
        return env.layout_name

    def trim_to_single_task(self, idx: int):
        self.max_soup_vals = self.max_soup_vals[idx:idx + 1]

    def run_name_tag(self, cfg) -> str:
        return "curriculum" if cfg.env.curriculum else f"{cfg.env.difficulty}"

    def make_eval_fn(self, cl, reset_switch, step_switch, network, agents, num_envs,
                     num_steps, use_cnn, eval_deterministic, seed):
        return _make_eval_fn(cl, reset_switch, step_switch, network, agents, num_envs,
                             num_steps, use_cnn, eval_deterministic, seed)

    def evaluate_all_envs(self, cl_state, rng, params, num_tasks, evaluate_env):
        return _evaluate_all_envs(cl_state, rng, params, num_tasks, evaluate_env)

    def add_eval_metrics(self, metrics, eval_result, env_names):
        avg_rewards, avg_soups, avg_het = eval_result
        metrics = add_eval_metrics(avg_rewards, avg_soups, env_names, self.max_soup_vals, metrics)
        metrics = add_het_metrics(avg_het, env_names, metrics)
        return metrics

    def compute_reward(self, reward, info, agents, cfg, current_timestep, rew_shaping_anneal):
        env_cfg = cfg.env
        if env_cfg.sparse_rewards:
            return reward
        elif env_cfg.individual_rewards:
            return jax.tree_util.tree_map(
                lambda x, y: x + y * rew_shaping_anneal(current_timestep),
                reward, info["shaped_reward"]
            )
        else:
            total_delivery_reward = sum(reward[agent] for agent in agents)
            shared_delivery_rewards = {agent: total_delivery_reward for agent in agents}
            return jax.tree_util.tree_map(
                lambda x, y: x + y * rew_shaping_anneal(current_timestep),
                shared_delivery_rewards, info["shaped_reward"]
            )

    def get_shaped_reward(self, infos, agents):
        return infos.pop("shaped_reward")

    def compute_step_metrics(self, metrics, cfg, info, traj_batch, env_idx, num_agents, agents,
                             current_timestep, rew_shaping_anneal):
        metrics = compute_soup_metrics(
            metrics, cfg, num_agents, agents, info, traj_batch, self.max_soup_vals, env_idx
        )
        metrics = compute_reward_metrics(metrics, agents, current_timestep, rew_shaping_anneal)
        return metrics

    def build_visualizer(self, cfg):
        env_cfg = cfg.env
        cache = {}

        def record_video(rng, train_state, network, env, task_idx, exp_dir):
            if "visualizer" not in cache:
                cache["visualizer"] = create_visualizer(env.num_agents, env_cfg.env_name)
            visualizer = cache["visualizer"]
            states = rollout_for_video(rng, cfg, train_state, env, network, task_idx, cfg.video_length,
                                       env_adapter=self)
            file_path = f"{exp_dir}/task_{task_idx}_{env.layout_name}.mp4"
            visualizer.animate(states, out_path=file_path, task_idx=task_idx, env=env)

        return record_video
