"""EnvAdapter: the interface an environment plugs into the algo class hierarchy through.

Per env (Overcooked, MPE, SMAX, JaxNav), an adapter owns everything env-specific:
building the CL task sequence, the evaluation function, per-update-step and per-eval
metrics, in-rollout reward computation (Overcooked shapes/decomposes reward; the other
three pass their native per-agent reward straight through), and optional video rollout.

The algo classes (IPPO/MAPPO/HAPPO/VDN/QMIX) never branch on which env is active --
they only ever call through `self.env_adapter`. An adapter instance is created once per
`AbstractAlgo.run()` and may cache derived, env-specific state (e.g. Overcooked's
per-task max-soup normalization values) as instance attributes set during
`build_sequence`.
"""


class EnvAdapter:
    #: subdirectory under checkpoints/ this env's runs are saved to (matches each
    #: environment's original script: "overcooked", "mpe", "smax", "jaxnav")
    checkpoint_subdir: str = "env"

    def build_sequence(self, cfg):
        """Build the CL task sequence for this env from `cfg.env` (the env-specific
        EnvConfig). Returns a list of (LogWrapper-wrapped) envs. May set additional
        instance attributes (e.g. `self.max_soup_vals`) for later use by
        `compute_step_metrics`/`add_eval_metrics`."""
        raise NotImplementedError

    def trim_to_single_task(self, idx: int):
        """Called (off-policy algos only, matching vdn.py/qmix.py's original
        single_task_idx handling) after `build_sequence` when `cfg.single_task_idx` is
        set, so any per-task adapter-owned state (e.g. Overcooked's `max_soup_vals`)
        stays in sync with the now-length-1 `envs`/`env_names` lists the caller trims
        separately. Default: no-op (most adapters cache no such per-task state)."""
        pass

    def env_display_name(self, env) -> str:
        """Human-readable task name for logging/checkpoint paths (Overcooked:
        `layout_name`; MPE/SMAX/JaxNav: `map_id`)."""
        raise NotImplementedError

    def observation_shape(self, env, agents):
        """`env.observation_space().shape`, normalized across envs whose
        `observation_space()` signature differs: Overcooked takes no `agent` argument
        at all (would error if given one), MPE/JaxNav accept it as optional, SMAX
        requires it (heterogeneous per-agent observation spaces). Default here matches
        Overcooked/MPE/JaxNav's no-arg call; SMAXAdapter overrides it."""
        return env.observation_space().shape

    def make_eval_fn(self, cl, reset_switch, step_switch, network, agents, num_envs,
                     num_steps, use_cnn, eval_deterministic, seed):
        """Returns a jitted `evaluate_env(cl_state, rng, params, env_idx) -> tuple`
        closure. Tuple arity/content is env-specific (rewards+soups+heterogeneity for
        Overcooked; rewards+coverage+num_covered for MPE; rewards+kill_fraction for
        SMAX; rewards+success for JaxNav) -- `add_eval_metrics` below is what knows how
        to unpack it."""
        raise NotImplementedError

    def evaluate_all_envs(self, cl_state, rng, params, num_tasks, evaluate_env):
        """vmaps `evaluate_env` over all tasks. Returns the same tuple shape as
        `evaluate_env`, each element now array-valued over tasks."""
        raise NotImplementedError

    def add_eval_metrics(self, metrics, eval_result, env_names):
        """Unpacks `evaluate_all_envs`'s return tuple into `Evaluation/...` metric
        dict entries, one set per task plus any aggregate keys."""
        raise NotImplementedError

    def compute_reward(self, reward, info, agents, cfg, current_timestep, rew_shaping_anneal):
        """Called once per env-step inside trajectory collection (on-policy algos:
        IPPO/MAPPO/HAPPO), before the reward is stored in the Transition. Default:
        identity passthrough (MPE/SMAX/JaxNav all use their native per-agent reward
        unchanged). Overcooked overrides this with its sparse/individual/shared+shaped
        decomposition (annealed via `rew_shaping_anneal(current_timestep)`)."""
        return reward

    def get_shaped_reward(self, infos, agents):
        """Pops and returns the env's per-agent shaped-reward dict from `infos`, for
        off-policy algos (VDN/QMIX) to add unconditionally to the sparse env reward --
        off-policy never annealed this the way `compute_reward` does for on-policy.
        Returns None if this env doesn't provide shaped rewards (MPE/SMAX/JaxNav)."""
        return None

    def compute_step_metrics(self, metrics, cfg, info, traj_batch, env_idx, num_agents, agents,
                             current_timestep, rew_shaping_anneal):
        """Called once per training update-step (after the epoch/minibatch loop).

        Calling convention: `metrics` arrives already populated with a blanket
        `jax.tree_util.tree_map(lambda x: x.mean(), info)` done by the caller; `info`
        is still the raw, unaveraged per-step data (shape (num_steps, num_envs, ...)).
        Adapters needing a non-naive reduction (e.g. episode-boundary-only averaging)
        pop/overwrite the relevant key(s) in `metrics`, computed from raw `info`
        (Overcooked's "Soup/*" via `algo_common.compute_soup_metrics`; MPE's
        "coverage_fraction"; SMAX's "kill_fraction"; JaxNav adds nothing beyond
        removing the stray "terminated" blanket-mean key). Mutates and returns
        `metrics`."""
        raise NotImplementedError

    def build_visualizer(self, cfg):
        """Returns a callable `rollout_and_record(rng, state, env, task_idx, exp_dir) ->
        None` that records a video for the given task, or `None` if this env doesn't
        support video recording (matches today: MPE/SMAX have `record_video` as a dead
        config field, never implemented)."""
        return None
