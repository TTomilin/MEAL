"""OffPolicyAlgo: shared pieces for VDN / QMIX.

Both use CTRolloutManager + epsilon-greedy collection, both disallow packnet, both
compute `num_actors`/`num_updates` and env-sequence setup the same way. What differs
(TD-target computation: plain sum for VDN vs a learned MixingNetwork for QMIX; the
AGEM/ER-ACE memory-gradient helpers, which take different arguments per algorithm --
`compute_vdn_memory_gradient` vs `compute_qmix_memory_gradient`) stays in each concrete
class, matching the boundary already used for the on-policy algorithms.
"""
import json
import os
from dataclasses import dataclass

import flax

from experiments.algo_common import convert_frozen_dict
from experiments.algos.base import AbstractAlgo, BaseConfig


@dataclass
class OffPolicyConfig(BaseConfig):
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    steps_per_task: float = 1e8
    num_envs: int = 2048
    num_steps: int = 400
    hidden_size: int = 256
    eps_start: float = 1.0
    eps_finish: float = 0.05
    eps_decay: float = 0.1  # fraction of num_updates over which eps decays
    max_grad_norm: float = 1.0
    update_epochs: int = 8  # passes over collected data per update
    num_minibatches: int = 16  # minibatches per epoch
    lr: float = 1e-3
    anneal_lr: bool = False
    gamma: float = 0.99
    tau: float = 1.0  # target network update rate (1 = hard copy)
    target_update_interval: int = 1


class OffPolicyAlgo(AbstractAlgo):
    def single_task_cl_method(self) -> str:
        return "FT"  # matches original vdn.py's exact casing (qmix.py used "ft")

    def build_cl_method(self):
        cfg = self.cfg
        method_map = self._method_map()
        cl = method_map[cfg.cl_method.lower()]
        if cfg.cl_method.lower() == "packnet":
            raise ValueError(
                f"Packnet is not supported for {cfg.alg_name.upper()} (value-based method)."
            )
        return cl

    def _method_map(self):
        raise NotImplementedError

    def setup_envs(self):
        super().setup_envs()
        cfg = self.cfg
        # Single-task baseline: trim to one env
        if cfg.single_task_idx is not None:
            idx = cfg.single_task_idx
            self.envs = [self.envs[idx]]
            self.env_names = [self.env_names[idx]]
            self.env_adapter.trim_to_single_task(idx)
            cfg.seq_length = 1
            self.temp_env = self.envs[0]
            self.num_agents = self.temp_env.num_agents
            self.agents = self.temp_env.agents

        cfg.num_actors = self.num_agents * cfg.num_envs
        cfg.num_updates = int(cfg.steps_per_task // cfg.num_steps // cfg.num_envs)

    def save_params(self, path, train_state, env_kwargs=None, layout_name=None, config=None, extra_fields=None):
        """Identical between vdn.py and qmix.py except QMIX adds `mixing_embed_dim` to
        the saved metadata -- passed in via `extra_fields`."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(flax.serialization.to_bytes({"params": train_state.params}))

        meta = {}
        if env_kwargs is not None:
            meta["env_kwargs"] = convert_frozen_dict(env_kwargs)
        if layout_name is not None:
            meta["layout_name"] = layout_name
        if config is not None:
            meta.update({
                "use_cnn": config.use_cnn,
                "num_tasks": self.cfg.seq_length,
                "use_multihead": config.use_multihead,
                "use_task_id": config.use_task_id,
                "use_layer_norm": config.use_layer_norm,
                "activation": config.activation,
                "seed": config.seed,
            })
            if extra_fields:
                meta.update(extra_fields)
        if meta:
            with open(f"{path}_config.json", "w") as f:
                json.dump(convert_frozen_dict(meta), f, indent=2)
        print("model saved to", path)
