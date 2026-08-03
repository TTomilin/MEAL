"""
AbstractAlgo: shared config fields and orchestration for all training algorithms
(IPPO, MAPPO, HAPPO, VDN, QMIX) across all environments (Overcooked, MPE, SMAX, JaxNav).

This is a Template Method organization, not a stateful OOP rewrite: JAX requires
train_state/cl_state pytrees to be threaded explicitly through jax.jit/jax.lax.scan
boundaries, so the actual jitted training functions are still built as closures
(via each subclass's `build_train_on_environment`), exactly as in the original
standalone scripts. `self` holds the same Python-level config/objects (cfg, envs,
agents, ...) those closures used to capture directly from `main()`'s locals.

Env selection is a second, orthogonal axis: `cfg.env` is one of the EnvConfig
dataclasses in `experiments/envs/`, and `self.env_adapter` (built from it in
`setup_envs`) is what every env-touching piece of the algo classes goes through --
never a hardcoded env-specific call. See `experiments/envs/base.py` for the adapter
interface.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import jax

from experiments.algo_common import init_wandb_and_tensorboard, resolve_reg_coef
from experiments.envs import EnvConfig, build_env_adapter
from experiments.envs.overcooked import OvercookedEnvConfig


@dataclass
class BaseConfig:
    # ═══════════════════════════════════════════════════════════════════════════
    # ENVIRONMENT SELECTION
    # ═══════════════════════════════════════════════════════════════════════════
    env: EnvConfig = field(default_factory=OvercookedEnvConfig)

    # ═══════════════════════════════════════════════════════════════════════════
    # NETWORK ARCHITECTURE PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    activation: str = "relu"
    use_cnn: bool = False
    use_layer_norm: bool = True

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTINUAL LEARNING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    cl_method: Optional[str] = None
    reg_coef: Optional[float] = None
    use_task_id: bool = True
    use_multihead: bool = True
    normalize_importance: bool = False

    importance_episodes: int = 5
    importance_stride: int = 5  # compute and accumulate importance once every N steps
    importance_steps: int = 500
    importance_mode: str = "online"  # "online", "last" or "multi", only for EWC & MAS
    importance_decay: float = 0.9  # Only for online EWC & MAS

    agem_memory_size: int = 100000
    agem_sample_size: int = 1024
    er_ace_coef: float = 1.0

    # Packnet specific parameters
    train_epochs: int = 8
    finetune_epochs: int = 2
    finetune_timesteps: float = 1e7

    # ═══════════════════════════════════════════════════════════════════════════
    # SEQUENCE / TASK PARAMETERS (universal across envs; per-env fields live on cfg.env)
    # ═══════════════════════════════════════════════════════════════════════════
    seq_length: int = 20
    single_task_idx: Optional[int] = None

    # ═══════════════════════════════════════════════════════════════════════════
    # EVALUATION PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    evaluation: bool = True
    record_video: bool = False
    video_length: int = 250
    log_interval: int = 5
    eval_deterministic: bool = False

    # ═══════════════════════════════════════════════════════════════════════════
    # LOGGING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    use_wandb: bool = True
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    entity: Optional[str] = ""
    project: str = "MEAL"
    tags: List[str] = field(default_factory=list)

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    seed: int = 1

    # ═══════════════════════════════════════════════════════════════════════════
    # RUNTIME COMPUTED PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    num_actors: int = 0
    num_updates: int = 0


class AbstractAlgo:
    """Shared setup/orchestration. Subclasses (OnPolicyAlgo, OffPolicyAlgo) fill in
    the abstract hooks; concrete algo classes (IPPO, MAPPO, ...) fill in the rest."""

    def __init__(self, cfg: BaseConfig):
        jax.config.update("jax_platform_name", "gpu")
        print("Device:", jax.devices())

        if cfg.single_task_idx is not None:
            cfg.cl_method = self.single_task_cl_method()
        if cfg.cl_method is None:
            raise ValueError(
                "cl_method is required. Please specify a continual learning method "
                "(e.g., ewc, mas, l2, ft, agem, packnet, er_ace)."
            )

        resolve_reg_coef(cfg)

        self.cfg = cfg
        self.rng = jax.random.PRNGKey(cfg.seed)
        self.cl = self.build_cl_method()

    def single_task_cl_method(self) -> str:
        """cl_method to force when cfg.single_task_idx is set. Since
        `cl.method_map[cfg.cl_method.lower()]` lookups are always lowercased
        anyway, this casing never changed behavior, so subclasses only need
        to override this if they want to preserve a specific log/display casing."""
        return "ft"

    def build_cl_method(self):
        raise NotImplementedError

    def setup_envs(self):
        cfg = self.cfg
        self.env_adapter = build_env_adapter(cfg.env)
        envs = self.env_adapter.build_sequence(cfg)

        self.envs = envs
        self.env_names = [self.env_adapter.env_display_name(e) for e in envs]
        self.temp_env = envs[0]
        self.num_agents = self.temp_env.num_agents
        self.agents = self.temp_env.agents

    def setup_logging(self):
        cfg = self.cfg
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
        network_arch = "cnn" if cfg.use_cnn else "mlp"
        env_tag = self.env_adapter.run_name_tag(cfg)
        self.run_name = (
            f'{cfg.alg_name}_{cfg.cl_method}_{env_tag}_{self.num_agents}agents_'
            f'{network_arch}_seq{cfg.seq_length}_seed_{cfg.seed}_{timestamp}'
        )
        self.exp_dir = os.path.join("runs", self.run_name)
        self.writer = init_wandb_and_tensorboard(cfg, self.run_name, self.exp_dir)

    def checkpoint_path(self, task_idx: int) -> str:
        repo_root = Path(__file__).resolve().parent.parent.parent
        return (f"{repo_root}/checkpoints/{self.env_adapter.checkpoint_subdir}/"
                f"{self.cfg.cl_method}/{self.run_name}/model_env_{task_idx + 1}")

    def run(self):
        raise NotImplementedError
