'''Main entry point for running teammate generation algorithms.'''
import copy
import json
import logging
import os
import pickle
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import wandb

from experiments.model.cnn import ActorCritic as CNNActorCritic
from experiments.model.mlp import ActorCritic as MLPActorCritic
from experiments.utils import init_cl_state, create_visualizer
from experiments.continual.agem import AGEM, init_agem_memory
from experiments.continual.er_ace import ERACE
from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.l2 import L2
from experiments.continual.mas import MAS
from meal.env.overcooked.layouts.presets import overcooked_layouts
from meal.env.overcooked.max_soup_calculator import calculate_max_soup
from meal import make_env
from meal.wrappers.logging import LogWrapper
from experiments.partner_adaptation.partner_agents.agent_interface import MLPActorCriticPolicyCL
from experiments.partner_adaptation.partner_agents.overcooked.agent_policy_wrappers import OvercookedIndependentPolicyWrapper, \
    OvercookedOnionPolicyWrapper, OvercookedPlatePolicyWrapper, OvercookedRandomPolicyWrapper, \
    OvercookedStaticPolicyWrapper
from experiments.partner_adaptation.partner_bank import load_partners, partner_identity, select_partner_bank
from experiments.partner_adaptation.run_outputs import (
    RunRecorder, RunStatus, allocate_run_dir, atomic_write, build_fingerprint, check_resume, code_revision,
    csv_has_agent_stats, device_info, read_checkpoint, resolve_resume, restore_state, save_checkpoint, truncate_csv,
    write_json)
from experiments.partner_adaptation.partner_generation.utils import frozendict_from_layout_repr
from experiments.partner_adaptation.train_br import (
    DummyPolicyPopulation, HeuristicPolicyPopulation, make_partner_population, rollout_ego_with_partner, run_br_training)
from experiments.partner_adaptation.train_ego import ACTION_NAMES, EVAL_SCHEDULES, eval_plan, make_optimizer

log = logging.getLogger(__name__)

HEURISTIC_NAMES = ["Independent_Policy", "Onion_Policy", "Plate_Policy", "Random_Policy", "Static_Policy"]
IMPORTANCE_METHODS = ("ewc", "mas", "l2")
SCHEDULES = ("sequential", "interleaved")


@dataclass
class TrainConfig:
    # Wandb and other logging
    project: str = "MEAL"
    mode: str = "online"  # Literal["online", "offline", "disabled"]
    group: str = "overcooked"
    entity: str = ""
    tags: List[str] = field(default_factory=list)
    checkpoint_path: str = "checkpoints"
    checkpoint_freq: int = 50  # Checkpoint every N updates
    save_dir: str = ""  # Set programmatically: the unique run directory created under `checkpoint_path`
    # Resume a run from its last completed partner: a run directory (or its latest.ckpt). Pass the original
    # arguments as well; anything that differs from the saved run is rejected (see run_outputs.py).
    resume: str = ""
    save_checkpoints: bool = True  # Replace <run dir>/latest.ckpt after every completed partner

    # MEAL
    # Pregenerated MEAL layouts that we are interested in.
    layouts_path: str = "meal/env/layouts/"

    # Overcooked
    env_name: str = "overcooked"
    layout_difficulty: str = "easy"
    layout_idx: int = 0
    layout_name: str = ""  # If specified, overrides layout_idx

    reward_shaping_horizon: float = 2.5e7
    num_agents: int = 2

    # best_response
    alg: str = "br"

    # Actor-Critic
    fc_dim_size: int = 256
    gru_hidden_dim: int = 256

    seed: int = 0
    num_checkpoints: int = 20

    # Training
    lr: float = 1e-3
    anneal_lr: bool = False
    num_envs: int = 2048
    num_steps: int = 400
    total_timesteps: float = 1e8
    update_epochs: int = 8
    num_minibatches: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0

    # ═══════════════════════════════════════════════════════════════════════════
    # NETWORK ARCHITECTURE PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    activation: str = "relu"
    use_cnn: bool = False
    use_layer_norm: bool = True
    hidden_size: int = 128
    num_layers: int = 2

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTINUAL LEARNING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    cl_method: Optional[str] = None
    reg_coef: Optional[float] = None
    use_task_id: bool = True
    use_multihead: bool = True
    shared_backbone: bool = False
    normalize_importance: bool = False
    regularize_critic: bool = False

    # Importance / regularization parameters (EWC, MAS, L2)
    importance_mode: str = "online"   # "online", "last", or "multi"
    importance_decay: float = 0.9
    importance_episodes: int = 5
    importance_steps: int = 500
    importance_stride: int = 5

    # AGEM / ER-ACE specific parameters
    agem_memory_size: int = 100000
    agem_sample_size: int = 1024
    er_ace_coef: float = 1.0

    # Eval
    num_eval_episodes: int = 5
    eval_every: int = 2         # Run eval every N update steps (1 = every step); used by eval_schedule="all"
    # "all": every evaluation partner after update 1, every `eval_every` updates and after the last (original).
    # "pilot": every partner before the first stage and after each stage (stage end = next stage's start), plus
    # the partner being trained every `eval_current_every` updates (0 = never). Events never overlap.
    eval_schedule: str = "all"
    eval_current_every: int = 5
    # Profiling only: train just these stage indices (heads, evaluation partners and schedules stay those of the
    # whole bank). Empty = every stage. Not resumable, not for results.
    only_stages: List[int] = field(default_factory=list)
    # "sequential": partner after partner (the CPA protocol). 
    # "interleaved": joint-training upper bound with the same total budget: 
    # the partners are cycled in micro-stages of `interleave_updates` updates until every partner has
    # had num_updates updates; the optimizer state and the reward-shaping anneal (horizon counted over the whole
    # run) carry across micro-stages, and every partner is evaluated before training and after every cycle.
    # Plain fine-tuning only (no CL state), constant learning rate.
    schedule: str = "sequential"
    interleave_updates: int = 2
    # After each stage, record one episode of the ego with the frozen partner it was just trained with and upload it
    # to W&B as Video/stage_<k>_<partner> (files under <run dir>/videos/).
    record_video: bool = True
    gif_len: int = 100          # Steps recorded per video (num_steps = one full episode)

    log_train_out: bool = True

    # Record per-agent diagnostics (ego vs partner soups, raw shaped return, action counts) in the CSVs and W&B.
    # Recording only: it changes neither the policy, the learning objective nor any random draw, and is not part of
    # the resume fingerprint. A run whose CSVs predate the diagnostics resumes without them (see README).
    log_agent_stats: bool = True

    # Partner/Task Configuration
    # Population partners come from the bundled BRDiv population of `layout_name` (first N members, default 3;
    # asking for more than exist is an error) or, with `partner_bank`, from that JSON manifest
    # (see partner_bank.py), which must declare exactly N partners; N then defaults to the bank size.
    num_population_partners: Optional[int] = None
    partner_bank: str = ""
    # Legacy heuristic partners (indp, onin, plate, rndm, static) trained after the bank. Default: 5 without
    # `partner_bank`, 0 with one (a manifest is the complete, fixed list of partners).
    num_heuristic_partners: Optional[int] = None

    def __post_init__(self):
        ### MEAL ###

        if self.layout_difficulty == "medium":
            self.layouts_path = self.layouts_path + "gen_20_medium.json"
        elif self.layout_difficulty == "easy":
            self.layouts_path = self.layouts_path + "gen_20_easy.json"
        elif self.layout_difficulty == "hard":
            self.layouts_path = self.layouts_path + "gen_20_hard.json"

        self.num_actors = 2 * self.num_envs
        self.num_controlled_actors = self.num_envs
        self.num_uncontrolled_actors = self.num_envs
        self.num_updates = self.total_timesteps // self.num_envs // self.num_steps

        # Hardcoded for Overcooked
        self.num_actions = 6

        self.minibatch_size = (
                                      self.num_controlled_actors * self.num_steps) // self.num_minibatches

        #############
        print("Number of updates: ", self.num_updates)
        self.num_checkpoints = 1  # Only final params are used; avoid 122× param memory in scan carry


def read_layouts(config):
    with open(config.layouts_path, "r") as f:
        layouts = json.load(f)
    return layouts


def get_run_string(config: TrainConfig):
    cl_method = config.cl_method if config.cl_method is not None else "none"
    layout = config.layout_name if config.layout_name else f"layout{config.layout_idx}"
    network = "cnn" if config.use_cnn else "mlp"
    # Runs with the default identity access (identity input and head routing) keep their historical names; any other
    # condition says so, which also names its W&B run.
    identity = "" if config.use_task_id and config.use_multihead else (
        f"_{'id' if config.use_task_id else 'noid'}-{'mh' if config.use_multihead else '1h'}")
    return (
        f"br_{cl_method}_{config.layout_difficulty}_{layout}"
        f"_{network}_pop{config.num_population_partners}"
        f"_heur{config.num_heuristic_partners}_seed{config.seed}{identity}"
    )


def validate_config(config: TrainConfig):
    """Reject settings that would silently train or evaluate something other than what was asked for."""
    problems = []
    update_size = config.num_envs * config.num_steps
    if config.num_envs < 1 or config.num_steps < 1:
        problems.append("num_envs and num_steps must be positive")
    elif int(config.num_updates) < 1:
        problems.append(f"total_timesteps={config.total_timesteps:g} is less than one update "
                        f"({update_size} joint environment steps = num_envs x num_steps); the run would not train")
    if config.num_minibatches < 1 or update_size % max(config.num_minibatches, 1):
        problems.append(f"num_envs x num_steps = {update_size} is not divisible by "
                        f"num_minibatches={config.num_minibatches}; samples would be dropped")
    if config.update_epochs < 1:
        problems.append("update_epochs must be positive")
    if config.num_eval_episodes < 1:
        problems.append("num_eval_episodes must be at least 1")
    if config.eval_schedule not in EVAL_SCHEDULES:
        problems.append(f"eval_schedule must be one of {EVAL_SCHEDULES}, got {config.eval_schedule!r}")
    if config.eval_schedule == "all" and config.eval_every < 1:
        problems.append("eval_every must be at least 1")
    if config.eval_current_every < 0:
        problems.append("eval_current_every must be >= 0 (0 disables current-partner evaluation)")
    if config.cl_method and config.cl_method.lower() in IMPORTANCE_METHODS and (
            config.importance_episodes < 1 or config.importance_steps < 1):
        problems.append("importance_episodes and importance_steps must be positive")
    if config.schedule not in SCHEDULES:
        problems.append(f"schedule must be one of {SCHEDULES}, got {config.schedule!r}")
    elif config.schedule == "interleaved":
        if config.cl_method and config.cl_method.lower() != "ft":
            problems.append(f"schedule=interleaved is the joint-training upper bound and supports cl_method ft (or "
                            f"none) only, got {config.cl_method!r}")
        if config.interleave_updates < 1 or int(config.num_updates) % config.interleave_updates:
            problems.append(f"interleave_updates={config.interleave_updates} must be positive and divide the "
                            f"{int(config.num_updates)} updates per partner")
        if config.anneal_lr:
            problems.append("schedule=interleaved needs a constant learning rate (anneal_lr would restart at every "
                            "micro-stage)")
        if config.only_stages:
            problems.append("only_stages (profiling) is not supported with schedule=interleaved")
    elif config.reward_shaping_horizon > config.total_timesteps:
        log.warning("reward_shaping_horizon=%g exceeds the %g joint steps per partner: the shaped reward never "
                    "reaches 0 within a stage", config.reward_shaping_horizon, config.total_timesteps)
    if problems:
        raise ValueError("invalid configuration:\n  " + "\n  ".join(problems))


class Resolved(NamedTuple):
    bank: Any
    cl: Any
    identities: list
    labels: list
    fingerprint: dict
    num_heuristics: int


def resolve_run(config: TrainConfig) -> Resolved:
    """Fill in derived settings of `config` (in place) and return the partner list and resume fingerprint.

    Shared by `execute` and by the launcher, so a planned job and the run it produces are compared on the
    same identity.
    """
    bank = select_partner_bank(config.layout_name, config.partner_bank, config.num_population_partners)
    config.num_population_partners = len(bank)
    if config.num_heuristic_partners is None:
        config.num_heuristic_partners = 0 if config.partner_bank else 5
    validate_config(config)
    cl = build_cl_method(config)
    num_heuristics = min(config.num_heuristic_partners, len(HEURISTIC_NAMES))
    identities = [partner_identity(r) for r in bank.records] + [
        dict(partner_id=len(bank) + i, kind="heuristic", label=HEURISTIC_NAMES[i]) for i in range(num_heuristics)]
    labels = [i["label"] for i in identities]
    config_dict = asdict(config)
    if config.schedule == "sequential":  # sequential runs keep the fingerprint they had before schedules existed
        config_dict.pop("schedule"), config_dict.pop("interleave_updates")
    elif config.reward_shaping_horizon > len(identities) * config.total_timesteps:
        raise ValueError(f"invalid configuration:\n  reward_shaping_horizon={config.reward_shaping_horizon:g} exceeds "
                         f"the whole interleaved run ({len(identities)} partners x {config.total_timesteps:g} "
                         f"joint steps); the horizon counts over the run in this schedule")
    return Resolved(bank, cl, identities, labels, build_fingerprint(config_dict, identities), num_heuristics)


class Stage(NamedTuple):
    """One partner of the training sequence; its index in the sequence is the ego head / evaluation id."""
    identity: dict
    policy: Any
    params: Any
    agent_config: dict


class RunResult(NamedTuple):
    run_dir: Path
    ego_params: Any
    cl_state: Any
    counters: dict


def build_cl_method(config):
    if config.cl_method is None:
        return None
    # Set default regularization coefficient based on the CL method if not specified
    if config.reg_coef is None:
        if config.cl_method.lower() == "ewc":
            config.reg_coef = 1e11
        elif config.cl_method.lower() == "mas":
            config.reg_coef = 1e9
        elif config.cl_method.lower() == "l2":
            config.reg_coef = 1e7

    method_map = dict(
        ewc=EWC(mode=config.importance_mode, decay=config.importance_decay),
        mas=MAS(mode=config.importance_mode, decay=config.importance_decay),
        l2=L2(),
        ft=FT(),
        agem=AGEM(memory_size=config.agem_memory_size, sample_size=config.agem_sample_size),
        er_ace=ERACE(memory_size=config.agem_memory_size, sample_size=config.agem_sample_size),
    )
    if config.cl_method.lower() not in method_map:
        raise ValueError(f"Unknown continual learning method: {config.cl_method}")
    print(f"Initialized continual learning method: {config.cl_method.upper()}")
    return method_map[config.cl_method.lower()]


def schedule_description(config, labels):
    """What is, and is not, carried from one partner to the next (read from train_ego.py)."""
    spu = int(config.num_envs) * int(config.num_steps)
    return {
        "stage_order": labels,
        "stage_budget": f"every partner is trained for {int(config.num_updates)} updates ({int(config.num_updates) * spu} "
                        f"joint environment steps); the budget is per partner, not cumulative",
        "optimizer": "Adam with global-norm clipping, re-created at every partner: moments and step count reset "
                     "at each boundary, so a boundary resume starts from a fresh optimizer by design",
        "learning_rate": ("linear anneal over the updates of each partner, restarting at every partner"
                          if config.anneal_lr else "constant"),
        "reward_shaping": f"linear 1 -> 0 over {config.reward_shaping_horizon:g} joint steps counted from the "
                          f"start of every partner",
        "carried_across_partners": "ego parameters and the continual-learning state only",
        "continual_learning_update": ("after every partner: importance rollouts with the frozen partner acting"
                                      if config.cl_method and config.cl_method.lower() in IMPORTANCE_METHODS else
                                      "memory updated inside training" if config.cl_method else "none"),
        "rng": "train, evaluation and importance keys are derived from (seed, partner index); no RNG state is carried",
        "evaluation": (f"{config.num_eval_episodes} episodes per partner: every partner before stage 0 and after "
                       f"every stage; the current partner every {config.eval_current_every} updates"
                       if config.eval_schedule == "pilot" else
                       f"every {config.eval_every} updates, after update 1 and after the last, "
                       f"{config.num_eval_episodes} episodes against every evaluation partner"),
    } if config.schedule != "interleaved" else {
        "stage_order": labels,
        "stage_budget": f"joint-training upper bound: the partners are cycled in micro-stages of "
                        f"{config.interleave_updates} updates for {int(config.num_updates) // config.interleave_updates} "
                        f"cycles, so every partner gets {int(config.num_updates)} updates "
                        f"({int(config.num_updates) * spu} joint environment steps), as in the sequential runs",
        "optimizer": "Adam with global-norm clipping, created once and carried across micro-stages (and checkpoints)",
        "learning_rate": "constant",
        "reward_shaping": f"linear 1 -> 0 over {config.reward_shaping_horizon:g} joint steps counted over the whole run",
        "carried_across_partners": "ego parameters and optimizer state",
        "continual_learning_update": "none (plain fine-tuning)",
        "rng": "train and evaluation keys are derived from (seed, micro-stage index); no RNG state is carried",
        "evaluation": (f"{config.num_eval_episodes} episodes per partner: every partner before training"
                       f"{'' if config.eval_schedule == 'pilot' else ' (pilot schedule only)'} and after every "
                       f"cycle; no current-partner evaluations"),
    }


def execute(config: TrainConfig) -> RunResult:
    """Train the ego agent partner by partner and record the run (see run_outputs.py)."""
    bank, cl, identities, labels, fingerprint, num_heuristics = resolve_run(config)
    run_string = get_run_string(config) + ("_interleaved" if config.schedule == "interleaved" else "")
    interleaved = config.schedule == "interleaved"
    cycles = int(config.num_updates) // config.interleave_updates if interleaved else 1
    total_stages = len(identities) * cycles  # partners, or micro-stages of the interleaved schedule

    checkpoint_meta = None
    if config.resume:
        run_dir, checkpoint_file = resolve_resume(config.resume)
        checkpoint_meta, checkpoint_state = read_checkpoint(checkpoint_file)
        check_resume(checkpoint_meta, fingerprint, num_stages=total_stages)
        run_uid = checkpoint_meta["run_uid"]
    else:
        run_dir, run_uid = allocate_run_dir(config.checkpoint_path, run_string)
    config.save_dir = str(run_dir)
    run_name = run_dir.name
    start_stage = checkpoint_meta["stages_completed"] if checkpoint_meta else 0

    status = RunStatus(run_dir)
    code, device = code_revision(), device_info()
    attempt = status.begin_attempt({"resumed_from_stage": start_stage if checkpoint_meta else None,
                                    "code": code, "device": device})
    recorder = run = None
    try:
        counters = (dict(checkpoint_meta["counters"]) if checkpoint_meta else dict(
            stages_completed=0, train_env_steps=0, train_episodes=0, eval_events=0, eval_full_events=0,
            eval_current_events=0, eval_episodes=0, eval_env_steps=0, importance_env_steps=0))

        if config.layout_name != "":
            layout_dict = {"layout": overcooked_layouts[config.layout_name]}
        else:
            layouts = read_layouts(config)
            layout_dict = {"layout": frozendict_from_layout_repr(
                layouts[config.layout_idx]["layout"])}

        config.layout = layout_dict.copy()  # These are env kwargs
        # CSV schema of this run: a resumed run keeps the one on disk (old files are never rewritten) and can
        # record the diagnostics only if that schema has them.
        csv_schema = config.log_agent_stats
        if config.resume:
            on_disk = csv_has_agent_stats(run_dir / "train_metrics.csv")
            csv_schema = config.log_agent_stats if on_disk is None else on_disk
        record_agent_stats = bool(config.log_agent_stats and csv_schema)
        if config.log_agent_stats and not record_agent_stats:
            log.warning("resuming a run whose CSV files have no per-agent diagnostic columns: they stay unavailable "
                        "for this run (existing logs are not rewritten)")
        status.update_attempt(attempt, agent_stats={"requested": config.log_agent_stats,
                                                    "recorded": record_agent_stats, "csv_columns": csv_schema})

        env = make_env(config.env_name, **config.layout, max_steps=config.num_steps)
        env = LogWrapper(env, agent_stats=record_agent_stats)

        # Calculate max soup for the layout
        layout_name = config.layout_name if config.layout_name != "" else f"layout_{config.layout_idx}"
        max_soup_dict = {layout_name: calculate_max_soup(config.layout["layout"], env.max_steps, n_agents=env.num_agents)}

        # Visualization extras (pygame/imageio) are only needed when recording videos
        visualizer = None
        if config.record_video:
            visualizer = create_visualizer(env.num_agents, config.env_name)

        rng = jax.random.PRNGKey(config.seed)
        rng, init_rng = jax.random.split(rng, 2)

        partners = load_partners(bank, obs_dim=np.prod(env.observation_space().shape))

        if config.alg != "br":
            raise NotImplementedError("Selected method not implemented.")

        # Initialize ego agent
        ac_cls = CNNActorCritic if config.use_cnn else MLPActorCritic

        seq_length = config.num_population_partners + config.num_heuristic_partners

        ego_network = ac_cls(
            len(env.action_set), config.activation, seq_length, config.use_multihead,
            config.shared_backbone, config.hidden_size, config.num_layers, config.use_task_id,
            config.use_layer_norm)

        obs_dim = env.observation_space().shape
        if not config.use_cnn:
            obs_dim = np.prod(obs_dim)

        ego_policy = MLPActorCriticPolicyCL(ego_network, obs_dim)

        # Initialize the network
        rng, network_rng = jax.random.split(rng)

        ego_params = ego_policy.init_params(network_rng)

        # Initialize continual learning state if CL method is specified
        cl_state = None
        if cl is not None:
            cl_state = init_cl_state(ego_params, config.regularize_critic, not config.use_multihead, cl, config)

            # Initialize AGEM memory if using AGEM or ER-ACE
            if config.cl_method.lower() in ("agem", "er_ace"):
                obs_dim_agem = env.observation_space().shape
                if not config.use_cnn:
                    obs_dim_agem = (np.prod(obs_dim_agem),)
                cl_state = init_agem_memory(config.agem_memory_size, obs_dim_agem)

            print(f"Initialized CL state for method: {config.cl_method.upper()}")

        steps_per_update = int(config.num_envs) * int(config.num_steps)
        # Interleaved schedule: every micro-stage reuses one config of `interleave_updates` updates without in-scan
        # evaluation (the cycle-end evaluation runs on the host), and the optimizer state is carried along.
        micro_config = opt_state = None
        if interleaved:
            micro_config = copy.copy(config)
            micro_config.total_timesteps = config.interleave_updates * steps_per_update
            micro_config.num_updates = config.interleave_updates
            micro_config.eval_schedule, micro_config.eval_current_every = "none", 0
            opt_state = make_optimizer(micro_config).init(ego_params)

        if checkpoint_meta:
            ego_params = restore_state(checkpoint_state["ego_params"], ego_params, "ego parameters")
            if interleaved:
                saved = restore_state(checkpoint_state["cl_state"], {"cl": cl_state, "opt": opt_state},
                                      "continual-learning and optimizer state")
                cl_state, opt_state = saved["cl"], saved["opt"]
            else:
                cl_state = restore_state(checkpoint_state["cl_state"], cl_state, "continual-learning state")
        num_stages = len(identities)
        num_eval_partners = num_stages
        if not config.resume:
            config_dict = asdict(config)
            write_json(run_dir / "config.json", dict(config_dict, layout_name_resolved=layout_name))
            atomic_write(run_dir / "config.pckl", pickle.dumps(config_dict))
            write_json(run_dir / "partner_bank.json", identities)
            write_json(run_dir / "run.json", {
                "run_name": run_name, "run_uid": run_uid, "created": status.data["attempts"][0]["started_utc"],
                "code": code, "device": device, "fingerprint": fingerprint,
                "architecture": {
                    "network": f"{ac_cls.__module__}.{ac_cls.__name__}", "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers, "activation": config.activation,
                    "use_layer_norm": config.use_layer_norm, "use_cnn": config.use_cnn,
                    "use_task_id": config.use_task_id, "use_multihead": config.use_multihead,
                    "shared_backbone": config.shared_backbone,
                    # output heads the actor and critic really have: one unless the stage index routes to a head
                    "num_heads": seq_length if config.use_multihead else 1,
                    "observation_dim": int(np.prod(env.observation_space().shape)),
                    "parameter_count": int(sum(np.size(x) for x in jax.tree.leaves(ego_params)))},
                "budget": {
                    "unit": "joint environment steps (one step = both agents act once in each parallel env)",
                    "num_envs": config.num_envs, "num_steps": config.num_steps,
                    "total_timesteps_per_stage": config.total_timesteps,
                    "updates_per_stage": int(config.num_updates), "env_steps_per_update": steps_per_update,
                    "env_steps_per_stage": int(config.num_updates) * steps_per_update, "planned_stages": num_stages,
                    "planned_train_env_steps": num_stages * int(config.num_updates) * steps_per_update,
                    "evaluation_partners": num_eval_partners, "eval_episodes_per_partner": config.num_eval_episodes,
                    "eval_env_steps_per_full_event": num_eval_partners * config.num_eval_episodes * config.num_steps,
                    "evaluation_schedule": config.eval_schedule,
                    "planned_full_evaluations": (num_stages + (1 if config.eval_schedule == "pilot" else 0)
                                                 if config.eval_schedule == "pilot" else
                                                 num_stages * list(eval_plan(config).values()).count("full")),
                    "planned_current_evaluations": num_stages * list(eval_plan(config).values()).count("current"),
                    "importance_env_steps_per_stage": (
                        config.importance_episodes * config.importance_steps
                        if config.cl_method and config.cl_method.lower() in IMPORTANCE_METHODS else 0),
                    **({"schedule": "interleaved", "interleave_updates": config.interleave_updates,
                        "cycles": cycles, "micro_stages": total_stages,
                        "planned_full_evaluations": cycles + (1 if config.eval_schedule == "pilot" else 0),
                        "planned_current_evaluations": 0} if interleaved else {})},
                "schedule": schedule_description(config, labels),
                "normalization": {
                    "max_soup": float(max_soup_dict[layout_name]),
                    "definition": "W&B *_scaled metrics = soups / max_soup; the CSVs hold raw soups"},
                "partners": identities, "bank": {"name": bank.name, "source": str(bank.source)},
                "diagnostics": {
                    "per_agent_stats": record_agent_stats, "ego": "agent_0", "partner": "agent_1",
                    "action_order": list(ACTION_NAMES),
                    "definition": "see the CPA README: per-agent soups, raw unannealed shaped return and action counts"},
            })
        elif interleaved:  # rows of an uncommitted micro-stage lie beyond the checkpoint's environment steps
            last = start_stage * config.interleave_updates * steps_per_update
            dropped = sum(truncate_csv(run_dir / f, last, column=c) for f, c in (
                ("train_metrics.csv", "env_steps"), ("eval_metrics.csv", "env_steps"),
                ("stage_timings.csv", "env_steps_end")))
            status.update_attempt(attempt, discarded_rows_of_uncommitted_stages=dropped)
            for stale in run_dir.glob(".*.tmp"):
                stale.unlink()
        else:
            dropped = sum(truncate_csv(run_dir / f, start_stage - 1)
                          for f in ("train_metrics.csv", "eval_metrics.csv", "stage_timings.csv"))
            status.update_attempt(attempt, discarded_rows_of_uncommitted_stages=dropped)
            for stale in run_dir.glob(".*.tmp"):
                stale.unlink()

        # Initialize WandB (credentials come from the environment / stored login, never from this code)
        wandb_name = run_name if attempt == 0 else f"{run_name}_resume{attempt}"
        if config.mode == "online":
            wandb.login(key=os.environ.get("WANDB_API_KEY"))
        run = wandb.init(
            project=config.project,
            config=asdict(config),
            sync_tensorboard=True,
            mode=config.mode,
            tags=config.tags if config.tags is not None else [],
            group=config.group,
            name=wandb_name,
            id=wandb_name,
            save_code=True,
        )
        status.update_attempt(attempt, wandb={"mode": config.mode, "run_name": wandb_name})
        print("XPID ID name:")
        print(run.name)
        print("-------------")

        recorder = RunRecorder(run_dir, labels, config.seed, attempt=attempt,
                               wandb_log=lambda d, step: run.log(d, step=step), agent_stats=csv_schema)

        # Compiled functions shared by all stages of this run (see train_ppo_ego_agent)
        compiled_cache = {}

        indp = OvercookedIndependentPolicyWrapper(
            layout=config.layout["layout"], p_onion_on_counter=0.5, p_plate_on_counter=0.5)
        onin = OvercookedOnionPolicyWrapper(layout=config.layout["layout"])
        plate = OvercookedPlatePolicyWrapper(layout=config.layout["layout"])
        rndm = OvercookedRandomPolicyWrapper(layout=config.layout["layout"])
        static = OvercookedStaticPolicyWrapper(layout=config.layout["layout"])
        heuristic_policies = [indp, onin, plate, rndm, static]

        fake_params = jax.tree.map(
            lambda x: x[jnp.newaxis, ...], ego_params)

        stages = [Stage(identity, partner.policy, partner.params, partner.record.config)
                  for identity, partner in zip(identities, partners)]
        stages += [Stage(identities[len(partners) + i], heuristic_policies[i], None, bank.records[0].config)
                   for i in range(num_heuristics)]

        # Build evaluation partner list based on configuration
        eval_partner = []
        partner_idx = 0

        # Add population partners
        for partner in partners:
            population_cls = HeuristicPolicyPopulation if partner.record.kind == "planner" else DummyPolicyPopulation
            eval_partner.append((
                population_cls(policy_cls=partner.policy),
                jax.tree.map(lambda x: x[jnp.newaxis, ...], partner.params),
                partner_idx
            ))
            partner_idx += 1

        # Add heuristic partners
        for i in range(num_heuristics):
            eval_partner.append((
                HeuristicPolicyPopulation(policy_cls=heuristic_policies[i]),
                fake_params,
                partner_idx
            ))
            partner_idx += 1

        # Train the ego against the partners in order: bank partners first, then legacy heuristics. The interleaved
        # schedule cycles through them in micro-stages m instead, (cycle, partner) = divmod(m, num_stages).
        profiling = bool(config.only_stages)
        plan = eval_plan(config)
        micro = config.interleave_updates
        for m in (config.only_stages if profiling else range(start_stage, total_stages)):
            cycle, k = divmod(m, num_stages) if interleaved else (0, m)
            stage, stats, carry = stages[k], {}, {}
            stage_start = time.perf_counter()
            init_eval = config.eval_schedule == "pilot" and m == 0 and not checkpoint_meta
            if interleaved:
                final_eval = k == num_stages - 1  # every partner is evaluated after every cycle
                recorder.begin_stage(k, micro, {micro: "full"} if final_eval else {}, init_eval=init_eval,
                                     update_base=cycle * micro)
                schedule_args = dict(key_index=m, init_opt_state=opt_state, update_base=cycle * micro,
                                     step_base=m * micro, shaping_base=m * micro, final_eval=final_eval, carry=carry)
            else:
                recorder.begin_stage(k, int(config.num_updates), plan, init_eval=init_eval)
                schedule_args = {}
            ego_params, cl_state = run_br_training(
                micro_config if interleaved else config, env, stage.agent_config, ego_policy,
                ego_params, stage.policy, stage.params, env_id_idx=k, eval_partner=eval_partner,
                max_soup_dict=max_soup_dict, cl=cl, cl_state=cl_state, compiled_cache=compiled_cache,
                record_fn=recorder.handle, stats=stats, init_eval=init_eval, **schedule_args)
            if interleaved:
                opt_state = carry["final_opt_state"]
            summary = recorder.end_stage()
            counters["stages_completed"] = m + 1
            counters["train_env_steps"] += summary["updates"] * steps_per_update
            counters["train_episodes"] += summary["train_episodes"]
            for key in ("eval_events", "eval_full_events", "eval_current_events", "eval_episodes"):
                counters[key] = counters.get(key, 0) + summary[key]
            counters["eval_env_steps"] += summary["eval_episodes"] * config.num_steps
            if "importance_s" in stats:
                counters["importance_env_steps"] += config.importance_episodes * config.importance_steps
            peak = stats.get("peak_bytes_in_use")
            recorder.write_timing(dict(
                stage=k, partner_id=k, partner=labels[k], attempt=attempt, updates=summary["updates"],
                env_steps_end=summary["env_steps_end"], compile_s=stats.get("compile_s"),
                compile_cached=stats.get("compile_cached"), train_eval_s=stats.get("train_eval_s"),
                eval_events=summary["eval_full_events"], eval_event_s=stats.get("eval_event_s"),
                eval_current_events=summary["eval_current_events"],
                eval_current_event_s=stats.get("eval_current_event_s"), init_eval_s=stats.get("init_eval_s"),
                importance_s=stats.get("importance_s"),
                importance_compiled_here=stats.get("importance_compiled_here"),
                stage_wall_s=time.perf_counter() - stage_start, backend=device["backend"],
                device_kind=device["devices"][0]["device_kind"], peak_bytes_in_use=peak,
                memory_status="device peak since process start" if peak is not None else "unavailable"))
            # A partner counts as done only once its records are on disk and the checkpoint has replaced the old one.
            schedule_meta = dict(cycle=cycle, partner=k, micro_stages=total_stages) if interleaved else {}
            if config.save_checkpoints and not profiling:
                save_checkpoint(run_dir, ego_params, {"cl": cl_state, "opt": opt_state} if interleaved else cl_state, {
                    "run_uid": run_uid, "stages_completed": m + 1, "counters": counters,
                    "fingerprint": fingerprint, "attempt": attempt, **schedule_meta})
            status.update(state="running", stages_completed=m + 1, counters=counters,
                          wandb_failures=recorder.wandb_failures, wandb_error=recorder.wandb_error, **schedule_meta)

            # Record the ego with the frozen partner it was just trained with (interleaved: in the last cycle)
            if config.record_video and cycle == cycles - 1:
                try:
                    record_stage_video(config, env, ego_policy, ego_params, stage, k, labels[k], run_dir, visualizer,
                                       compiled_cache, run, summary["env_steps_end"])
                except Exception as e:  # noqa: BLE001 - videos are diagnostics; the stage itself is complete
                    log.warning("recording the stage-%d video failed: %s: %s", k, type(e).__name__, e)

        if profiling:
            status.update(state="profile", counters=counters)
            return RunResult(run_dir, ego_params, cl_state, counters)
        atomic_write(run_dir / f"params_seed{config.seed}.pt", pickle.dumps({"actor_params": ego_params}))
        status.update(state="complete", stages_completed=total_stages, counters=counters,
                      wandb_failures=recorder.wandb_failures, wandb_error=recorder.wandb_error)
        return RunResult(run_dir, ego_params, cl_state, counters)
    except BaseException as e:
        status.update(state="interrupted" if isinstance(e, KeyboardInterrupt) else "failed",
                      error=f"{type(e).__name__}: {e}")
        raise
    finally:
        if recorder is not None:
            recorder.close()
        if run is not None:
            try:
                run.finish()
            except Exception as e:  # noqa: BLE001 - local records are already complete
                log.warning("closing the W&B run failed: %s", e)


def video_name(k: int, label: str) -> str:
    return f"stage_{k:02d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', label).strip('_')}"


def record_stage_video(config, env, ego_policy, ego_params, stage, k, label, run_dir, visualizer, cache, run,
                       wandb_step):
    """Write <run dir>/videos/stage_<k>_<label>.mp4 (ego = agent_0, frozen partner = agent_1) and log it to W&B."""
    population, partner_params = make_partner_population(stage.policy, stage.params, ego_params)
    rng = jax.random.fold_in(jax.random.PRNGKey(config.seed), 10_000 + k)
    states = rollout_ego_with_partner(rng, env, ego_policy.network, ego_params, k, population, partner_params,
                                      config.gif_len, use_cnn=config.use_cnn, cache=cache)
    name = video_name(k, label)
    path = visualizer.animate(states, out_path=str(Path(run_dir) / "videos" / f"{name}.mp4"), task_idx=k, env=env,
                              log_to_wandb=False)
    if config.mode != "disabled" and run is not None:
        try:
            run.log({f"Video/{name}": wandb.Video(path, format="mp4")}, step=wandb_step)
        except Exception as e:  # noqa: BLE001 - the mp4 is on disk; a W&B hiccup must not stop training
            log.warning("uploading %s to W&B failed: %s", path, e)
    return path


def run_training():
    execute(tyro.cli(TrainConfig))


if __name__ == '__main__':
    run_training()
