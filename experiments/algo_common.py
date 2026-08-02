import os
from dataclasses import asdict

import flax
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from tensorboardX import SummaryWriter


def convert_frozen_dict(obj):
    """Recursively converts FrozenDict/JAX arrays to plain Python types for JSON dumping.

    Identical (modulo an unnecessary explicit `unfreeze()` call in the on-policy scripts,
    which is a no-op since FrozenDict already supports `.items()`) across ippo.py, mappo.py,
    happo.py, vdn.py, qmix.py.
    """
    if isinstance(obj, flax.core.frozen_dict.FrozenDict):
        return {k: convert_frozen_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: convert_frozen_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_frozen_dict(item) for item in obj]
    elif isinstance(obj, jax.Array):
        array_obj = np.array(obj)
        return array_obj.item() if array_obj.size == 1 else array_obj.tolist()
    else:
        return obj


def resolve_reg_coef(cfg):
    """Sets cfg.reg_coef to the method's default if the user didn't specify one.

    Identical across all scripts for the methods that actually read reg_coef
    (ewc/mas/l2 — the only `RegCLMethod` instances). ft/agem/er_ace/packnet never read
    reg_coef, so leaving it None for them (as ippo/mappo/happo do) vs. VDN/QMIX's
    `.get(..., 0.0)` default are behaviorally equivalent.
    """
    if cfg.reg_coef is None:
        if cfg.cl_method.lower() == "ewc":
            cfg.reg_coef = 1e11
        elif cfg.cl_method.lower() == "mas":
            cfg.reg_coef = 1e9
        elif cfg.cl_method.lower() == "l2":
            cfg.reg_coef = 1e7


def init_wandb_and_tensorboard(cfg, run_name, exp_dir) -> SummaryWriter:
    """wandb.init + TensorBoard hyperparameter table. Identical across all 5 scripts."""
    if cfg.use_wandb:
        wandb_tags = cfg.tags if cfg.tags is not None else []
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(
            project=cfg.project,
            config=asdict(cfg),
            sync_tensorboard=True,
            mode=cfg.wandb_mode,
            tags=wandb_tags,
            group=cfg.cl_method.upper(),
            name=run_name,
            id=run_name,
        )

    writer = SummaryWriter(exp_dir)
    rows = []
    for key, value in vars(cfg).items():
        value_str = str(value).replace("\n", "<br>").replace("|", "\\|")
        rows.append(f"|{key}|{value_str}|")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(rows))
    return writer


def build_reset_step_switch(envs):
    """Returns (reset_switch, step_switch) that dispatch to the right task's env via
    jax.lax.switch. Identical across all scripts."""
    reset_fns = tuple(env.reset for env in envs)
    step_fns = tuple(env.step for env in envs)

    def reset_switch(key, task_idx):
        return jax.lax.switch(task_idx, reset_fns, key)

    def step_switch(key, state, actions, task_idx):
        return jax.lax.switch(task_idx, step_fns, key, state, actions)

    return reset_switch, step_switch


def compute_soup_metrics(metrics, cfg, num_agents, agents, info, traj_batch, max_soup_vals, env_idx):
    """Per-episode-average soup metrics ("Soup/total", "Soup/scaled", "Soup/<agent>").

    Identical (modulo local variable names) across ippo.py, mappo.py, happo.py. Mutates
    and returns `metrics`, reading raw per-step values from `info["soups"]` and popping
    the now-superseded mean-reduced `metrics["soups"]` entry at the end (`metrics` is
    expected to already be `jax.tree_util.tree_map(lambda x: x.mean(), info)`, per the
    original call site).
    """
    T, E, A = cfg.num_steps, cfg.num_envs, num_agents
    max_per_episode = max_soup_vals[env_idx]

    soups_tea = jnp.stack([info["soups"][a] for a in agents], axis=-1)
    soups_per_env = soups_tea.sum(axis=(0, 2))

    done_tea = traj_batch.done.reshape(T, E, A)
    done_te = done_tea[..., 0]
    episodes_per_env = done_te.sum(axis=0)

    mask = episodes_per_env > 0
    true_avg = jnp.where(mask, soups_per_env / jnp.maximum(episodes_per_env, 1), 0.0)
    num_finished = jnp.maximum(mask.sum(), 1)

    metrics["Soup/total"] = true_avg.sum() / num_finished
    metrics["Soup/scaled"] = jnp.where(
        max_per_episode > 0, (true_avg / max_per_episode).sum() / num_finished, 0.0
    )
    for ai, agent in enumerate(agents):
        soups_te = soups_tea[:, :, ai].sum(axis=0)
        per_agent = jnp.where(mask, soups_te / jnp.maximum(episodes_per_env, 1), 0.0)
        metrics[f"Soup/{agent}"] = per_agent.sum() / num_finished

    metrics.pop("soups", None)
    return metrics


def compute_reward_metrics(metrics, agents, current_timestep, rew_shaping_anneal):
    """Per-agent shaped-reward metrics. Identical across ippo.py, mappo.py, happo.py.
    Mutates and returns `metrics`."""
    for agent in agents:
        metrics[f"General/shaped_reward_{agent}"] = metrics["shaped_reward"][agent]
        metrics[f"General/shaped_reward_annealed_{agent}"] = (
                metrics[f"General/shaped_reward_{agent}"] * rew_shaping_anneal(current_timestep)
        )
    metrics.pop("shaped_reward", None)
    return metrics


def apply_packnet_mask_or_plain(is_packnet, cl, cl_state, grads, train_state):
    """The non-AGEM/ER-ACE gradient-application branch: mask grads if packnet, then
    apply_gradients. Identical across ippo.py, mappo.py, happo.py (happo calls this once
    per agent inside its sequential HAPPO update).

    `is_packnet` is passed in (rather than computed here from `cfg.cl_method`) because
    ippo.py/mappo.py check `cfg.cl_method == "packnet"` while happo.py checks
    `cfg.cl_method.lower() == "packnet"` — a pre-existing casing inconsistency preserved
    here rather than silently unified.
    """
    if is_packnet:
        grads = cl.mask_gradients(cl_state, grads)
    return train_state.apply_gradients(grads=grads)


def run_packnet_train_then_finetune(cfg, cl, update_step_fn, runner_state, num_finetune_updates):
    """Runs the main training scan, then (if cl_method == packnet) prunes via
    `cl.on_train_end`, runs a finetune scan, and wraps up via `cl.on_finetune_end`.

    Generic over runner_state shape: element [0] is the TrainState that packnet prunes
    (train_state for ippo/mappo, actor_ts for happo — critic_ts, if present, passes
    through untouched), element [-1] is cl_state, element [-2] is rng. This matches the
    structure of ippo.py, mappo.py, and happo.py exactly, including
    the original's quirk of reinserting `finetune_rng` (the pre-finetune-scan seed) as the
    final rng rather than whatever rng the finetune scan produced.
    """
    runner_state, metrics = jax.lax.scan(
        update_step_fn, runner_state, xs=None, length=cfg.num_updates
    )

    if cfg.cl_method.lower() != "packnet":
        return runner_state, metrics

    state = list(runner_state)
    state[0], state[-1] = cl.on_train_end(state[0], state[-1])

    rng = state[-2]
    rng, finetune_rng = jax.random.split(rng)
    state[-2] = finetune_rng
    runner_state = tuple(state)

    runner_state, metrics = jax.lax.scan(
        update_step_fn, runner_state, xs=None, length=num_finetune_updates
    )

    state = list(runner_state)
    state[0], state[-1] = cl.on_finetune_end(state[0], state[-1])
    state[-2] = finetune_rng
    runner_state = tuple(state)
    return runner_state, metrics
