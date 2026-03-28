"""
Coordination compatibility matrix for MEAL (Overcooked).

Entry M[i, j] = normalised soup score when:
  - Agent 0 uses the checkpoint saved after task i
  - Agent 1 uses the checkpoint saved after task j
  - Both are evaluated on task i's layout

The diagonal (i == j) is the standard within-task evaluation.
Off-diagonal entries reveal whether coordination conventions survive
continual learning.  A drop in the upper triangle (j > i) means later
checkpoints can no longer coordinate with the partner they trained with
on task i — coordination forgetting, invisible to the standard metric F.

Usage:
    python scripts/coordination_matrix.py \\
        --run_dir checkpoints/overcooked/EWC/ippo_ewc_easy_2agents_... \\
        --num_envs 64 --num_steps 400 --seed 0

    # compare multiple runs side-by-side
    python scripts/coordination_matrix.py \\
        --run_dir checkpoints/overcooked/EWC/run_A \\
                  checkpoints/overcooked/FT/run_B \\
        --labels EWC FT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import flax
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meal import Overcooked
from meal.wrappers.logging import LogWrapper
from meal.env.utils.max_soup_calculator import calculate_max_soup
from experiments.model.mlp import ActorCritic as MLPActorCritic
from experiments.model.cnn import ActorCritic as CNNActorCritic
from experiments.utils import batchify, unbatchify


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_run(run_dir: Path):
    """
    Discover all task checkpoints in a run directory.

    Returns
    -------
    checkpoints : list of Path  (model_env_1, model_env_2, ...)
    configs     : list of dict  (corresponding _config.json contents)
    """
    checkpoints, configs = [], []
    idx = 1
    while True:
        ckpt = run_dir / f"model_env_{idx}"
        cfg_path = run_dir / f"model_env_{idx}_config.json"
        if not ckpt.exists():
            break
        with open(cfg_path) as f:
            configs.append(json.load(f))
        checkpoints.append(ckpt)
        idx += 1
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return checkpoints, configs


def restore_params(path: Path, template):
    """Load a flax checkpoint into *template*'s pytree structure."""
    with open(path, "rb") as f:
        return flax.serialization.from_bytes({"params": template}, f.read())["params"]


# ---------------------------------------------------------------------------
# Environment reconstruction
# ---------------------------------------------------------------------------

_IDX_KEYS = ("wall_idx", "onion_pile_idx", "plate_pile_idx", "pot_idx", "goal_idx",
             "onion_idx", "plate_idx", "agent_idx")


def _fix_layout(layout: dict) -> dict:
    """Ensure all index fields are JAX arrays (JSON serialisation can collapse
    single-element arrays to bare ints, and plain lists break JAX indexing)."""
    out = dict(layout)
    for k in _IDX_KEYS:
        if k in out:
            v = out[k]
            if not isinstance(v, list):
                v = [v]
            out[k] = jnp.array(v, dtype=jnp.int32)
    return out


def rebuild_env(cfg: dict, num_agents: int = 2) -> LogWrapper:
    """Reconstruct a LogWrapper(Overcooked) from a saved _config.json."""
    env = Overcooked(
        layout=_fix_layout(cfg["env_kwargs"]),
        layout_name=cfg.get("layout_name", "custom"),
        num_agents=num_agents,
    )
    return LogWrapper(env, replace_info=False)


# ---------------------------------------------------------------------------
# Cross-checkpoint evaluation
# ---------------------------------------------------------------------------

def make_cross_eval(env, network, agents: list, num_envs: int, num_steps: int, task_idx: int):
    """
    Build a jitted function:
        cross_eval(rng, params_a, params_b) -> mean_soups_per_env

    Agent 0 uses params_a, agent 1 uses params_b.
    Both conditioned on task_idx (static — one compiled fn per task).
    """
    @jax.jit
    def cross_eval(rng: jnp.ndarray, params_a, params_b) -> float:
        rng, env_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(env_rng, num_envs)
        obs, env_state = jax.vmap(env.reset)(reset_rngs)

        total_soups = jnp.zeros((num_envs,), jnp.float32)

        def one_step(carry, _):
            env_state, obs, soups, rng = carry

            # Split the joint observation batch and run each agent's params
            obs_batch = batchify(obs, agents, len(agents) * num_envs, True)
            obs_a = obs_batch[:num_envs]   # (num_envs, obs_dim)
            obs_b = obs_batch[num_envs:]   # (num_envs, obs_dim)

            pi_a, _, _ = network.apply(params_a, obs_a, env_idx=task_idx)
            pi_b, _, _ = network.apply(params_b, obs_b, env_idx=task_idx)

            action = jnp.concatenate([pi_a.mode(), pi_b.mode()], axis=0)
            env_act = unbatchify(action, agents, num_envs, len(agents))
            env_act = {k: v.flatten() for k, v in env_act.items()}

            rng, sub = jax.random.split(rng)
            step_rngs = jax.random.split(sub, num_envs)
            obs2, env_state2, reward, done, info = jax.vmap(env.step)(
                step_rngs, env_state, env_act
            )

            soups += sum(info["soups"][a] for a in agents)
            return (env_state2, obs2, soups, rng), None

        (_, _, total_soups, _), _ = jax.lax.scan(
            one_step,
            (env_state, obs, total_soups, rng),
            xs=None,
            length=num_steps,
        )
        return total_soups.mean()

    return cross_eval


# ---------------------------------------------------------------------------
# Matrix computation
# ---------------------------------------------------------------------------

def compute_matrix(
    run_dir: Path,
    num_envs: int,
    num_steps: int,
    seed: int,
    num_agents: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns
    -------
    matrix      : (N, N) raw mean-soup scores
    matrix_norm : (N, N) normalised by per-task max_soup
    layout_names: list of task layout names (length N)
    """
    checkpoints, configs = load_run(run_dir)
    N = len(checkpoints)
    cfg0 = configs[0]

    use_cnn   = cfg0.get("use_cnn", False)
    num_tasks = cfg0.get("num_tasks", N)

    # Rebuild environments and get layout names / max-soup values
    envs = []
    layout_names = []
    max_soups = []
    for cfg in configs:
        env = rebuild_env(cfg, num_agents=num_agents)
        envs.append(env)
        layout_names.append(cfg.get("layout_name", "?"))
        ms = calculate_max_soup(env.layout, env.max_steps, n_agents=num_agents)
        max_soups.append(ms)

    # Build network and initialise with dummy data to get param template
    temp_env = envs[0]
    agents   = temp_env.agents
    action_n = temp_env.action_space().n

    ac_cls = CNNActorCritic if use_cnn else MLPActorCritic
    network = ac_cls(
        action_n,
        cfg0.get("activation", "relu"),
        num_tasks,
        cfg0.get("use_multihead", True),
        cfg0.get("shared_backbone", False),
        cfg0.get("big_network", False),
        cfg0.get("use_task_id", True),
        cfg0.get("regularize_heads", False),
        cfg0.get("use_layer_norm", True),
    )

    rng = jax.random.PRNGKey(seed)
    rng, init_rng, reset_rng = jax.random.split(rng, 3)
    reset_rngs = jax.random.split(reset_rng, 1)
    init_obs, _ = jax.vmap(temp_env.reset)(reset_rngs)
    init_obs_batch = batchify(init_obs, agents, len(agents), True)
    # train_state.params = network.init() (full dict), so template must match that structure
    template_params = network.init(init_rng, init_obs_batch, env_idx=0)

    # Load all checkpoint params
    all_params = [restore_params(ckpt, template_params) for ckpt in checkpoints]

    # Build per-task cross-eval functions (compiled once per task)
    eval_fns = [
        make_cross_eval(envs[i], network, agents, num_envs, num_steps, i)
        for i in range(N)
    ]

    # Run N×N evaluation
    matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            rng, eval_rng = jax.random.split(rng)
            score = float(eval_fns[i](eval_rng, all_params[i], all_params[j]))
            matrix[i, j] = score
            print(f"  M[{i},{j}] task={layout_names[i]:20s}  "
                  f"ckpt_a=task{i+1}  ckpt_b=task{j+1}  soup={score:.3f}")

    matrix_norm = matrix / np.array(max_soups)[:, None]
    return matrix_norm, matrix, layout_names


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def coordination_retention(matrix: np.ndarray) -> dict:
    """
    Summary metrics derived from the N×N normalised matrix.

    diagonal_mean   : average on-diagonal (standard eval baseline)
    upper_tri_mean  : mean of strict upper triangle (j > i)
                      how well later checkpoints coordinate with earlier ones
    retention_ratio : upper_tri_mean / diagonal_mean  (1.0 = perfect retention)
    coord_forgetting: 1 - retention_ratio
    """
    N = matrix.shape[0]
    diag = np.diag(matrix)
    upper_idx = np.triu_indices(N, k=1)
    upper = matrix[upper_idx]

    diag_mean  = float(np.nanmean(diag))
    upper_mean = float(np.nanmean(upper))
    ratio      = upper_mean / diag_mean if diag_mean > 1e-8 else np.nan
    return {
        "diagonal_mean":    diag_mean,
        "upper_tri_mean":   upper_mean,
        "retention_ratio":  ratio,
        "coord_forgetting": 1.0 - ratio if not np.isnan(ratio) else np.nan,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_matrix(matrix: np.ndarray, layout_names: list[str], title: str, out_path: Path):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    N = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(5, N * 0.7), max(4, N * 0.65)))

    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Normalised soup score")

    # Annotate cells
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

    # Highlight diagonal
    for k in range(N):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   fill=False, edgecolor="black", lw=1.5))

    short = [n.replace("easy_gen_", "T").replace("medium_gen_", "T")
               .replace("hard_gen_", "T") for n in layout_names]
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels([f"ckpt {i+1}\n{short[i]}" for i in range(N)], fontsize=7)
    ax.set_yticklabels([f"task {i+1}\n{short[i]}" for i in range(N)], fontsize=7)
    ax.set_xlabel("Agent-1 checkpoint (params_j)")
    ax.set_ylabel("Evaluation task / Agent-0 checkpoint (params_i)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def print_latex(matrix: np.ndarray, layout_names: list[str], label: str):
    N = matrix.shape[0]
    short = [n.replace("easy_gen_", "T").replace("medium_gen_", "T")
               .replace("hard_gen_", "T") for n in layout_names]
    col_fmt = "l" + "c" * N
    header  = " & ".join(rf"$\mathbf{{ckpt_{{{i+1}}}}}$" for i in range(N))

    lines = [
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{Coordination compatibility matrix for {label}. "
        r"M[i,j]: normalised soup when agent-0 uses checkpoint $i$ and agent-1 uses checkpoint $j$, "
        r"evaluated on task $i$. Diagonal = standard eval; "
        r"upper triangle shows coordination retention under continual learning.}}",
        rf"\label{{tab:coord_matrix_{label.replace(' ', '_')}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        r"\textbf{Task} & " + header + r" \\",
        r"\midrule",
    ]
    for i in range(N):
        row_vals = []
        for j in range(N):
            v = f"{matrix[i, j]:.2f}"
            if i == j:
                v = rf"\textbf{{{v}}}"
            row_vals.append(v)
        lines.append(f"{short[i]} & " + " & ".join(row_vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Coordination compatibility matrix")
    p.add_argument("--run_dir", nargs="+", required=True,
                   help="One or more checkpoint run directories")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Display labels for each run (defaults to directory names)")
    p.add_argument("--num_envs",  type=int, default=64)
    p.add_argument("--num_steps", type=int, default=400)
    p.add_argument("--num_agents", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=Path("runs/coordination"),
                   help="Directory to save heatmap PNGs")
    p.add_argument("--latex", action="store_true",
                   help="Print LaTeX tables")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(d) for d in args.run_dir]
    labels   = args.labels or [d.name for d in run_dirs]

    for run_dir, label in zip(run_dirs, labels):
        print(f"\n{'='*60}")
        print(f"Run: {label}  ({run_dir})")
        print(f"{'='*60}")

        matrix_norm, matrix_raw, layout_names = compute_matrix(
            run_dir    = run_dir,
            num_envs   = args.num_envs,
            num_steps  = args.num_steps,
            seed       = args.seed,
            num_agents = args.num_agents,
        )

        metrics = coordination_retention(matrix_norm)
        print(f"\nCoordination metrics:")
        print(f"  diagonal mean (standard eval):  {metrics['diagonal_mean']:.3f}")
        print(f"  upper-triangle mean:             {metrics['upper_tri_mean']:.3f}")
        print(f"  retention ratio:                 {metrics['retention_ratio']:.3f}")
        print(f"  coordination forgetting (CF):    {metrics['coord_forgetting']:.3f}")

        out_path = args.out_dir / f"coord_matrix_{label.replace(' ', '_')}.png"
        plot_matrix(matrix_norm, layout_names, title=f"Coordination matrix — {label}", out_path=out_path)

        if args.latex:
            print("\nLaTeX table:")
            print_latex(matrix_norm, layout_names, label)
