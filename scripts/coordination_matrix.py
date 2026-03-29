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
# Batch discovery helpers
# ---------------------------------------------------------------------------

def parse_group(spec: str) -> tuple[str, str, str]:
    """Parse a group spec 'label:method_folder[:timestamp_prefix]'."""
    parts = spec.split(":", 2)
    label     = parts[0]
    folder    = parts[1]
    ts_prefix = parts[2] if len(parts) > 2 else ""
    return label, folder, ts_prefix


def discover_runs(
    checkpoint_root: Path,
    method_folder: str,
    difficulty: str,
    ts_prefix: str = "",
) -> List[Path]:
    """Return all run directories matching folder / difficulty / timestamp prefix."""
    base = checkpoint_root / method_folder
    if not base.exists():
        return []
    runs = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if difficulty.lower() not in name.lower():
            continue
        if ts_prefix and ts_prefix not in name:
            continue
        if (d / "model_env_1").exists():
            runs.append(d)
    return runs


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------

def aggregate_matrices(
    run_dirs: List[Path],
    num_envs: int,
    num_steps: int,
    seed: int,
    num_agents: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Compute coordination matrix for each seed run, then return
    (mean_norm, std_norm, mean_raw, layout_names).
    """
    matrices, layout_names_ref = [], None
    for rd in run_dirs:
        try:
            m_norm, _, layout_names = compute_matrix(
                run_dir=rd, num_envs=num_envs, num_steps=num_steps,
                seed=seed, num_agents=num_agents,
            )
            matrices.append(m_norm)
            if layout_names_ref is None:
                layout_names_ref = layout_names
        except Exception as e:
            print(f"[warn] {rd.name}: {e}")

    if not matrices:
        raise RuntimeError("No matrices computed — all runs failed.")

    stack = np.stack(matrices, axis=0)           # (n_seeds, N, N)
    return stack.mean(axis=0), stack.std(axis=0), layout_names_ref or []


# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------

def _fmt_cell(mean: float, ci: float, bold: bool) -> str:
    if np.isnan(mean):
        return "--"
    s = f"{mean:.3f}"
    if bold:
        s = rf"\textbf{{{s}}}"
    if ci > 0:
        s += rf"{{\scriptsize$\pm{ci:.3f}$}}"
    return s


def print_summary_table(results: dict, difficulties: List[str], latex: bool) -> None:
    """
    results: {label: {difficulty: {"cf": (mean, ci), "diag": (mean, ci), "upper": (mean, ci)}}}
    """
    labels = list(results.keys())

    print("\n" + "=" * 70)
    print("SUMMARY — Coordination Forgetting (CF) per method × difficulty")
    print("=" * 70)
    header = f"{'Method':<18}" + "".join(f"  {d.capitalize():<12}" for d in difficulties)
    print(header)
    for label in labels:
        row = f"{label:<18}"
        for diff in difficulties:
            cell = results[label].get(diff)
            if cell is None:
                row += "  --          "
            else:
                m, ci = cell["cf"]
                row += f"  {m:.3f}±{ci:.3f}   "
        print(row)

    if not latex:
        return

    n_diff = len(difficulties)
    col_fmt = "l" + "c" * n_diff
    diff_header = " & ".join(d.capitalize() for d in difficulties)

    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Coordination forgetting (CF $= 1 - $ upper-triangle mean / diagonal mean) "
        r"of the coordination compatibility matrix, averaged over seeds. "
        r"Higher CF means later checkpoints coordinate less well with earlier partners.}",
        r"\label{tab:coord_forgetting_batch}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        r"Method & " + diff_header + r" \\",
        r"\midrule",
    ]
    # best (lowest CF) per difficulty
    best = {}
    for diff in difficulties:
        vals = [results[l][diff]["cf"][0] for l in labels if diff in results[l] and not np.isnan(results[l][diff]["cf"][0])]
        best[diff] = min(vals) if vals else np.nan

    for label in labels:
        cells = [label]
        for diff in difficulties:
            cell = results[label].get(diff)
            if cell is None:
                cells.append("--")
            else:
                m, ci = cell["cf"]
                is_best = not np.isnan(m) and np.isclose(m, best.get(diff, np.nan))
                cells.append(_fmt_cell(m, ci, is_best))
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    print("\nLaTeX:\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Coordination compatibility matrix — single run or batch mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run
  python scripts/coordination_matrix.py --run_dir checkpoints/overcooked/EWC/run_A

  # Batch: aggregate over seeds, all methods × difficulties
  python scripts/coordination_matrix.py \\
      --checkpoint_root checkpoints/overcooked \\
      --groups "EWC:EWC:2026-03-28_20-29" "Online_EWC:EWC:2026-03-28_20-31" "FT:FT" "L2:L2" "MAS:MAS" \\
      --difficulties easy medium hard \\
      --latex
""")

    # Single-run mode
    p.add_argument("--run_dir", nargs="+", default=None,
                   help="One or more checkpoint run directories (single-run mode)")
    p.add_argument("--labels", nargs="+", default=None)

    # Batch mode
    p.add_argument("--checkpoint_root", type=Path, default=None,
                   help="Root of the checkpoints tree (enables batch mode)")
    p.add_argument("--groups", nargs="+", default=None,
                   help="Group specs 'label:method_folder[:timestamp_prefix]'")
    p.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"],
                   help="Difficulty keywords to match in run directory names")

    # Shared
    p.add_argument("--num_envs",  type=int, default=64)
    p.add_argument("--num_steps", type=int, default=400)
    p.add_argument("--num_agents", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=Path("runs/coordination"))
    p.add_argument("--latex", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Batch mode
    # ----------------------------------------------------------------
    if args.checkpoint_root is not None:
        if not args.groups:
            p.error("--groups is required with --checkpoint_root")

        groups = [parse_group(g) for g in args.groups]
        summary_results: dict = {}

        for label, method_folder, ts_prefix in groups:
            summary_results[label] = {}
            for difficulty in args.difficulties:
                run_dirs = discover_runs(
                    args.checkpoint_root, method_folder, difficulty, ts_prefix
                )
                if not run_dirs:
                    print(f"[warn] No runs found for {label} / {difficulty}")
                    continue

                print(f"\n{'='*60}")
                print(f"{label}  |  {difficulty}  ({len(run_dirs)} seeds)")
                for rd in run_dirs:
                    print(f"  {rd.name}")
                print(f"{'='*60}")

                try:
                    mean_norm, std_norm, layout_names = aggregate_matrices(
                        run_dirs, args.num_envs, args.num_steps,
                        args.seed, args.num_agents,
                    )
                except Exception as e:
                    print(f"[error] {label}/{difficulty}: {e}")
                    continue

                metrics = coordination_retention(mean_norm)
                n_seeds = len(run_dirs)
                # Compute per-seed CF for CI
                per_seed_cf = []
                for rd in run_dirs:
                    try:
                        m, _, _ = compute_matrix(rd, args.num_envs, args.num_steps, args.seed, args.num_agents)
                        per_seed_cf.append(coordination_retention(m)["coord_forgetting"])
                    except Exception:
                        pass
                cf_mean = float(np.mean(per_seed_cf)) if per_seed_cf else np.nan
                cf_ci   = (1.96 * np.std(per_seed_cf, ddof=1) / np.sqrt(len(per_seed_cf))
                           if len(per_seed_cf) > 1 else 0.0)

                summary_results[label][difficulty] = {
                    "cf":    (cf_mean, cf_ci),
                    "diag":  (metrics["diagonal_mean"], 0.0),
                    "upper": (metrics["upper_tri_mean"], 0.0),
                }

                print(f"  diagonal mean:      {metrics['diagonal_mean']:.3f}")
                print(f"  upper-triangle:     {metrics['upper_tri_mean']:.3f}")
                print(f"  coord forgetting:   {cf_mean:.3f} ± {cf_ci:.3f}  (n={n_seeds})")

                slug = f"{label.replace(' ', '_')}_{difficulty}"
                out_path = args.out_dir / f"coord_matrix_{slug}.png"
                plot_matrix(
                    mean_norm, layout_names,
                    title=f"Coordination matrix — {label} / {difficulty}  (mean over {n_seeds} seeds)",
                    out_path=out_path,
                )
                if args.latex:
                    print_latex(mean_norm, layout_names, label=f"{label} {difficulty}")

        print_summary_table(summary_results, args.difficulties, latex=args.latex)

    # ----------------------------------------------------------------
    # Single-run mode (original behaviour)
    # ----------------------------------------------------------------
    else:
        if not args.run_dir:
            p.error("Either --run_dir or --checkpoint_root is required")

        run_dirs = [Path(d) for d in args.run_dir]
        labels   = args.labels or [d.name for d in run_dirs]

        for run_dir, label in zip(run_dirs, labels):
            print(f"\n{'='*60}")
            print(f"Run: {label}  ({run_dir})")
            print(f"{'='*60}")

            matrix_norm, matrix_raw, layout_names = compute_matrix(
                run_dir=run_dir, num_envs=args.num_envs, num_steps=args.num_steps,
                seed=args.seed, num_agents=args.num_agents,
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
