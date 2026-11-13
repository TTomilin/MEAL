"""Plot partner-diversity figures from specialist cross-play and bank selection outputs."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from experiments.partner_adaptation.partner_bank import load_partner_bank
from experiments.results.cpa_eval import short
from experiments.results.plotting.utils import save_plot
from experiments.results.plotting.utils import cpa_csv

PERPLEXITY_MAX = 5
HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 1.2
DEFAULT_FORMATS = ["png", "pdf"]
LEARNED_COLOR = "#1f77b4"
PLANNER_COLOR = "#ff7f0e"
KIND_MARKERS = {
    "jaxaht": "o",
    "concord": "s",
    "brdiv": "D",
    "bc_proxy": "^",
    "planner": "X",
}
FALLBACK_MARKER = "o"


def relative_xp(xp):
    diag = np.diag(xp).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return xp / np.where(diag == 0, np.nan, diag)


def tsne_perplexity(n_partners):
    return min(PERPLEXITY_MAX, n_partners - 1)


def read_xp_subset(path, labels):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows or len(rows[0]) < 2:
        raise SystemExit(f"{path}: expected a labelled square matrix")
    header = rows[0][1:]
    col = {lab: j for j, lab in enumerate(header)}
    missing = [lab for lab in labels if lab not in col]
    if missing:
        raise SystemExit(f"{path}: missing columns for {missing}")
    by_row = {}
    for row in rows[1:]:
        if len(row) != len(header) + 1:
            raise SystemExit(f"{path}: row {row[0]!r} has the wrong width")
        by_row[row[0]] = row[1:]
    missing = [lab for lab in labels if lab not in by_row]
    if missing:
        raise SystemExit(f"{path}: missing rows for {missing}")
    p = len(labels)
    xp = np.empty((p, p), float)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            xp[i, j] = float(by_row[li][col[lj]])
    return xp


def read_distinctness(path, labels):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_label = {r["label"]: r for r in rows}
    missing = [lab for lab in labels if lab not in by_label]
    if missing:
        raise SystemExit(f"{path}: missing partners {missing}")
    return [by_label[lab] for lab in labels]


def _row_shares(counts):
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / total


def behaviour_features(action_counts, visits, pot_interactions):
    """Per-partner features: mean over specialists of action / visit / pot shares."""
    p = action_counts.shape[0]
    feats = []
    for j in range(p):
        act = np.mean([_row_shares(action_counts[i, j]) for i in range(p)], axis=0)
        vis = np.mean([_row_shares(visits[i, j]) for i in range(p)], axis=0)
        pot = np.mean([_row_shares(pot_interactions[i, j]) for i in range(p)], axis=0)
        feats.append(np.concatenate([act, vis, pot]))
    return np.asarray(feats, float)


def load_behaviour_subset(path, labels):
    data = np.load(path)
    beh_labels = [str(x) for x in data["labels"]]
    index = {lab: i for i, lab in enumerate(beh_labels)}
    missing = [lab for lab in labels if lab not in index]
    if missing:
        raise SystemExit(f"{path}: missing partners {missing}")
    idx = [index[lab] for lab in labels]
    return (data["action_counts"][np.ix_(idx, idx)],
            data["visits"][np.ix_(idx, idx)],
            data["pot_interactions"][np.ix_(idx, idx)])


def first_planner_index(kinds):
    for i, kind in enumerate(kinds):
        if kind == "planner":
            return i
    return None


def outline_cells(ax, cells):
    for i, j in cells:
        ax.add_patch(mpatches.Rectangle(
            (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.8, zorder=3))


def plot_heatmap(rel, labels, kinds, distinct_rows, layout_name, out_dir, formats):
    p = len(labels)
    short_labels = [short(lab) for lab in labels]
    n_distinct = sum(r["distinct"] == "true" for r in distinct_rows)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * p + 2.5), max(5.0, 0.55 * p + 2.0)))
    im = ax.imshow(rel, vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, cmap="viridis", aspect="equal")
    ax.set_xticks(range(p))
    ax.set_yticks(range(p))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("partner")
    ax.set_ylabel("specialist")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    planner_i = first_planner_index(kinds)
    if planner_i is not None and 0 < planner_i < p:
        ax.axhline(planner_i - 0.5, color="white", lw=1.4)
        ax.axvline(planner_i - 0.5, color="white", lw=1.4)

    cells = []
    label_index = {lab: i for i, lab in enumerate(labels)}
    for j, row in enumerate(distinct_rows):
        if row["distinct"] != "false":
            continue
        closest = row.get("closest_specialist") or ""
        if closest in label_index:
            cells.append((label_index[closest], j))
    outline_cells(ax, cells)

    ax.set_title(f"{layout_name}: {n_distinct} distinct of {p}")
    fig.tight_layout()
    save_plot(fig, out_dir, f"{layout_name}_crossplay_heatmap", formats)
    return fig


def plot_tsne(features, labels, kinds, layout_name, seed, out_dir, formats):
    p = features.shape[0]
    if p < 2:
        raise SystemExit("t-SNE needs at least 2 partners")
    scaled = StandardScaler().fit_transform(features)
    perp = tsne_perplexity(p)
    coords = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca",
                  learning_rate="auto").fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for i, (lab, kind) in enumerate(zip(labels, kinds)):
        color = PLANNER_COLOR if kind == "planner" else LEARNED_COLOR
        marker = KIND_MARKERS.get(kind, FALLBACK_MARKER)
        ax.scatter(coords[i, 0], coords[i, 1], c=color, marker=marker, s=70,
                   edgecolors="black", linewidths=0.4, zorder=2)
        ax.annotate(short(lab), (coords[i, 0], coords[i, 1]), textcoords="offset points",
                    xytext=(4, 4), fontsize=7)
    learned_handle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=LEARNED_COLOR,
                                markeredgecolor="black", markersize=8, label="learned")
    planner_handle = plt.Line2D([0], [0], marker="X", color="w", markerfacecolor=PLANNER_COLOR,
                                markeredgecolor="black", markersize=8, label="planner")
    ax.legend(handles=[learned_handle, planner_handle], frameon=True, fontsize=9)
    ax.set_title(f"{layout_name}: partner behaviour t-SNE")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.text(0.5, 0.02, "t-SNE distances are only meaningful locally.",
             ha="center", fontsize=8, color="#333333")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_plot(fig, out_dir, f"{layout_name}_behaviour_tsne", formats)

    csv_path = out_dir / f"{layout_name}_behaviour_tsne.csv"
    with open(csv_path, "w", newline="") as f:
        feat_cols = [f"f{k}" for k in range(features.shape[1])]
        w = csv.DictWriter(f, fieldnames=["label", "kind", "x", "y"] + feat_cols)
        w.writeheader()
        for i, (lab, kind) in enumerate(zip(labels, kinds)):
            row = {"label": lab, "kind": kind, "x": f"{coords[i, 0]:.6g}", "y": f"{coords[i, 1]:.6g}"}
            for k, val in enumerate(features[i]):
                row[f"f{k}"] = f"{val:.6g}"
            w.writerow(row)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__)
    ap.add_argument("--matrix", required=True, help="xp_matrix.csv from specialist_crossplay")
    ap.add_argument("--behaviour", required=True, help="partner_behaviour.npz from specialist_crossplay")
    ap.add_argument("--bank", required=True, help="main bank manifest (partners plotted, in bank order)")
    ap.add_argument("--distinctness", required=True, help="distinctness.csv from crossplay")
    ap.add_argument("--layout_name", required=True)
    ap.add_argument("--seed", type=int, default=0, help="t-SNE random_state")
    ap.add_argument("--output_dir", default=None, help="Where to save figures (default: experiments/results/plots)")
    ap.add_argument("--no_show", action="store_true", help="Save only; do not open a window (headless backend)")
    ap.add_argument("--formats", nargs="+", choices=["png", "pdf"], default=None,
                    help="Figure formats (default: png, pdf)")
    args = ap.parse_args(argv)

    out_dir, formats = cpa_csv.prepare_output(
        args, Path(__file__).resolve().parent.parent / "plots", DEFAULT_FORMATS)

    bank = load_partner_bank(args.bank, args.layout_name)
    labels = [r.label for r in bank.records]
    kinds = [r.kind for r in bank.records]

    xp = read_xp_subset(args.matrix, labels)
    rel = relative_xp(xp)
    distinct_rows = read_distinctness(args.distinctness, labels)
    fig1 = plot_heatmap(rel, labels, kinds, distinct_rows, args.layout_name, out_dir, formats)

    action_counts, visits, pot_interactions = load_behaviour_subset(args.behaviour, labels)
    features = behaviour_features(action_counts, visits, pot_interactions)
    fig2 = plot_tsne(features, labels, kinds, args.layout_name, args.seed, out_dir, formats)

    if not args.no_show:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    main()
