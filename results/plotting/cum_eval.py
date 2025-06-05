from pathlib import Path

import numpy as np

from results.plotting.utils import (
    collect_runs, setup_figure, add_task_boundaries,
    setup_task_axes, smooth_and_ci, save_plot, finalize_plot,
    METHOD_COLORS, create_eval_parser
)


def parse_args():
    """Parse command line arguments for the training plot script."""
    p = create_eval_parser(
        description="Plot training metrics for MARL continual-learning benchmark",
        metric_choices=['reward', 'soup']
    )
    # Set default confidence level to 0.9 (overriding the default from create_eval_parser)
    p.set_defaults(confidence=0.9, metric='soup')
    # Add script-specific arguments
    p.add_argument('--legend_anchor', type=float, default=0.87, help="Legend anchor position")
    return p.parse_args()


def plot():
    """
    Main plotting function for training metrics.

    Collects data for each method, creates a plot with method curves,
    adds task boundaries, and saves the plot.
    """
    args = parse_args()
    data_root = Path(__file__).resolve().parent.parent / args.data_root

    # Calculate total steps and set up figure
    total_steps = args.seq_len * args.steps_per_task
    width = min(max(args.seq_len, 8), 14)
    fig, ax = setup_figure(width=width, height=4)

    # Dictionary to store data for each method
    method_data = {}

    # Collect data for each method
    for method in args.methods:
        data, env_names = collect_runs(
            data_root, args.algo, method, args.strategy, args.seq_len, args.seeds, args.metric
        )
        method_data[method] = data

        # Calculate smoothed mean and confidence interval
        mu, ci = smooth_and_ci(data, args.sigma, args.confidence)

        # Plot the method curve
        x = np.linspace(0, total_steps, len(mu))
        color = METHOD_COLORS.get(method)
        ax.plot(x, mu, label=method, color=color)
        ax.fill_between(x, mu - ci, mu + ci, color=color, alpha=0.2)

    # Add task boundaries
    boundaries = [i * args.steps_per_task for i in range(args.seq_len + 1)]
    add_task_boundaries(ax, boundaries)

    # Set up task axes (primary and secondary x-axes)
    setup_task_axes(ax, boundaries, args.seq_len)

    # Finalize plot with labels, limits, and legend
    finalize_plot(
        ax,
        xlabel='Environment Steps',
        ylabel=f'{args.metric.capitalize()} Delivered Normalized',
        xlim=(0, total_steps),
        ylim=(0, None),
        legend_loc='lower center',
        legend_bbox_to_anchor=(0.5, args.legend_anchor),
        legend_ncol=len(args.methods)
    )

    # Save the plot
    out_dir = Path(__file__).resolve().parent.parent / 'plots'
    stem = args.plot_name or f"avg_norm_{args.metric}"
    save_plot(fig, out_dir, stem)

    # Display the plot
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    plot()
