"""Matplotlib-based visualiser for SMAX battle rollouts.

Blue circles = ally units (label = unit type shorthand).
Red circles  = enemy units (label = unit type shorthand).
Unit types: m=marine, M=marauder, s=stalker, Z=zealot, z=zergling, h=hydralisk

Promoted from the standalone `scripts/visualize_smax.py` tool so it can be
reused for trained-policy video recording during training (see
`experiments/envs/smax.py::SMAXAdapter.build_visualizer`).

Consumes a sequence of *inner* SMAX states (i.e. already unwrapped past the
`HeuristicEnemySMAX`/`LogWrapper` layers -- `state.unit_positions` etc, not
the outer wrapper state).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


class SMAXVisualizer:
    """Animate a sequence of SMAX inner states (unit positions, health, alive)."""

    ALLY_COLOR = "#4fc3f7"  # light blue
    ENEMY_COLOR = "#ef9a9a"  # light red
    DEAD_ALPHA = 0.15
    BG_COLOR = "#1a1a2e"

    def __init__(self, env, state_seq: list, map_id: str):
        """`env` is the (possibly-wrapped) HeuristicEnemySMAX env -- only used
        for static layout info (map size, unit counts/types), so `env._env`
        (the raw SMAX instance) is what actually gets read from."""
        self._inner_env = env._env
        self.state_seq = state_seq
        self.map_id = map_id
        self._init_figure()

    def _init_figure(self):
        smax = self._inner_env
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.patch.set_facecolor(self.BG_COLOR)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)
        ax = self.ax
        ax.set_facecolor(self.BG_COLOR)
        ax.set_xlim(0, smax.map_width)
        ax.set_ylim(0, smax.map_height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        state0 = self.state_seq[0]

        self._patches = []
        self._labels = []

        for i in range(smax.num_allies):
            pos = np.array(state0.unit_positions[i])
            r = float(smax.unit_type_radiuses[state0.unit_types[i]])
            sh = smax.unit_type_shorthands[int(state0.unit_types[i])]
            c = Circle(pos, r, facecolor=self.ALLY_COLOR, edgecolor="white",
                      linewidth=0.8, zorder=2)
            ax.add_patch(c)
            txt = ax.text(pos[0], pos[1], sh, ha="center", va="center",
                          fontsize=5, color="black", fontweight="bold", zorder=3)
            self._patches.append(c)
            self._labels.append(txt)

        for i in range(smax.num_enemies):
            idx = i + smax.num_allies
            pos = np.array(state0.unit_positions[idx])
            r = float(smax.unit_type_radiuses[state0.unit_types[idx]])
            sh = smax.unit_type_shorthands[int(state0.unit_types[idx])]
            c = Circle(pos, r, facecolor=self.ENEMY_COLOR, edgecolor="white",
                      linewidth=0.8, zorder=2)
            ax.add_patch(c)
            txt = ax.text(pos[0], pos[1], sh, ha="center", va="center",
                          fontsize=5, color="black", fontweight="bold", zorder=3)
            self._patches.append(c)
            self._labels.append(txt)

        self._step_text = ax.text(
            0.5, 1.02, f"Step 0 | {self.map_id}",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=7, color="white",
        )

    def _update(self, frame: int):
        smax = self._inner_env
        state = self.state_seq[frame]

        for i in range(smax.num_allies):
            pos = np.array(state.unit_positions[i])
            alive = bool(state.unit_alive[i])
            p = self._patches[i]
            t = self._labels[i]
            p.center = tuple(pos)
            t.set_position(pos)
            p.set_alpha(1.0 if alive else self.DEAD_ALPHA)
            t.set_alpha(1.0 if alive else 0.0)

        for i in range(smax.num_enemies):
            idx = i + smax.num_allies
            pos = np.array(state.unit_positions[idx])
            alive = bool(state.unit_alive[idx])
            pidx = smax.num_allies + i
            p = self._patches[pidx]
            t = self._labels[pidx]
            p.center = tuple(pos)
            t.set_position(pos)
            p.set_alpha(1.0 if alive else self.DEAD_ALPHA)
            t.set_alpha(1.0 if alive else 0.0)

        self._step_text.set_text(f"Step {frame} | {self.map_id}")

    def save_frame(self, save_fname: str):
        self.fig.savefig(save_fname, dpi=150, pad_inches=0,
                         facecolor=self.fig.get_facecolor())

    def animate(self, save_fname: str, fps: int = 10):
        ani = animation.FuncAnimation(
            self.fig, self._update,
            frames=len(self.state_seq),
            interval=1000 // fps,
            blit=False,
        )
        ani.save(save_fname, writer="ffmpeg", fps=fps, extra_args=["-pix_fmt", "yuv420p"],
                 savefig_kwargs={"pad_inches": 0, "facecolor": self.fig.get_facecolor()})
        plt.close(self.fig)
