"""Matplotlib-based visualiser for MPESpreadEnv.

Renders agents (solid circles), landmarks (X markers), and obstacles
(gray filled circles with outline) at each step. Promoted from the standalone
`scripts/visualize_mpe.py` tool so it can be reused for trained-policy video
recording during training (see `experiments/envs/mpe.py::MPEAdapter.build_visualizer`).
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class MPEObstacleVisualizer:
    AGENT_COLORS = [
        "#4CAF50",  # green
        "#2196F3",  # blue
        "#FF5722",  # orange-red
        "#9C27B0",  # purple
        "#00BCD4",  # cyan
    ]
    LANDMARK_COLOR = "#424242"   # dark grey
    OBSTACLE_COLOR = "#9E9E9E"   # medium grey
    BG_COLOR = "#FAFAFA"

    def __init__(self, env, state_seq: list):
        self.env = env
        self.state_seq = state_seq
        self._init_figure()

    def _init_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.patch.set_facecolor(self.BG_COLOR)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set_facecolor(self.BG_COLOR)
        lim = 1.3
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        state = self.state_seq[0]
        A = self.env.num_agents
        L = self.env._num_goals
        K = self.env._num_obstacles

        # Draw obstacles (static -- drawn once, never updated)
        for i in range(K):
            idx = A + L + i
            pos = np.array(state.p_pos[idx])
            r = float(self.env.rad[idx])
            circ = Circle(pos, r, facecolor=self.OBSTACLE_COLOR, edgecolor="#616161",
                          linewidth=1.5, zorder=1)
            self.ax.add_patch(circ)

        # Landmark markers (animated)
        self._landmark_patches = []
        for i in range(L):
            idx = A + i
            pos = np.array(state.p_pos[idx])
            r = float(self.env.rad[idx])
            circ = Circle(pos, r, color=self.LANDMARK_COLOR, zorder=2)
            self.ax.add_patch(circ)
            self._landmark_patches.append(circ)

        # Agent circles (animated)
        self._agent_patches = []
        for i in range(A):
            pos = np.array(state.p_pos[i])
            r = float(self.env.rad[i])
            color = self.AGENT_COLORS[i % len(self.AGENT_COLORS)]
            circ = Circle(pos, r, color=color, zorder=3)
            self.ax.add_patch(circ)
            self._agent_patches.append(circ)

        self._step_text = self.ax.text(
            -1.25, 1.18, f"Step 0 | {self.env.map_id}",
            fontsize=7, va="top", color="#333333"
        )

    def _update(self, frame: int):
        state = self.state_seq[frame]
        A = self.env.num_agents
        L = self.env._num_goals

        for i, patch in enumerate(self._agent_patches):
            patch.center = tuple(np.array(state.p_pos[i]))

        for i, patch in enumerate(self._landmark_patches):
            patch.center = tuple(np.array(state.p_pos[A + i]))

        self._step_text.set_text(f"Step {frame} | {self.env.map_id}")

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
