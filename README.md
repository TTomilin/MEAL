# MEAL: A Benchmark for Continual Multi-Agent Reinforcement Learning

MEAL is the first **Continual Multi‑Agent Reinforcement Learning (CMARL)** benchmark built around cooperative 
Overcooked‑style tasks, implemented in JAX for high‑performance training and evaluation. It focuses on learning over 
extensive sequences of procedurally generated tasks, across different team sizes and difficulty levels.

<div align="center">
  <table>
    <tr>
      <th>Overcooked</th><th>MPE</th><th>SMAX</th><th>JaxNav</th>
    </tr>
    <tr>
      <td style="padding: 4px; width: 25%"><img src="./docs/assets/gifs/2_agents_med.gif" width="100%" /></td>
      <td style="padding: 4px; width: 25%"><img src="./docs/assets/gifs/MPE.gif" width="100%" /></td>
      <td style="padding: 4px; width: 25%"><img src="./docs/assets/gifs/SMAX.gif" width="100%" /></td>
      <td style="padding: 4px; width: 25%"><img src="./docs/assets/gifs/JaxNav.gif" width="100%" /></td>
    </tr>
  </table>
</div>

## Key Features

- JAX/Flax implementation for scalable, accelerated training
- Procedurally generated cooperative tasks with adjustable difficulty, across four environments
  (Overcooked, MPE, SMAX, JaxNav)
- Multi-agent algorithms: IPPO, MAPPO, HAPPO, VDN, QMIX
- Continual learning methods: EWC, MAS, L2, FT, AGEM, ER-ACE, PackNet
- Results tooling: W&B integration, download utilities, and plotting scripts

## Installation

Requires Python 3.10.

```bash
# Create and activate an environment (Conda example)
conda create -n meal python=3.10 -y
conda activate meal

# Install MEAL in editable mode and optional extras
pip install -e .
pip install -e ".[viz]"
pip install -e ".[utils]"

# Optional: GPU acceleration for JAX (pick your CUDA version)
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# or
pip install -U "jax[cuda11]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Quick Start

The single entry point is `experiments/train.py`. It takes the MARL algorithm as the first
argument (`ippo`, `mappo`, `happo`, `vdn`, `qmix`), then the continual-learning method, then
the environment (`overcooked`, `mpe`, `smax`, `jaxnav`) as a subcommand. Outer flags must
come before the `env:...` subcommand token.

### Example: IPPO + EWC on generated medium Overcooked tasks with 2 agents

```bash
python -m experiments.train ippo \
  --cl-method ewc \
  --seq-length 20 \
  --num-agents 2 \
  --num-envs 2048 \
  --num-steps 400 \
  env:overcooked \
  --env.difficulty medium
```

Swap `ippo` for `mappo`/`happo`/`vdn`/`qmix`, `--cl-method ewc` for any of `mas`, `l2`, `ft`,
`agem`, `er_ace`, `packnet`, and `env:overcooked` for `env:mpe`/`env:smax`/`env:jaxnav`. Every
combination shares this same CLI shape. Full flag reference, including every algorithm- and
environment-specific option, is in [experiments/README.MD](experiments/README.MD).

### Running Experiments
For running experiments, please refer to [experiments/README.MD](experiments/README.MD).
To reproduce the experiments from the paper specifically, see
[scripts/README.MD](scripts/README.MD).

## Python API

Besides the training CLI, MEAL is directly importable as a library. For writing your own
training loop, or just poking at an environment:

```python
import meal

env = meal.make_env('overcooked', difficulty='medium')
obs, state = env.reset(reset_key)
obs, state, reward, done, info = env.step(step_key, state, actions)

# A continual sequence of tasks, same generation logic the CLI uses:
tasks = meal.make_sequence(sequence_length=6, num_agents=3, difficulty='hard')
```

See [examples/](examples/README.md) for small, runnable scripts covering each environment, CL
task sequences, partial observability, and stochastic-transition wrappers.

## Environments

MEAL benchmarks continual learning across four environments, each with its own README covering
observation/action space, reward structure, how CL task diversity is generated, and every
`--env.*` flag it exposes:

- [Overcooked](meal/env/overcooked/README.md) — cooperative kitchen, procedurally generated layouts
- [MPE](meal/env/mpe/README.md) — particle-agent coverage with obstacles
- [SMAX](meal/env/smax/README.md) — StarCraft-style unit-composition battles
- [JaxNav](meal/env/jaxnav/README.md) — multi-robot 2D navigation

For details on how Overcooked layouts themselves are procedurally generated, see
[meal/README.MD](meal/README.MD).

## Project Structure

- `experiments/`
  - `train.py`: single training entry point (algo x cl-method x env)
  - `algos/`: AbstractAlgo/OnPolicyAlgo/OffPolicyAlgo hierarchy (IPPO, MAPPO, HAPPO, VDN, QMIX)
  - `envs/`: EnvAdapter per environment (Overcooked, MPE, SMAX, JaxNav)
  - `continual/`: implementations of EWC, MAS, L2, FT, AGEM, ER-ACE, PackNet
  - `results/`: W&B downloaders and plotting scripts
- `scripts/`: paper-reproduction sweeps (see [scripts/README.MD](scripts/README.MD)) + env visualization tooling
- `meal/`
  - `env/`: layouts and utilities
  - `wrappers/`: logging and environment wrappers
  - `visualization/`: rendering utilities
- `tests/`: environment smoke tests, image comparisons, and algorithm-level regression tests

## Documentation

Every subdirectory with its own concerns has a dedicated README:

| Topic | README |
| --- | --- |
| Full training CLI reference (all flags, per-algo and per-CL-method options) | [experiments/README.MD](experiments/README.MD) |
| Downloading W&B run data and generating result tables/figures | [experiments/results/README.MD](experiments/results/README.MD) |
| Reproducing the paper's experiment sweeps | [scripts/README.MD](scripts/README.MD) |
| Runnable library-API example scripts (no training CLI) | [examples/README.md](examples/README.md) |
| Procedural Overcooked layout generation | [meal/README.MD](meal/README.MD) |
| Overcooked environment details | [meal/env/overcooked/README.md](meal/env/overcooked/README.md) |
| MPE environment details | [meal/env/mpe/README.md](meal/env/mpe/README.md) |
| SMAX environment details | [meal/env/smax/README.md](meal/env/smax/README.md) |
| JaxNav environment details | [meal/env/jaxnav/README.md](meal/env/jaxnav/README.md) |

## Acknowledgments

- The environments are based on [JaxMARL](https://github.com/FLAIROx/JaxMARL).
- Our experiments were managed using [WandB](https://wandb.ai).

## Citation
If you use our work in your research, please cite it as follows:
```
@article{tomilin2026meal,
  title={MEAL: A Benchmark for Continual Multi-Agent Reinforcement Learning},
  author={Tomilin, Tristan and van den Boogaard, Luka and Garcin, Samuel and Ruhdorfer, Constantin and Grooten, Bram and Kusters, Fabrice and Du, Yali and Bulling, Andreas and Pechenizkiy, Mykola and Fang, Meng},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```