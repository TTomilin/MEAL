import os
import uuid
from datetime import datetime
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
import wandb
from dotenv import load_dotenv
from flax.core.frozen_dict import FrozenDict
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from jax_marl.environments.env_selection import generate_sequence


class Transition(NamedTuple):
    '''
    Named tuple to store the transition information
    '''
    done: jnp.ndarray  # whether the episode is done
    action: jnp.ndarray  # the action taken
    value: jnp.ndarray  # the value of the state
    reward: jnp.ndarray  # the reward received
    log_prob: jnp.ndarray  # the log probability of the action
    obs: jnp.ndarray  # the observation
    # info: jnp.ndarray # additional information


class Transition_CNN(NamedTuple):
    '''
    Named tuple to store the transition information
    '''
    done: jnp.ndarray  # whether the episode is done
    action: jnp.ndarray  # the action taken
    value: jnp.ndarray  # the value of the state
    reward: jnp.ndarray  # the reward received
    log_prob: jnp.ndarray  # the log probability of the action
    obs: jnp.ndarray  # the observation
    info: jnp.ndarray  # additional information


def batchify(x: dict, agent_list, num_actors, flatten=True):
    '''
    converts the observations of a batch of agents into an array of size (num_actors, -1) that can be used by the network
    @param flatten: for MLP architectures
    @param x: dictionary of observations
    @param agent_list: list of agents
    @param num_actors: number of actors
    returns the batchified observations
    '''
    x = jnp.stack([x[a] for a in agent_list])
    batched = jnp.concatenate(x, axis=0)
    if flatten:
        batched = batched.reshape((num_actors, -1))
    return batched


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    '''
    converts the array of size (num_actors, -1) into a dictionary of observations for all agents
    @param unflatten: for MLP architectures
    @param x: array of observations
    @param agent_list: list of agents
    @param num_envs: number of environments
    @param num_actors: number of actors
    returns the unbatchified observations
    '''
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def sample_discrete_action(key, action_space):
    """Samples a discrete action based on the action space provided."""
    num_actions = action_space.n
    return jax.random.randint(key, (1,), 0, num_actions)


def make_task_onehot(task_idx: int, num_tasks: int) -> jnp.ndarray:
    """
    Returns a one-hot vector of length `num_tasks` with a 1 at `task_idx`.
    """
    return jnp.eye(num_tasks, dtype=jnp.float32)[task_idx]


def copy_params(params):
    return jax.tree_util.tree_map(lambda x: x.copy(), params)


def compute_fwt(matrix):
    """
    Computes the forward transfer for all tasks in a sequence
    param matrix: a 2D array of shape (num_tasks + 1, num_tasks) where each entry is the performance of the model on the task
    """
    # Assert that the matrix has the correct shape
    assert matrix.shape[0] == matrix.shape[1] + 1, "Matrix must have shape (num_tasks + 1, num_tasks)"

    num_tasks = matrix.shape[1]

    fwt_matrix = np.full((num_tasks, num_tasks), np.nan)

    for i in range(1, num_tasks):
        for j in range(i):  # j < i
            before_learning = matrix[0, i]
            after_task_j = matrix[j + 1, i]
            fwt_matrix[i, j] = after_task_j - before_learning

    return fwt_matrix


def compute_bwt(matrix):
    """
    Computes the backward transfer for all tasks in a sequence
    param matrix: a 2D array of shape (num_tasks + 1, num_tasks) where each entry is the performance of the model on the task
    """
    assert matrix.shape[0] == matrix.shape[1] + 1, "Matrix must have shape (num_tasks + 1, num_tasks)"
    num_tasks = matrix.shape[1]

    bwt_matrix = jnp.full((num_tasks, num_tasks), jnp.nan)

    for i in range(num_tasks - 1):
        for j in range(i + 1, num_tasks):
            after_j = matrix[j + 1, i]  # performance on task i after learning task j
            after_i = matrix[i + 1, i]  # performance on task i after learning task i
            bwt_matrix = bwt_matrix.at[i, j].set(after_j - after_i)

    return bwt_matrix


def show_heatmap_bwt(matrix, run_name, save_folder="heatmap_images"):
    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    bwt_matrix = compute_bwt(matrix)
    avg_bwt_per_step = np.nanmean(bwt_matrix, axis=0)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(bwt_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                xticklabels=[f"Task {j}" for j in range(bwt_matrix.shape[1])],
                yticklabels=[f"Task {i}" for i in range(bwt_matrix.shape[0])],
                cbar_kws={"label": "BWT"})
    ax.set_title("Progressive Backward Transfer Matrix")
    ax.set_xlabel("Task B")
    ax.set_ylabel("Task A")
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

    # Add average BWT per step below the heatmap
    for j, val in enumerate(avg_bwt_per_step):
        if not np.isnan(val):
            ax.text(j + 0.5, len(avg_bwt_per_step) + 0.2, f"{val:.2f}",
                    ha='center', va='bottom', fontsize=9, color='black')
    plt.text(-0.7, len(avg_bwt_per_step) + 0.2, "Avg", fontsize=10, va='bottom', weight='bold')

    plt.tight_layout()

    # Save the figure
    file_path = os.path.join(save_folder, f"{run_name}_bwt_heatmap.png")
    plt.savefig(file_path)
    plt.close()


def show_heatmap_fwt(matrix, run_name, save_folder="heatmap_images"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    fwt_matrix = compute_fwt(matrix)
    avg_fwt_per_step = np.nanmean(fwt_matrix, axis=0)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(fwt_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                xticklabels=[f"Task {j}" for j in range(fwt_matrix.shape[1])],
                yticklabels=[f"Task {i}" for i in range(fwt_matrix.shape[0])],
                cbar_kws={"label": "FWT"})
    ax.set_title("Progressive Forward Transfer Matrix")
    ax.set_xlabel("Task B")
    ax.set_ylabel("Task A")

    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

    for j, val in enumerate(avg_fwt_per_step):
        if not np.isnan(val):
            ax.text(j + 0.5, len(avg_fwt_per_step) + 0.2, f"{val:.2f}",
                    ha='center', va='bottom', fontsize=9, color='black')

    plt.text(-0.7, len(avg_fwt_per_step) + 0.2, "Avg", fontsize=10, va='bottom', weight='bold')

    plt.tight_layout()

    file_path = os.path.join(save_folder, f"{run_name}_fwt_heatmap.png")
    plt.savefig(file_path)
    plt.close()


def add_eval_metrics(avg_rewards, avg_soups, layout_names, max_soup_dict, metrics):
    for i, layout_name in enumerate(layout_names):
        metrics[f"Evaluation/Returns/{i}__{layout_name}"] = avg_rewards[i]
        metrics[f"Evaluation/Soup/{i}__{layout_name}"] = avg_soups[i]
        metrics[f"Evaluation/Soup_Scaled/{i}__{layout_name}"] = avg_soups[i] / max_soup_dict[layout_name]
    return metrics


def build_reg_weights(params, regularize_critic: bool, regularize_heads: bool) -> FrozenDict:
    def _mark(path, x):
        path_str = "/".join(map(str, path)).lower()
        if not regularize_heads and ("actor_head" in path_str or "critic_head" in path_str):
            return jnp.zeros_like(x)
        if not regularize_critic and "critic" in path_str:
            return jnp.zeros_like(x)
        return jnp.ones_like(x)

    return jax.tree_util.tree_map_with_path(_mark, params)


# -------------------------------------------------------------------
# helper: pad (or crop) an (H,W,C) image to `target_shape`
#         – no tracers in pad_width, 100 % JIT-safe
# -------------------------------------------------------------------
def _pad_to(img: jnp.ndarray, target_shape):
    th, tw, tc = target_shape  # target (height, width, channels)
    h, w, c = img.shape  # current shape – *Python* ints
    assert c == tc, "channel mismatch"

    dh = th - h  # + ⇒ need pad, − ⇒ need crop
    dw = tw - w

    # amounts have to be Python ints so jnp.pad sees concrete values
    pad_top = max(dh // 2, 0)
    pad_bottom = max(dh - pad_top, 0)
    pad_left = max(dw // 2, 0)
    pad_right = max(dw - pad_left, 0)

    img = jnp.pad(
        img,
        ((pad_top, pad_bottom),
         (pad_left, pad_right),
         (0, 0)),  # no channel padding
        mode="constant",
    )

    # If the image was *larger* than the target we crop back
    return img[:th, :tw, :]


# ---------------------------------------------------------------
# util: build a (2, …) batch without Python branches
# ---------------------------------------------------------------
def _prep_obs(raw_obs: dict[str, jnp.ndarray], use_cnn: bool) -> jnp.ndarray:
    """
    Stack per‐agent observations into a single array of shape
    (num_agents, …).

    If use_cnn=False, each obs is flattened to a 1D float32 vector first.
    """

    def _single(obs: jnp.ndarray) -> jnp.ndarray:
        # flatten & cast when not using CNN
        if not use_cnn:
            obs = obs.reshape(-1).astype(jnp.float32)
        # introduce a leading "agent" axis
        return obs[None]

    # Sort the keys so that the agent‐ordering is deterministic
    agent_ids = sorted(raw_obs.keys())

    # Build a list of (1, …) arrays, one per agent
    per_agent = [_single(raw_obs[a]) for a in agent_ids]

    # Concatenate along the new leading axis → (num_agents, …)
    return jnp.concatenate(per_agent, axis=0)


def generate_sequence_of_tasks(config):
    """
    Generates a sequence of tasks based on the provided configuration.
    """
    config.env_kwargs, config.layout_name = generate_sequence(
        sequence_length=config.seq_length,
        strategy=config.strategy,
        layout_names=config.layouts,
        seed=config.seed
    )

    # for layout_config in config.env_kwargs:
    #     layout_name = layout_config["layout"]
    #     layout_config["layout"] = overcooked_layouts[layout_name]

    return config


def create_run_name(config, network_architecture):
    """
    Generates a unique run name based on the config, current timestamp and a UUID.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = uuid.uuid4()
    run_name = f'{config.alg_name}_{config.cl_method}_{network_architecture}_\
        seq{config.seq_length}_{config.strategy}_seed_{config.seed}_{timestamp}_{unique_id}'
    return run_name


def initialize_logging_setup(config, run_name, exp_dir):
    """
    Initializes WandB and TensorBoard logging setup.
    """
    # Initialize WandB
    load_dotenv()
    wandb_tags = config.tags if config.tags is not None else []
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project=config.project,
        config=config,
        sync_tensorboard=True,
        mode=config.wandb_mode,
        name=run_name,
        id=run_name,
        tags=wandb_tags,
        group=config.group
    )

    # Set up Tensorboard
    writer = SummaryWriter(exp_dir)

    # add the hyperparameters to the tensorboard
    rows = []
    for key, value in vars(config).items():
        value_str = str(value).replace("\n", "<br>")
        value_str = value_str.replace("|", "\\|")  # escape pipe chars if needed
        rows.append(f"|{key}|{value_str}|")

    table_body = "\n".join(rows)
    markdown = f"|param|value|\n|-|-|\n{table_body}"
    writer.add_text("hyperparameters", markdown)

    return writer
