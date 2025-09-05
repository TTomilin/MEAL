import json
import os
from pathlib import Path

from jax._src.flatten_util import ravel_pytree

from cl_methods.AGEM import AGEM, init_agem_memory, sample_memory, compute_memory_gradient, agem_project, \
    update_agem_memory
from cl_methods.FT import FT
from cl_methods.L2 import L2
from cl_methods.MAS import MAS

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
from typing import Sequence, Any, Optional, List

import flax
import optax
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState

from jax_marl.registration import make
from jax_marl.eval.visualizer import OvercookedVisualizer
from jax_marl.eval.visualizer_po import OvercookedVisualizerPO
from jax_marl.wrappers.baselines import LogWrapper
from jax_marl.environments.overcooked.upper_bound import estimate_max_soup
from architectures.mlp import ActorCritic as MLPActorCritic
from architectures.cnn import ActorCritic as CNNActorCritic
from baselines.utils import *
from cl_methods.EWC import EWC
from jax_marl.environments.difficulty_config import apply_difficulty_to_config

import wandb
from functools import partial
from dataclasses import dataclass, field
import tyro
from tensorboardX import SummaryWriter


@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING / PPO PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    alg_name: str = "ippo"
    lr: float = 3e-4
    anneal_lr: bool = False
    num_envs: int = 16
    num_steps: int = 128
    steps_per_task: float = 1e7
    update_epochs: int = 8
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.957
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Reward shaping
    reward_shaping: bool = True
    reward_shaping_horizon: float = 2.5e6

    # Reward distribution settings
    sparse_rewards: bool = False  # Only shared reward for soup delivery
    individual_rewards: bool = False  # Only respective agent gets reward for their actions

    # ═══════════════════════════════════════════════════════════════════════════
    # NETWORK ARCHITECTURE PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    activation: str = "relu"
    use_cnn: bool = False
    use_layer_norm: bool = True
    big_network: bool = False

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
    regularize_heads: bool = False

    # Regularization method specific parameters
    importance_episodes: int = 5
    importance_steps: int = 500

    # EWC specific parameters
    ewc_mode: str = "multi"  # "online", "last" or "multi"
    ewc_decay: float = 0.9  # Only for online EWC

    # AGEM specific parameters
    agem_memory_size: int = 100000
    agem_sample_size: int = 1024
    agem_gradient_scale: float = 1.0

    # ═══════════════════════════════════════════════════════════════════════════
    # ENVIRONMENT PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    env_name: str = "overcooked"
    seq_length: int = 10
    repeat_sequence: int = 1
    strategy: str = "generate"
    layouts: Optional[Sequence[str]] = field(default_factory=lambda: [])
    env_kwargs: Optional[Sequence[dict]] = None
    difficulty: Optional[str] = None
    single_task_idx: Optional[int] = None
    layout_file: Optional[str] = None
    random_reset: bool = True

    # Random layout generator parameters
    height_min: int = 6  # minimum layout height
    height_max: int = 7  # maximum layout height
    width_min: int = 6  # minimum layout width
    width_max: int = 7  # maximum layout width
    wall_density: float = 0.15  # fraction of internal tiles that are untraversable

    # Agent restriction parameters
    complementary_restrictions: bool = False  # One agent can't pick up onions, other can't pick up plates

    # ═══════════════════════════════════════════════════════════════════════════
    # EVALUATION PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    evaluation: bool = True
    eval_forward_transfer: bool = False
    eval_num_steps: int = 1000
    eval_num_episodes: int = 5
    record_gif: bool = True
    gif_len: int = 300
    log_interval: int = 75

    # ═══════════════════════════════════════════════════════════════════════════
    # LOGGING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    wandb_mode: str = "online"
    entity: Optional[str] = ""
    project: str = "MEAL"
    tags: List[str] = field(default_factory=list)

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    seed: int = 30
    num_seeds: int = 1

    # ═══════════════════════════════════════════════════════════════════════════
    # RUNTIME COMPUTED PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    num_actors: int = 0
    num_updates: int = 0
    minibatch_size: int = 0


############################
######  MAIN FUNCTION  #####
############################


def main():
    # set the device to the first available GPU
    jax.config.update("jax_platform_name", "gpu")

    # print the device that is being used
    print("Device: ", jax.devices())

    config = tyro.cli(Config)

    # Validate reward settings
    if config.sparse_rewards and config.individual_rewards:
        raise ValueError(
            "Cannot enable both sparse_rewards and individual_rewards simultaneously. "
            "Please choose only one reward setting."
        )

    if config.single_task_idx is not None:  # single-task baseline
        config.cl_method = "ft"
    if config.cl_method is None:
        raise ValueError(
            "cl_method is required. Please specify a continual learning method (e.g., ewc, mas, l2, ft, agem).")

    difficulty = config.difficulty
    seq_length = config.seq_length
    strategy = config.strategy
    seed = config.seed

    # Set height_min, height_max, width_min, width_max, and wall_density based on difficulty
    if difficulty:
        apply_difficulty_to_config(config, difficulty)

    # Set default regularization coefficient based on the CL method if not specified
    if config.reg_coef is None:
        if config.cl_method.lower() == "ewc":
            config.reg_coef = 1e11
        elif config.cl_method.lower() == "mas":
            config.reg_coef = 1e9
        elif config.cl_method.lower() == "l2":
            config.reg_coef = 1e7

    method_map = dict(ewc=EWC(mode=config.ewc_mode, decay=config.ewc_decay),
                      mas=MAS(),
                      l2=L2(),
                      ft=FT(),
                      agem=AGEM(memory_size=config.agem_memory_size, sample_size=config.agem_sample_size))

    cl = method_map[config.cl_method.lower()]

    # generate a sequence of tasks
    config.env_kwargs, layout_names = generate_sequence(
        sequence_length=seq_length,
        strategy=strategy,
        layout_names=config.layouts,
        seed=seed,
        height_rng=(config.height_min, config.height_max),
        width_rng=(config.width_min, config.width_max),
        wall_density=config.wall_density,
        layout_file=config.layout_file,
        complementary_restrictions=config.complementary_restrictions,
    )

    # Add view parameters for PO environments when difficulty is specified
    if config.env_name == "overcooked_po" and difficulty:
        for env_args in config.env_kwargs:
            env_args["view_ahead"] = config.view_ahead
            env_args["view_sides"] = config.view_sides
            env_args["view_behind"] = config.view_behind

    # Add random_reset parameter to all environments
    for env_args in config.env_kwargs:
        env_args["random_reset"] = config.random_reset

    # ── optional single-task baseline ─────────────────────────────────────────
    if config.single_task_idx is not None:
        idx = config.single_task_idx
        config.env_kwargs = [config.env_kwargs[idx]]
        layout_names = [layout_names[idx]]
        config.seq_length = 1

    # repeat the base sequence `repeat_sequence` times
    config.env_kwargs = config.env_kwargs * config.repeat_sequence
    layout_names = layout_names * config.repeat_sequence

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    network = "cnn" if config.use_cnn else "mlp"
    run_name = f'{config.alg_name}_{config.cl_method}_{difficulty}_{network}_seq{seq_length}_{strategy}_seed_{seed}_{timestamp}'
    exp_dir = os.path.join("runs", run_name)

    # Initialize WandB
    load_dotenv()
    wandb_tags = config.tags if config.tags is not None else []
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project=config.project,
        config=config,
        sync_tensorboard=True,
        mode=config.wandb_mode,
        tags=wandb_tags,
        group=config.cl_method.upper(),
        name=run_name,
        id=run_name,
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

    def get_view_params():
        '''
        Get view parameters for overcooked_po environments from config.
        Returns a dictionary with view parameters if applicable, empty dict otherwise.
        '''
        params = {"random_reset": config.random_reset}
        if config.env_name == "overcooked_po" and difficulty:
            params.update({
                "view_ahead": config.view_ahead,
                "view_sides": config.view_sides,
                "view_behind": config.view_behind
            })
        return params

    def create_environments():
        '''
        Creates environments, with padding for regular Overcooked but not for PO environments
        since PO environments have local observations that don't need padding.
        returns the environment layouts and agent restrictions
        '''
        agent_restrictions_list = []
        for env_args in config.env_kwargs:
            # Extract agent restrictions from env_args
            agent_restrictions_list.append(env_args.get('agent_restrictions', {}))

        # For PO environments, no padding is needed since observations are local
        # PO environments naturally have consistent observation spaces based on view parameters
        if config.env_name == "overcooked_po":
            # Return the original layouts without modification
            env_layouts = []
            for env_args in config.env_kwargs:
                temp_env = make(config.env_name, **env_args)
                env_layouts.append(temp_env.layout)
            return env_layouts, agent_restrictions_list

        # For regular environments, apply padding as before
        # Create environments first
        envs = []
        for env_args in config.env_kwargs:
            env = make(config.env_name, **env_args)
            envs.append(env)

        # find the environment with the largest observation space
        max_width, max_height = 0, 0
        for env in envs:
            max_width = max(max_width, env.layout["width"])
            max_height = max(max_height, env.layout["height"])

        # pad the observation space of all environments to be the same size by adding extra walls to the outside
        padded_envs = []
        for env in envs:
            # unfreeze the environment so that we can apply padding
            env = unfreeze(env.layout)

            # calculate the padding needed
            width_diff = max_width - env["width"]
            height_diff = max_height - env["height"]

            # determine the padding needed on each side
            left = width_diff // 2
            right = width_diff - left
            top = height_diff // 2
            bottom = height_diff - top

            width = env["width"]

            # Adjust the indices of the observation space to match the padded observation space
            def adjust_indices(indices):
                '''
                adjusts the indices of the observation space
                @param indices: the indices to adjust
                returns the adjusted indices
                '''
                indices = jnp.asarray(indices)
                rows, cols = jnp.divmod(indices, width)
                return (rows + top) * (width + left + right) + (cols + left)

            # adjust the indices of the observation space to account for the new walls
            env["wall_idx"] = adjust_indices(env["wall_idx"])
            env["agent_idx"] = adjust_indices(env["agent_idx"])
            env["goal_idx"] = adjust_indices(env["goal_idx"])
            env["plate_pile_idx"] = adjust_indices(env["plate_pile_idx"])
            env["onion_pile_idx"] = adjust_indices(env["onion_pile_idx"])
            env["pot_idx"] = adjust_indices(env["pot_idx"])

            # pad the observation space with walls
            padded_wall_idx = list(env["wall_idx"])  # Existing walls

            # Top and bottom padding
            for y in range(top):
                for x in range(max_width):
                    padded_wall_idx.append(y * max_width + x)  # Top row walls

            for y in range(max_height - bottom, max_height):
                for x in range(max_width):
                    padded_wall_idx.append(y * max_width + x)  # Bottom row walls

            # Left and right padding
            for y in range(top, max_height - bottom):
                for x in range(left):
                    padded_wall_idx.append(y * max_width + x)  # Left column walls

                for x in range(max_width - right, max_width):
                    padded_wall_idx.append(y * max_width + x)  # Right column walls

            env["wall_idx"] = jnp.array(padded_wall_idx)

            # set the height and width of the environment to the new padded height and width
            env["height"] = max_height
            env["width"] = max_width

            padded_envs.append(freeze(env))  # Freeze the environment to prevent further modifications

        return padded_envs, agent_restrictions_list

    env_layouts, agent_restrictions_list = create_environments()

    envs = []
    env_names = []
    max_soup_dict = {}
    for i, env_layout in enumerate(env_layouts):
        # Create the environment with agent restrictions
        agent_restrictions = agent_restrictions_list[i]
        view_params = get_view_params()
        env = make(config.env_name, layout=env_layout, layout_name=layout_names[i], task_id=i,
                   agent_restrictions=agent_restrictions, **view_params)
        env = LogWrapper(env, replace_info=False)
        env_name = env.layout_name
        envs.append(env)
        env_names.append(env_name)
        max_soup_dict[env_name] = estimate_max_soup(env_layout, env.max_steps, n_agents=env.num_agents)

    # === Build eval envs ONCE ===
    eval_envs = []
    for i, env_layout in enumerate(env_layouts):
        agent_restrictions = agent_restrictions_list[i]
        view_params = get_view_params()
        ev = make(
            config.env_name,
            layout=env_layout,
            layout_name=layout_names[i],
            task_id=i,
            agent_restrictions=agent_restrictions,
            **view_params
        )
        eval_envs.append(ev)

    # set extra config parameters based on the environment
    temp_env = envs[0]
    config.num_actors = temp_env.num_agents * config.num_envs
    config.num_updates = config.steps_per_task // config.num_steps // config.num_envs
    config.minibatch_size = (config.num_actors * config.num_steps) // config.num_minibatches

    def linear_schedule(count):
        '''
        Linearly decays the learning rate depending on the number of minibatches and number of epochs
        returns the learning rate
        '''
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac

    ac_cls = CNNActorCritic if config.use_cnn else MLPActorCritic

    network = ac_cls(temp_env.action_space().n, config.activation, seq_length, config.use_multihead,
                     config.shared_backbone, config.big_network, config.use_task_id, config.regularize_heads,
                     config.use_layer_norm)

    obs_dim = temp_env.observation_space().shape
    if not config.use_cnn:
        obs_dim = np.prod(obs_dim)

    # Initialize the network
    rng = jax.random.PRNGKey(seed)
    rng, network_rng = jax.random.split(rng)
    init_x = jnp.zeros((1, *obs_dim)) if config.use_cnn else jnp.zeros((1, obs_dim,))
    network_params = network.init(network_rng, init_x)

    # Initialize the optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
    )

    # jit the apply function
    network.apply = jax.jit(network.apply)

    # Initialize the training state
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx
    )

    # === JIT-compiled single-env evaluator ===
    @partial(jax.jit, static_argnums=(2, 3))
    def _eval_one_env(params, key, env, env_idx):
        """Return (avg_reward, total_soups) for one env over config.eval_num_episodes."""

        def run_episode(key_ep):
            key_ep, key_reset = jax.random.split(key_ep)
            obs, state = env.reset(key_reset)

            class EvalState(NamedTuple):
                key: Any
                state: Any
                obs: Any
                done: bool
                total_reward: float
                soup: float
                step_count: int

            def cond(s):
                return jnp.logical_and(jnp.logical_not(s.done), s.step_count < config.eval_num_steps)

            def body(s):
                key, state, obs, _, tot_r, tot_s, t = s
                key, key_a0, key_a1, key_step = jax.random.split(key, 4)

                # Batched obs -> network
                batched = {}
                for agent, v in obs.items():
                    v_b = jnp.expand_dims(v, 0)  # (1, H, W, C)
                    if not config.use_cnn:
                        v_b = v_b.reshape((1, -1))  # (1, F)
                    batched[agent] = v_b

                # Sample actions
                pi0, _ = network.apply(params, batched["agent_0"], env_idx=env_idx)
                a0 = jnp.squeeze(pi0.sample(seed=key_a0), 0)
                pi1, _ = network.apply(params, batched["agent_1"], env_idx=env_idx)
                a1 = jnp.squeeze(pi1.sample(seed=key_a1), 0)

                actions = {"agent_0": a0, "agent_1": a1}
                next_obs, next_state, reward, done, info = env.step(key_step, state, actions)

                done_flag = done["__all__"]
                step_rew = reward["agent_0"]  # shared reward setup
                soups = info["soups"]["agent_0"] + info["soups"]["agent_1"]

                return EvalState(key, next_state, next_obs,
                                 done_flag, tot_r + step_rew, tot_s + soups, t + 1)

            init = EvalState(key_ep, state, obs, False, 0.0, 0.0, 0)
            out = jax.lax.while_loop(cond, body, init)
            return out.total_reward, out.soup

        keys = jax.random.split(key, config.eval_num_episodes)
        rewards, soups = jax.vmap(run_episode)(keys)
        return rewards.mean(), soups.sum()

    def evaluate_model(train_state, key):
        """Host orchestrates over envs; compute is JIT on device."""
        rs = []
        ss = []
        keys = jax.random.split(key, len(eval_envs))
        for idx, (ev, k) in enumerate(zip(eval_envs, keys)):
            r, s = _eval_one_env(train_state.params, k, ev, idx)
            rs.append(r)
            ss.append(s)
        return rs, ss


    # === Split training into chunks so we can eval between them ===
    @partial(jax.jit, static_argnums=(2, 4))
    def _init_runner_state(rng, train_state, env, cl_state, env_idx):
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        runner_state = (train_state, env_state, obsv, 0, 0, rng, cl_state)
        return runner_state

    @partial(jax.jit, static_argnums=(2, 4, 5))
    def _train_chunk(rng, runner_state, env, cl_state, env_idx, num_updates_chunk):
        """
        Trains for `num_updates_chunk` update steps, *continuing* from runner_state.
        Returns (new_runner_state, last_step_metrics) where metrics are scalars.
        """

        # Bring local copies from runner_state for readability
        train_state, env_state, _, update_step, steps_for_env, rng, cl_state = runner_state

        reward_shaping_horizon = config.steps_per_task / 2
        rew_shaping_anneal = optax.linear_schedule(init_value=1., end_value=0., transition_steps=reward_shaping_horizon)

        def _update_step(runner_state, _):
            '''
            perform a single update step in the training loop
            @param runner_state: the carry state that contains all important training information
            returns the updated runner state and the metrics
            '''

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                '''
                selects an action based on the policy, calculates the log probability of the action,
                and performs the selected action in the environment
                @param runner_state: the current state of the runner
                returns the updated runner state and the transition
                '''
                # Unpack the runner state
                train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state

                # split the random number generator for action selection
                rng, _rng = jax.random.split(rng)

                # prepare the observations for the network
                obs_batch = batchify(last_obs, env.agents, config.num_actors,
                                     not config.use_cnn)  # (num_actors, obs_dim)
                # print("obs_shape", obs_batch.shape)

                # apply the policy network to the observations to get the suggested actions and their values
                pi, value = network.apply(train_state.params, obs_batch, env_idx=env_idx)

                # Sample and action from the policy
                action = pi.sample(seed=_rng)

                log_prob = pi.log_prob(action)

                # format the actions to be compatible with the environment
                env_act = unbatchify(action, env.agents, config.num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                # split the random number generator for stepping the environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.num_envs)

                # simultaniously step all environments with the selected actions (parallelized over the number of environments with vmap)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                current_timestep = update_step * config.num_steps * config.num_envs

                # Apply different reward settings based on configuration
                if config.sparse_rewards:
                    # Sparse rewards: only delivery rewards (no shaped rewards)
                    # reward already contains individual delivery rewards from environment
                    pass
                elif config.individual_rewards:
                    # Individual rewards: delivery rewards + individual shaped rewards
                    # Environment now provides individual delivery rewards directly
                    reward = jax.tree_util.tree_map(lambda x, y:
                                                    x + y * rew_shaping_anneal(current_timestep),
                                                    reward,
                                                    info["shaped_reward"]
                                                    )
                else:
                    # Default behavior: shared delivery rewards + individual shaped rewards
                    # Convert individual delivery rewards to shared rewards (both agents get total)
                    total_delivery_reward = reward["agent_0"] + reward["agent_1"]
                    shared_delivery_rewards = {"agent_0": total_delivery_reward, "agent_1": total_delivery_reward}

                    reward = jax.tree_util.tree_map(lambda x, y:
                                                    x + y * rew_shaping_anneal(current_timestep),
                                                    shared_delivery_rewards,
                                                    info["shaped_reward"]
                                                    )

                transition = Transition(
                    batchify(done, env.agents, config.num_actors, not config.use_cnn).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config.num_actors).squeeze(),
                    log_prob,
                    obs_batch
                )

                # Increment steps_for_env by the number of parallel envs
                steps_for_env = steps_for_env + config.num_envs

                runner_state = (train_state, env_state, obsv, update_step, steps_for_env, rng, cl_state)
                return runner_state, (transition, info)
            # Apply the _env_step function a series of times, while keeping track of the runner state
            runner_state, (traj_batch, info) = jax.lax.scan(
                f=_env_step,
                init=runner_state,
                xs=None,
                length=config.num_steps
            )

            # unpack the runner state that is returned after the scan function
            train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state

            # create a batch of the observations that is compatible with the network
            last_obs_batch = batchify(last_obs, env.agents, config.num_actors, not config.use_cnn)

            # apply the network to the batch of observations to get the value of the last state
            _, last_val = network.apply(train_state.params, last_obs_batch, env_idx=env_idx)

            def _calculate_gae(traj_batch, last_val):
                '''
                calculates the generalized advantage estimate (GAE) for the trajectory batch
                @param traj_batch: the trajectory batch
                @param last_val: the value of the last state
                returns the advantages and the targets
                '''

                def _get_advantages(gae_and_next_value, transition):
                    '''
                    calculates the advantage for a single transition
                    @param gae_and_next_value: the GAE and value of the next state
                    @param transition: the transition to calculate the advantage for
                    returns the updated GAE and the advantage
                    '''
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config.gamma * next_value * (1 - done) - value  # calculate the temporal difference
                    gae = (
                            delta
                            + config.gamma * config.gae_lambda * (1 - done) * gae
                    )  # calculate the GAE (used instead of the standard advantage estimate in PPO)

                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    f=_get_advantages,
                    init=(jnp.zeros_like(last_val), last_val),
                    xs=traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            # calculate the generalized advantage estimate (GAE) for the trajectory batch
            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                '''
                performs a single update epoch in the training loop
                @param update_state: the current state of the update
                returns the updated update_state and the total loss
                '''

                def _update_minbatch(carry, batch_info):
                    '''
                    performs a single update minibatch in the training loop
                    @param train_state: the current state of the training
                    @param batch_info: the information of the batch
                    returns the updated train_state and the total loss
                    '''
                    train_state, cl_state = carry
                    # unpack the batch information
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        '''
                        calculates the loss of the network
                        @param params: the parameters of the network
                        @param traj_batch: the trajectory batch
                        @param gae: the generalized advantage estimate
                        @param targets: the targets
                        @param network: the network
                        returns the total loss and the value loss, actor loss, and entropy
                        '''
                        # apply the network to the observations in the trajectory batch
                        pi, value = network.apply(params, traj_batch.obs, env_idx=env_idx)
                        log_prob = pi.log_prob(traj_batch.action)

                        # calculate critic loss
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config.clip_eps,
                                                                                                config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

                        # Calculate actor loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor_unclipped = ratio * gae
                        loss_actor_clipped = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config.clip_eps,
                                    1.0 + config.clip_eps,
                                    )
                                * gae
                        )

                        loss_actor = -jnp.minimum(loss_actor_unclipped,
                                                  loss_actor_clipped)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # CL penalty (for regularization-based methods)
                        cl_penalty = cl.penalty(params, cl_state, config.reg_coef)

                        total_loss = (loss_actor
                                      + config.vf_coef * value_loss
                                      - config.ent_coef * entropy
                                      + cl_penalty)
                        return total_loss, (value_loss, loss_actor, entropy, cl_penalty)

                    # returns a function with the same parameters as loss_fn that calculates the gradient of the loss function
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                    # call the grad_fn function to get the total loss and the gradients
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)

                    # For AGEM, we need to project the gradients
                    agem_stats = {}

                    def apply_agem_projection():
                        # Sample from memory
                        rng_1, sample_rng = jax.random.split(rng)
                        # Pick a random sample from AGEM memory
                        mem_obs, mem_actions, mem_log_probs, mem_advs, mem_targets, mem_values = sample_memory(
                            cl_state, config.agem_sample_size, sample_rng
                        )

                        # Compute memory gradient
                        grads_mem, grads_stats = compute_memory_gradient(
                            network, train_state.params,
                            config.clip_eps, config.vf_coef, config.ent_coef,
                            mem_obs, mem_actions, mem_advs, mem_log_probs,
                            mem_targets, mem_values,
                            env_idx=env_idx
                        )

                        # scale memory gradient by batch-size ratio
                        # ppo_bs = config.num_actors * config.num_steps
                        # mem_bs = config.agem_sample_size
                        g_ppo, _ = ravel_pytree(grads)  # grads  = fresh PPO grads
                        g_mem, _ = ravel_pytree(grads_mem)  # grads_mem = memory grads
                        norm_ppo = jnp.linalg.norm(g_ppo) + 1e-12
                        norm_mem = jnp.linalg.norm(g_mem) + 1e-12
                        scale = norm_ppo / norm_mem * config.agem_gradient_scale
                        grads_mem_scaled = jax.tree_util.tree_map(lambda g: g * scale, grads_mem)

                        # Project new grads
                        projected_grads, proj_stats = agem_project(grads, grads_mem_scaled)

                        # Combine stats for logging
                        combined_stats = {**grads_stats, **proj_stats}

                        scaled_norm = jnp.linalg.norm(ravel_pytree(grads_mem_scaled)[0])
                        combined_stats["agem/mem_grad_norm_scaled"] = scaled_norm

                        # Add memory buffer fullness percentage
                        total_used = jnp.sum(cl_state.sizes)
                        total_capacity = cl_state.max_tasks * cl_state.max_size_per_task
                        memory_fullness_pct = (total_used / total_capacity) * 100.0
                        combined_stats["agem/memory_fullness_pct"] = memory_fullness_pct

                        return projected_grads, combined_stats

                    def no_agem_projection():
                        # Return empty stats with the same structure as apply_agem_projection
                        empty_stats = {
                            "agem/agem_alpha": jnp.array(0.0),
                            "agem/agem_dot_g": jnp.array(0.0),
                            "agem/agem_final_grad_norm": jnp.array(0.0),
                            "agem/agem_is_proj": jnp.array(False),
                            "agem/agem_mem_grad_norm": jnp.array(0.0),
                            "agem/agem_ppo_grad_norm": jnp.array(0.0),
                            "agem/agem_projected_grad_norm": jnp.array(0.0),
                            "agem/mem_grad_norm_scaled": jnp.array(0.0),
                            "agem/memory_fullness_pct": jnp.array(0.0),
                            "agem/ppo_actor_loss": jnp.array(0.0),
                            "agem/ppo_entropy": jnp.array(0.0),
                            "agem/ppo_total_loss": jnp.array(0.0),
                            "agem/ppo_value_loss": jnp.array(0.0)
                        }
                        return grads, empty_stats

                    # Use JAX-compatible conditional logic
                    if config.cl_method.lower() == "agem" and cl_state is not None:
                        grads, agem_stats = jax.lax.cond(
                            jnp.sum(cl_state.sizes) > 0,
                            lambda: apply_agem_projection(),
                            lambda: no_agem_projection()
                        )

                    loss_information = total_loss, grads, agem_stats

                    # apply the gradients to the network
                    train_state = train_state.apply_gradients(grads=grads)

                    # Of course we also need to add the network to the carry here
                    return (train_state, cl_state), loss_information

                train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state

                # set the batch size and check if it is correct
                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                        batch_size == config.num_steps * config.num_actors
                ), "batch size must be equal to number of steps * number of actors"

                # create a batch of the trajectory, advantages, and targets
                batch = (traj_batch, advantages, targets)

                # reshape the batch to be compatible with the network
                batch = jax.tree_util.tree_map(
                    f=(lambda x: x.reshape((batch_size,) + x.shape[2:])), tree=batch
                )
                # split the random number generator for shuffling the batch
                rng, _rng = jax.random.split(rng)

                # creates random sequences of numbers from 0 to batch_size, one for each vmap
                permutation = jax.random.permutation(_rng, batch_size)

                # shuffle the batch
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )  # outputs a tuple of the batch, advantages, and targets shuffled

                minibatches = jax.tree_util.tree_map(
                    f=(lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:]))), tree=shuffled_batch,
                )

                (train_state, cl_state), loss_information = jax.lax.scan(
                    f=_update_minbatch,
                    init=(train_state, cl_state),
                    xs=minibatches
                )

                # Handle different return formats based on CL method
                total_loss, grads, agem_stats = loss_information
                # Create a dictionary to store all loss information
                loss_dict = {
                    "total_loss": total_loss
                }
                if config.cl_method.lower() == "agem":
                    loss_dict["agem_stats"] = agem_stats

                avg_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)

                update_state = (train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state)
                return update_state, loss_dict

            # create a tuple to be passed into the jax.lax.scan function
            update_state = (train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state)

            update_state, loss_info = jax.lax.scan(
                f=_update_epoch,
                init=update_state,
                xs=None,
                length=config.update_epochs
            )

            # unpack update_state
            train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state
            current_timestep = update_step * config.num_steps * config.num_envs
            metrics = jax.tree_util.tree_map(lambda x: x.mean(), info)

            if config.cl_method.lower() == "agem" and cl_state is not None:
                rng, mem_rng = jax.random.split(rng)
                perm = jax.random.permutation(mem_rng, advantages.shape[0])  # length = traj_len
                idx = perm[: config.agem_sample_size]

                obs_for_mem = traj_batch.obs[idx].reshape(-1, traj_batch.obs.shape[-1])
                acts_for_mem = traj_batch.action[idx].reshape(-1)
                logp_for_mem = traj_batch.log_prob[idx].reshape(-1)
                adv_for_mem = advantages[idx].reshape(-1)
                tgt_for_mem = targets[idx].reshape(-1)
                val_for_mem = traj_batch.value[idx].reshape(-1)

                cl_state = update_agem_memory(
                    cl_state, env_idx,
                    obs_for_mem, acts_for_mem, logp_for_mem,
                    adv_for_mem, tgt_for_mem, val_for_mem
                )

            # General section
            # Update the step counter
            update_step += 1

            metrics["General/env_index"] = env_idx
            metrics["General/update_step"] = update_step
            metrics["General/steps_for_env"] = steps_for_env
            metrics["General/env_step"] = update_step * config.num_steps * config.num_envs
            if config.anneal_lr:
                metrics["General/learning_rate"] = linear_schedule(
                    update_step * config.num_minibatches * config.update_epochs)
            else:
                metrics["General/learning_rate"] = config.lr

            # Losses section
            # Extract total_loss and components from loss_info
            loss_dict = loss_info
            total_loss = loss_dict["total_loss"]
            # Unpack the components of total_loss
            value_loss, loss_actor, entropy, reg_loss = total_loss[1]
            total_loss = total_loss[0]  # The actual scalar loss value

            metrics["Losses/total_loss"] = total_loss.mean()
            metrics["Losses/value_loss"] = value_loss.mean()
            metrics["Losses/actor_loss"] = loss_actor.mean()
            metrics["Losses/entropy"] = entropy.mean()
            metrics["Losses/reg_loss"] = reg_loss.mean()

            # Add AGEM stats to metrics if they exist
            if "agem_stats" in loss_dict:
                agem_stats = loss_dict["agem_stats"]
                for k, v in agem_stats.items():
                    if v.size > 0:  # Only add if there are values
                        metrics[k] = v.mean()

            # Soup section
            agent_0_soup = info["soups"]["agent_0"].sum()
            agent_1_soup = info["soups"]["agent_1"].sum()
            soup_delivered = agent_0_soup + agent_1_soup
            episode_frac = config.num_steps / env.max_steps
            metrics["Soup/agent_0_soup"] = agent_0_soup
            metrics["Soup/agent_1_soup"] = agent_1_soup
            metrics["Soup/total"] = soup_delivered
            metrics["Soup/scaled"] = soup_delivered / (max_soup_dict[env_names[env_idx]] * episode_frac)
            metrics.pop('soups', None)

            # Rewards section
            metrics["General/shaped_reward_agent0"] = metrics["shaped_reward"]["agent_0"]
            metrics["General/shaped_reward_agent1"] = metrics["shaped_reward"]["agent_1"]
            metrics.pop('shaped_reward', None)
            metrics["General/shaped_reward_annealed_agent0"] = metrics[
                                                                   "General/shaped_reward_agent0"] * rew_shaping_anneal(
                current_timestep)
            metrics["General/shaped_reward_annealed_agent1"] = metrics[
                                                                   "General/shaped_reward_agent1"] * rew_shaping_anneal(
                current_timestep)

            # Advantages and Targets section
            metrics["Advantage_Targets/advantages"] = advantages.mean()
            metrics["Advantage_Targets/targets"] = targets.mean()

            runner_state = (train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state)

            return runner_state, metrics

        runner_state, metrics = jax.lax.scan(
            f=_update_step,
            init=runner_state,
            xs=None,
            length=num_updates_chunk
        )

        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics)
        return runner_state, metrics

    def loop_over_envs(rng, train_state, cl_state, envs):
        '''
        Loops over the environments and trains the network
        @param rng: random number generator
        @param train_state: the current state of the training
        @param envs: the environments
        returns the runner state and the metrics
        '''
        # split the random number generator for training on the environments
        rng, *env_rngs = jax.random.split(rng, len(envs) + 1)

        # Create appropriate visualizer based on environment type
        if config.env_name == "overcooked_po":
            visualizer = OvercookedVisualizerPO(num_agents=temp_env.num_agents)
        else:
            visualizer = OvercookedVisualizer(num_agents=temp_env.num_agents)

        for i, (rng_i, env) in enumerate(zip(env_rngs, envs)):
            # --- Init runner state once per env ---
            runner_state = _init_runner_state(rng_i, train_state, env, cl_state, i)

            # Reset LR schedule / optimizer at the start of this task
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
            )
            train_state = train_state.replace(tx=tx, opt_state=tx.init(train_state.params))

            # put the updated train_state back in the runner_state tuple
            runner_state = (train_state, runner_state[1], runner_state[2],
                            runner_state[3], runner_state[4], runner_state[5], runner_state[6])

            updates_left = int(config.num_updates)
            while updates_left > 0:
                k = int(min(config.log_interval, updates_left))
                prev_update = int(runner_state[3])  # previous update_step
                runner_state, metrics_last = _train_chunk(rng_i, runner_state, env, cl_state, i, k)

                # Unpack new states
                train_state, env_state, obsv, update_step, steps_for_env, rng_i, cl_state = runner_state
                new_update = int(update_step)
                real_step = i * int(config.num_updates) + new_update

                # --- TRAINING LOGS (host-side) ---
                from jax import device_get
                tm = device_get(metrics_last)
                for key, value in tm.items():
                    writer.add_scalar(key, float(np.asarray(value)), int(real_step))

                # --- EVAL only when crossing a log boundary ---
                crossed = (prev_update // int(config.log_interval)) < (new_update // int(config.log_interval))
                if config.evaluation and crossed:
                    rng_i, eval_rng = jax.random.split(rng_i)
                    avg_rewards, avg_soups = evaluate_model(train_state, eval_rng)
                    episode_frac = config.eval_num_steps / env.max_steps
                    avg_soups = [s * episode_frac for s in avg_soups]
                    eval_metrics = add_eval_metrics(avg_rewards, avg_soups, env_names, max_soup_dict, {})
                    for key, value in eval_metrics.items():
                        writer.add_scalar(key, float(np.asarray(value)), int(real_step))

                updates_left -= k

            # === After finishing the env, do importance / CL update ===
            importance = cl.compute_importance(
                train_state.params, env, network, i, rng_i, config.use_cnn,
                config.importance_episodes, config.importance_steps, config.normalize_importance
            )
            cl_state = cl.update_state(cl_state, train_state.params, importance)

            # === GIFs ===
            if config.record_gif:
                env_name_disp = f"{i}__{env.layout_name}"
                states = record_gif_of_episode(config, train_state, env, network, env_idx=i, max_steps=config.gif_len)
                if config.env_name == "overcooked_po":
                    visualizer.animate(states, agent_view_size=5, task_idx=i, task_name=env_name_disp, exp_dir=exp_dir, env=env)
                else:
                    visualizer.animate(states, agent_view_size=5, task_idx=i, task_name=env_name_disp, exp_dir=exp_dir)

            # === Save ===
            repo_root = Path(__file__).resolve().parent.parent
            path = f"{repo_root}/checkpoints/overcooked/{config.cl_method}/{run_name}/model_env_{i + 1}"
            save_params(path, train_state, env_kwargs=env.layout, layout_name=env.layout_name, config=config)

            if config.single_task_idx is not None:
                break

    def save_params(path, train_state, env_kwargs=None, layout_name=None, config=None):
        '''
        Saves the parameters of the network along with environment configuration
        @param path: the path to save the parameters
        @param train_state: the current state of the training
        @param env_kwargs: the environment kwargs used to create the environment
        @param layout_name: the name of the layout
        @param config: the configuration used for training
        returns None
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model parameters
        with open(path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    {"params": train_state.params}
                )
            )

        # Save configuration and layout information
        if env_kwargs is not None or layout_name is not None or config is not None:
            # Define a recursive function to convert FrozenDict to regular dict
            def convert_frozen_dict(obj):
                if isinstance(obj, flax.core.frozen_dict.FrozenDict):
                    return {k: convert_frozen_dict(v) for k, v in unfreeze(obj).items()}
                elif isinstance(obj, dict):
                    return {k: convert_frozen_dict(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_frozen_dict(item) for item in obj]
                elif isinstance(obj, jax.Array):
                    # Convert JAX arrays to native Python types
                    array_obj = np.array(obj)
                    # Handle scalar values
                    if array_obj.size == 1:
                        return array_obj.item()
                    # Handle arrays
                    return array_obj.tolist()
                else:
                    return obj

            # Convert env_kwargs to regular dict
            env_kwargs = convert_frozen_dict(env_kwargs)

            config_data = {
                "env_kwargs": env_kwargs,
                "layout_name": layout_name
            }

            # Add relevant configuration parameters
            if config is not None:
                config_dict = {
                    "use_cnn": config.use_cnn,
                    "num_tasks": config.seq_length,
                    "use_multihead": config.use_multihead,
                    "shared_backbone": config.shared_backbone,
                    "big_network": config.big_network,
                    "use_task_id": config.use_task_id,
                    "regularize_heads": config.regularize_heads,
                    "use_layer_norm": config.use_layer_norm,
                    "activation": config.activation,
                    "strategy": config.strategy,
                    "seed": config.seed,
                    "height_min": config.height_min,
                    "height_max": config.height_max,
                    "width_min": config.width_min,
                    "width_max": config.width_max,
                    "wall_density": config.wall_density
                }
                # Convert any FrozenDict objects in the config
                config_dict = convert_frozen_dict(config_dict)
                config_data.update(config_dict)

            # Apply the conversion to the entire config_data to ensure all nested FrozenDict objects are converted
            config_data = convert_frozen_dict(config_data)

            config_path = f"{path}_config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

        print('model saved to', path)

    # Run the model
    rng, train_rng = jax.random.split(rng)
    cl_state = cl.init_state(train_state.params, config.regularize_critic, config.regularize_heads)

    # Initialize AGEM memory if using AGEM and this is the first environment
    if config.cl_method.lower() == "agem":
        # Get observation dimension
        obs_dim = envs[0].observation_space().shape
        if not config.use_cnn:
            obs_dim = (np.prod(obs_dim),)
        # Initialize memory buffer
        cl_state = init_agem_memory(config.agem_memory_size, obs_dim, max_tasks=config.seq_length)

    # apply the loop_over_envs function to the environments
    loop_over_envs(train_rng, train_state, cl_state, envs)


if __name__ == "__main__":
    print("Running main...")
    main()
