'''
Script for training a PPO ego agent against a *population* of homogeneous, RL-based partner agents. 
Does not support training against heuristic partner agents. 
**Warning**: modify with caution, as this script is used as the main script for ego training throughout the project.

If running the script directly, please specify a partner agent config at 
`ego_agent_training/configs/algorithm/ppo_ego/_base_.yaml`.

Command to run PPO ego training:
python ego_agent_training/run.py algorithm=ppo_ego/lbf task=lbf label=test_ppo_ego

Suggested debug command:
python ego_agent_training/run.py algorithm=ppo_ego/lbf task=lbf logger.mode=disabled label=debug algorithm.TOTAL_TIMESTEPS=1e5
'''
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState

# Import unified evaluation utilities
from experiments.utils import add_eval_metrics
from experiments.partner_adaptation.partner_agents.population_interface import AgentPopulation
from experiments.partner_adaptation.partner_generation.run_episodes import run_episodes
from experiments.partner_adaptation.partner_generation.utils import _create_minibatches_no_time, Transition, unbatchify
from experiments.partner_adaptation.partner_generation.utils import get_stats

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_ppo_ego_agent(
        config, env, train_rng,
        ego_policy, init_ego_params, n_ego_train_seeds,
        partner_population: AgentPopulation,
        partner_params, env_id_idx=0, eval_partner=[], cl=None, cl_state=None
):
    '''
    Train PPO ego agent using the given partner checkpoints and initial ego parameters.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        train_rng: jax.random.PRNGKey, random key for training
        ego_policy: AgentPolicy, policy for the ego agent
        init_ego_params: dict, initial parameters for the ego agent
        n_ego_train_seeds: int, number of ego training seeds
        partner_population: AgentPopulation, population of partner agents
        partner_params: pytree of parameters for the population of agents of shape (pop_size, ...).
    '''
    # Get partner parameters from the population
    num_total_partners = partner_population.pop_size

    # ------------------------------
    # Build the PPO training function
    # ------------------------------
    def make_ppo_train(config):
        '''agent 0 is the ego agent while agent 1 is the confederate'''
        num_agents = env.num_agents
        assert num_agents == 2, "This snippet assumes exactly 2 agents."

        def linear_schedule(count):
            frac = 1.0 - \
                   (count // (config.num_minibatches *
                              config.update_epochs)) / config.num_updates
            return config.lr * frac

        def train(rng):
            if config.anneal_lr:
                tx = optax.chain(
                    optax.clip_by_global_norm(config.max_grad_norm),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config.max_grad_norm),
                    optax.adam(config.lr, eps=1e-5),
                )

            train_state = TrainState.create(
                apply_fn=ego_policy.network.apply,
                params=init_ego_params,
                tx=tx,
            )
            #  Init ego and partner hstates
            init_ego_hstate = ego_policy.init_hstate(
                config.num_controlled_actors)

            init_partner_hstate = partner_population.init_hstate(
                config.num_uncontrolled_actors)

            # near the top of train(...) right after you compute/know config.num_updates
            eval_every = int(getattr(config, "eval_every", 1))  # evaluate every N updates
            num_ckpts = int(getattr(config, "num_checkpoints", 1))  # 1 = only final

            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                train_state, env_state, prev_obs, prev_done, ego_hstate, partner_hstate, partner_indices, rng = runner_state
                rng, actor_rng, partner_rng, step_rng = jax.random.split(
                    rng, 4)

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Conditionally resample partners based on prev_done["__all__"]
                needs_resample = prev_done["__all__"]  # shape (NUM_ENVS,) bool
                sampled_indices_all = partner_population.sample_agent_indices(
                    config.num_controlled_actors, partner_rng)

                # Determine final indices based on whether resampling was needed for each env
                updated_partner_indices = jnp.where(
                    needs_resample,  # Mask shape (NUM_ENVS,)
                    sampled_indices_all,  # Use newly sampled index if True
                    partner_indices  # Else, keep index from previous step
                )

                # Note that we do not need to reset the hiden states for both the ego and partner agents
                # as the recurrent states are automatically reset when done is True, and the partner indices are only reset when done is True.

                # Agent_0 (ego) action, value, log_prob
                act_0, val_0, pi_0, new_ego_hstate = ego_policy.get_action_value_policy(
                    params=train_state.params,
                    obs=prev_obs["agent_0"].reshape(
                        config.num_controlled_actors, -1),
                    done=prev_done["agent_0"].reshape(
                        config.num_controlled_actors),
                    avail_actions=avail_actions_0,
                    hstate=ego_hstate,
                    rng=actor_rng,
                    env_id_idx=env_id_idx,
                )
                logp_0 = pi_0.log_prob(act_0)

                act_0 = act_0.squeeze()
                logp_0 = logp_0.squeeze()
                val_0 = val_0.squeeze()

                # Agent_1 (partner) action using the AgentPopulation interface
                act_1, new_partner_hstate = partner_population.get_actions(
                    partner_params,
                    updated_partner_indices,
                    prev_obs["agent_1"].reshape(
                        config.num_controlled_actors, 1, -1),
                    prev_done["agent_1"].reshape(
                        config.num_controlled_actors, 1, -1),
                    avail_actions_1,
                    partner_hstate,
                    partner_rng,
                    env_state=env_state,
                    aux_obs=None
                )
                act_1 = act_1.squeeze()

                # Combine actions into the env format
                combined_actions = jnp.concatenate(
                    [act_0, act_1], axis=0)  # shape (2*num_envs,)
                env_act = unbatchify(
                    combined_actions, env.agents, config.num_envs, num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # Step env
                step_rngs = jax.random.split(step_rng, config.num_envs)
                obs_next, env_state_next, reward, done_next, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    step_rngs, env_state, env_act
                )
                # note that num_actors = num_envs * num_agents
                keys_to_drop = {"shaped_reward", "soups"}
                info = {k: v for k, v in info.items() if k not in keys_to_drop}
                info_0 = jax.tree.map(lambda x: x[:, 0], info)

                # Store agent_0 data in transition
                transition = Transition(
                    done=done_next["agent_0"],
                    action=act_0,
                    value=val_0,
                    reward=reward["agent_0"],
                    log_prob=logp_0,
                    obs=prev_obs["agent_0"].reshape(
                        config.num_controlled_actors, -1),
                    info=info_0,
                    avail_actions=avail_actions_0
                )
                new_runner_state = (train_state, env_state_next, obs_next, done_next,
                                    new_ego_hstate, new_partner_hstate, updated_partner_indices, rng)
                return new_runner_state, transition

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config.gamma * \
                            next_value * (1 - done) - value
                    gae = (
                            delta
                            + config.gamma *
                            config.gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )

                return advantages, advantages + traj_batch.value

            def _update_minbatch(train_state, batch_info):
                init_ego_hstate, traj_batch, advantages, returns = batch_info

                def _loss_fn(params, init_ego_hstate, traj_batch, gae, target_v):
                    _, value, pi, _ = ego_policy.get_action_value_policy(
                        params=params,
                        obs=traj_batch.obs,
                        done=traj_batch.done,
                        avail_actions=traj_batch.avail_actions,
                        hstate=init_ego_hstate,
                        # only used for action sampling, which is unused here
                        rng=jax.random.PRNGKey(0),
                        env_id_idx=env_id_idx,
                    )
                    log_prob = pi.log_prob(traj_batch.action)

                    # Value loss
                    value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                    ).clip(
                        -config.clip_eps, config.clip_eps)
                    value_losses = jnp.square(value - target_v)
                    value_losses_clipped = jnp.square(
                        value_pred_clipped - target_v)
                    value_loss = (
                        jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # Policy gradient loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                    pg_loss_1 = ratio * gae_norm
                    pg_loss_2 = jnp.clip(
                        ratio,
                        1.0 - config.clip_eps,
                        1.0 + config.clip_eps) * gae_norm
                    pg_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))

                    # Entropy
                    entropy = jnp.mean(pi.entropy())

                    # Continual learning penalty (for regularization-based methods)
                    cl_penalty = 0.0
                    if cl is not None and cl_state is not None:
                        cl_penalty = cl.penalty(params, cl_state, config.reg_coef)

                    total_loss = pg_loss + \
                                 config.vf_coef * value_loss - \
                                 config.ent_coef * entropy + \
                                 cl_penalty
                    return total_loss, (value_loss, pg_loss, entropy, cl_penalty)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss_val, aux_vals), grads = grad_fn(
                    train_state.params, init_ego_hstate, traj_batch, advantages, returns)
                train_state = train_state.apply_gradients(grads=grads)

                # compute average grad norm
                grad_l2_norms = jax.tree.map(
                    lambda g: jnp.linalg.norm(g.astype(jnp.float32)), grads)
                sum_of_grad_norms = jax.tree.reduce(
                    lambda x, y: x + y, grad_l2_norms)
                n_elements = len(jax.tree.leaves(grad_l2_norms))
                avg_grad_norm = sum_of_grad_norms / n_elements

                return train_state, (loss_val, aux_vals, avg_grad_norm)

            def _update_epoch(update_state, unused):
                train_state, init_ego_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)

                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                        batch_size == config.num_steps * config.num_controlled_actors
                ), "batch size must be equal to number of steps * number of actors"

                minibatches = _create_minibatches_no_time(
                    traj_batch, advantages, targets, init_ego_hstate, config.num_controlled_actors,
                    config.num_minibatches, batch_size, perm_rng)
                train_state, losses_and_grads = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_ego_hstate,
                                traj_batch, advantages, targets, rng)
                return update_state, losses_and_grads

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. PPO updates
                """
                (train_state, rng, update_steps) = update_runner_state
                # Init envs & partner indices
                rng, reset_rng, p_rng = jax.random.split(rng, 3)
                reset_rngs = jax.random.split(reset_rng, config.num_envs)
                init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config.num_envs), dtype=bool) for k in env.agents + ["__all__"]}
                new_partner_indices = partner_population.sample_agent_indices(config.num_uncontrolled_actors, p_rng)

                # 1) rollout
                runner_state = (train_state, init_env_state, init_obs, init_done,
                                init_ego_hstate, init_partner_hstate, new_partner_indices, rng)

                runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.num_steps)
                (train_state, env_state, obs, done, ego_hstate, partner_hstate, partner_indices, rng) = runner_state

                # 2) advantage
                # Get available actions for agent 0 from environment state
                avail_actions_0 = jax.vmap(env.get_avail_actions)(env_state.env_state)["agent_0"].astype(jnp.float32)

                # Get final value estimate for completed trajectory
                _, last_val, _, _ = ego_policy.get_action_value_policy(
                    params=train_state.params,
                    obs=obs["agent_0"].reshape(
                        config.num_controlled_actors, -1),
                    done=done["agent_0"].reshape(
                        config.num_controlled_actors),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=ego_hstate,
                    # Dummy key since we're just extracting the value
                    rng=jax.random.PRNGKey(0),
                    env_id_idx=env_id_idx,
                )
                last_val = last_val.squeeze()
                advantages, targets = _calculate_gae(traj_batch, last_val)

                # 3) PPO update
                update_state = (
                    train_state,
                    # shape is (num_controlled_actors, gru_hidden_dim) with all-0s value
                    init_ego_hstate,
                    # obs has shape (rollout_len, num_controlled_actors, -1)
                    traj_batch,
                    advantages,
                    targets,
                    rng
                )
                update_state, losses_and_grads = jax.lax.scan(
                    _update_epoch, update_state, None, config.update_epochs)
                train_state = update_state[0]
                _, loss_terms, avg_grad_norm = losses_and_grads

                metric = traj_batch.info
                metric["update_steps"] = update_steps
                metric["actor_loss"] = loss_terms[1]
                metric["value_loss"] = loss_terms[0]
                metric["entropy_loss"] = loss_terms[2]
                metric["cl_penalty"] = loss_terms[3]
                metric["avg_grad_norm"] = avg_grad_norm
                new_runner_state = (train_state, rng, update_steps + 1)
                return (new_runner_state, metric)

            # PPO Update and Checkpoint saving
            # -1 because we store a ckpt at the last update
            # ckpt_and_eval_interval = config.num_updates // max(1, config.num_checkpoints - 1)
            # num_ckpts = config.num_checkpoints

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                    params_pytree)

            max_episode_steps = config.num_steps

            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_state, checkpoint_array, ckpt_idx,
                 init_eval_last_info, init_eval_infos) = state_with_ckpt

                # Single PPO update
                new_update_state, metric = _update_step(
                    update_state,
                    None
                )
                (train_state, rng, update_steps) = new_update_state

                # To eval or not to eval
                to_eval = jnp.logical_or(
                    jnp.equal(jnp.mod(update_steps - 1, eval_every), 0),
                    jnp.equal(update_steps, config.num_updates),
                )

                # Only store a checkpoint at the very end when num_ckpts == 1
                to_store_ckpt = jnp.equal(update_steps, config.num_updates)

                def do_eval(args):
                    rng, prev_eval_last_info, prev_eval_infos = args

                    eval_partner_indices = jnp.arange(num_total_partners)
                    gathered_params = partner_population.gather_agent_params(partner_params, eval_partner_indices)

                    rng, eval_rng = jax.random.split(rng)
                    eval_eps_last_infos = jax.vmap(lambda x: run_episodes(
                        eval_rng, env,
                        agent_0_param=train_state.params, agent_0_policy=ego_policy,
                        agent_1_param=x, agent_1_policy=partner_population.policy_cls,
                        max_episode_steps=max_episode_steps,
                        env_id_idx=env_id_idx,
                        num_eps=config.num_eval_episodes
                    ))(gathered_params)

                    eval_infos = []
                    for eval_policy, params, idx in eval_partner:
                        eval_info = jax.vmap(lambda x: run_episodes(
                            eval_rng,
                            env,
                            agent_0_param=train_state.params, agent_0_policy=ego_policy,
                            agent_1_param=x, agent_1_policy=eval_policy.policy_cls,
                            max_episode_steps=max_episode_steps,
                            env_id_idx=idx,
                            num_eps=config.num_eval_episodes
                        ))(params)
                        eval_infos.append((idx, eval_info))

                    return (rng, eval_eps_last_infos, eval_infos)

                def skip_eval(args):
                    return args  # keep previous eval results

                def store_ckpt(args):
                    ckpt_arr, cidx = args
                    new_ckpt_arr = jax.tree.map(lambda c_arr, p: c_arr.at[cidx].set(p),
                                                checkpoint_array, train_state.params)
                    return (new_ckpt_arr, cidx + 1)

                def skip_ckpt(args):
                    return args

                (checkpoint_array, ckpt_idx) = jax.lax.cond(
                    jnp.logical_and(to_store_ckpt, num_ckpts == 1),
                    store_ckpt, skip_ckpt, (checkpoint_array, ckpt_idx)
                )

                (rng, eval_last_infos, eval_infos) = jax.lax.cond(
                    to_eval, do_eval, skip_eval, (rng, init_eval_last_info, init_eval_infos)
                )

                metric["eval_ep_last_info"] = eval_last_infos
                metric["eval_infos"] = eval_infos
                return ((train_state, rng, update_steps), checkpoint_array, ckpt_idx, eval_last_infos,
                        eval_infos), metric

            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            rng, rng_eval, rng_train = jax.random.split(rng, 3)
            # Init eval return infos
            eval_partner_indices = jnp.arange(num_total_partners)
            gathered_params = partner_population.gather_agent_params(partner_params, eval_partner_indices)
            eval_eps_last_infos = jax.vmap(lambda x: run_episodes(
                rng_eval, env,
                agent_0_param=train_state.params, agent_0_policy=ego_policy,
                agent_1_param=x, agent_1_policy=partner_population.policy_cls,
                max_episode_steps=max_episode_steps,
                env_id_idx=env_id_idx,
                num_eps=config.num_eval_episodes))(gathered_params)

            eval_infos = []

            for eval_policy, params, idx in eval_partner:
                eval_info = jax.vmap(lambda x: run_episodes(
                    rng_eval,
                    env,
                    agent_0_param=train_state.params, agent_0_policy=ego_policy,
                    agent_1_param=x, agent_1_policy=eval_policy.policy_cls,
                    max_episode_steps=max_episode_steps,
                    env_id_idx=idx,
                    num_eps=config.num_eval_episodes))(params)
                eval_infos.append((idx, eval_info))

            # initial runner state for scanning
            update_steps = 0
            rng_train, partner_rng = jax.random.split(rng_train)

            update_runner_state = (train_state, rng_train, update_steps)
            state_with_ckpt = (update_runner_state, checkpoint_array, ckpt_idx, eval_eps_last_infos, eval_infos)

            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config.num_updates
            )
            (final_runner_state, checkpoint_array, final_ckpt_idx, eval_eps_last_infos, eval_infos) = state_with_ckpt
            out = {
                "final_params": final_runner_state[0].params,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints": checkpoint_array,
            }
            return out

        return train

    # ------------------------------
    # Actually run the PPO training
    # ------------------------------
    rngs = jax.random.split(train_rng, n_ego_train_seeds)
    if n_ego_train_seeds == 1:
        train_fn = jax.jit(make_ppo_train(config))
        out = train_fn(rngs[0])
    else:
        train_fn = jax.jit(jax.vmap(make_ppo_train(config)))
        out = train_fn(rngs)
    return out


def add_seed_axis_if_missing(x, expected_ndim_with_seed):
    x = np.asarray(x)
    if x.ndim == expected_ndim_with_seed - 1:
        return x[None, ...]  # add leading seed axis
    return x


def mean_over_all_but_updates(arr, num_updates: int):
    # force integer
    num_updates = int(float(num_updates))

    a = np.asarray(arr)
    if a.size == 0:
        return np.zeros((num_updates,), dtype=float)

    # locate an axis equal to num_updates; else heuristic
    cand = [i for i, s in enumerate(a.shape) if s == num_updates]
    upd_ax = cand[0] if cand else (1 if a.ndim >= 3 else 0)

    # Ensure the chosen axis actually matches num_updates; if not, fall back
    if a.shape[upd_ax] != num_updates:
        # try the first axis if it matches
        if a.ndim > 0 and a.shape[0] == num_updates:
            upd_ax = 0
        else:
            # last resort: set num_updates to the length of the chosen axis
            num_updates = int(a.shape[upd_ax])

    a = np.moveaxis(a, upd_ax, 0)  # (num_updates, ...)
    a = a.reshape(num_updates, -1)  # (num_updates, rest)
    return a.mean(axis=1)


def _stat_mean_at_step(stat_data, step: int) -> float:
    """
    Accepts shapes:
      - (U,)           -> mean only
      - (U, 2)         -> (mean, std)
      - higher-dim     -> if last dim >=2, use [:, 0] as mean; else squeeze
    Returns a scalar float for the requested step (clamped to available length).
    """
    arr = np.asarray(stat_data)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        i = min(step, arr.shape[0] - 1)
        return float(arr[i])
    # ndim >= 2
    if arr.shape[-1] >= 2:
        mean_series = arr[(slice(None),) * (arr.ndim - 1) + (0,)]
        i = min(step, mean_series.shape[0] - 1)
        return float(mean_series[i])
    arr_sq = np.squeeze(arr)
    i = min(step, arr_sq.shape[0] - 1)
    return float(arr_sq[i])


def _extract_partner_id(idx):
    """
    Robustly turn idx into an int.
    Handles: int, numpy scalar/array, jax scalar/array, lists/tuples (even nested).
    Falls back to 0 if empty.
    """
    # If it's a list/tuple, recurse on the first element that exists
    if isinstance(idx, (list, tuple)):
        if not idx:
            return 0
        return _extract_partner_id(idx[0])

    # If it looks like an array (NumPy or JAX), make a NumPy view
    # and grab the first element if needed
    if hasattr(idx, "shape"):
        arr = np.asarray(idx)
        if arr.shape == ():  # scalar
            return int(arr.item())
        if arr.size > 0:  # vector or higher-dim
            return int(arr.reshape(-1)[0])
        return 0

    return int(idx)


def log_metrics(config, train_out, metric_names: tuple, max_soup_dict=None, layout_names=None):
    """Process training metrics and log them using the provided logger.

    Args:
        config: Configuration object
        train_out: dict, the logs from training
        metric_names: tuple, names of metrics to extract from training logs
        max_soup_dict: dict, maximum soup counts for each layout (for unified soup metrics)
        layout_names: list, names of layouts/partners for evaluation metrics
    """
    train_metrics = train_out["metrics"]

    #### Extract train metrics ####
    train_stats = get_stats(train_metrics, metric_names)
    # each key in train_stats is a metric name, and the value is an array of shape (num_seeds, num_updates, 2)
    # where the last dimension contains the mean and std of the metric
    train_stats = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}

    # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_value_losses = np.asarray(train_metrics["value_loss"])
    # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_actor_losses = np.asarray(train_metrics["actor_loss"])
    # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_entropy_losses = np.asarray(train_metrics["entropy_loss"])
    # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_grad_norms = np.asarray(train_metrics["avg_grad_norm"])
    # Process eval return metrics - average across ego seeds, eval episodes,  training partners
    # and num_agents per game for each checkpoint
    # shape (n_ego_train_seeds, num_updates, num_partners, num_eval_episodes, nuM_agents_per_game)
    all_ego_returns = np.asarray(train_metrics["eval_ep_last_info"]["returned_episode_returns"])
    all_ego_returns = all_ego_returns.sum(axis=-1)
    all_ego_returns = add_seed_axis_if_missing(all_ego_returns, expected_ndim_with_seed=4)
    average_ego_rets_per_iter = np.mean(all_ego_returns, axis=(0, 2, 3))

    # Extract soup metrics
    all_ego_soups = None
    average_ego_soups_per_iter = None
    if "returned_episode_soups" in train_metrics["eval_ep_last_info"]:
        all_ego_soups = np.asarray(train_metrics["eval_ep_last_info"]["returned_episode_soups"])
        all_ego_soups = all_ego_soups.sum(axis=-1)
        average_ego_soups_per_iter = np.mean(all_ego_soups, axis=(0, 2, 3))

    per_partner_per_iter = {}
    per_partner_soup_per_iter = {}
    for (idx, metrics) in train_metrics["eval_infos"]:
        return_per_partner = np.asarray(metrics["returned_episode_returns"])
        return_per_partner = return_per_partner.sum(axis=-1)
        return_per_partner = add_seed_axis_if_missing(return_per_partner, expected_ndim_with_seed=4)
        average_return_per_partner_per_iters = np.mean(return_per_partner, axis=(0, 2, 3))
        partner_id = _extract_partner_id(idx)

        per_partner_per_iter[f"Eval/EgoReturn_Partner{partner_id}"] = average_return_per_partner_per_iters

        # Add soup metrics for per-partner evaluation
        if "returned_episode_soups" in metrics:
            soup_per_partner = np.asarray(metrics["returned_episode_soups"])
            soup_per_partner = soup_per_partner.sum(axis=-1)
            soup_per_partner = add_seed_axis_if_missing(soup_per_partner, expected_ndim_with_seed=4)
            average_soup_per_partner_per_iters = np.mean(soup_per_partner, axis=(0, 2, 3))
            per_partner_soup_per_iter[f"Eval/EgoSoup_Partner{partner_id}"] = average_soup_per_partner_per_iters

    # Process loss metrics - average across ego seeds, partners and minibatches dims
    # Loss metrics shape should be (n_ego_train_seeds, num_updates, ...)
    average_ego_value_losses = mean_over_all_but_updates(all_ego_value_losses, config.num_updates)
    average_ego_actor_losses = mean_over_all_but_updates(all_ego_actor_losses, config.num_updates)
    average_ego_entropy_losses = mean_over_all_but_updates(all_ego_entropy_losses, config.num_updates)
    average_ego_grad_norms = mean_over_all_but_updates(all_ego_grad_norms, config.num_updates)

    # ---- figure out how many steps to log safely ----
    num_updates = int(len(average_ego_value_losses))

    # also cap by the shortest train_stat series (handles 1-D or 2-D stats)
    if train_stats:
        shortest = min(max(1, np.asarray(v).shape[0]) for v in train_stats.values())
        num_updates = min(num_updates, int(shortest))

    for step in range(num_updates):
        metrics = {}

        # robust stat extraction (works for 1-D or (mean,std))
        for stat_name, stat_data in train_stats.items():
            stat_mean = _stat_mean_at_step(stat_data, step)
            metrics[f"Train/Ego_{stat_name}"] = stat_mean

        metrics["Eval/EgoReturn"] = float(average_ego_rets_per_iter[step])

        if average_ego_soups_per_iter is not None:
            metrics["Eval/EgoSoup"] = float(average_ego_soups_per_iter[step])

        for partner_name, partner_data in per_partner_per_iter.items():
            metrics[partner_name] = float(partner_data[step])

        for partner_name, partner_data in per_partner_soup_per_iter.items():
            metrics[partner_name] = float(partner_data[step])

        if max_soup_dict is not None and layout_names is not None and average_ego_soups_per_iter is not None:
            avg_rewards = [float(average_ego_rets_per_iter[step])]
            avg_soups = [float(average_ego_soups_per_iter[step])]
            metrics = add_eval_metrics(avg_rewards, avg_soups, layout_names, max_soup_dict, metrics)

        metrics["Train/EgoValueLoss"] = float(average_ego_value_losses[step])
        metrics["Train/EgoActorLoss"] = float(average_ego_actor_losses[step])
        metrics["Train/EgoEntropyLoss"] = float(average_ego_entropy_losses[step])
        metrics["Train/EgoGradNorm"] = float(average_ego_grad_norms[step])
        metrics["train_step"] = int(step)

        wandb.log(metrics, commit=True)
