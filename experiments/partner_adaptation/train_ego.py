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
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState

# Import unified evaluation utilities
from experiments.utils import add_eval_metrics
from experiments.continual.agem import AGEMMemory, update_agem_memory
from experiments.continual.er_ace import ERACE, compute_er_ace_gradient
from experiments.partner_adaptation.partner_agents.population_interface import AgentPopulation
from experiments.partner_adaptation.partner_generation.run_episodes import run_episodes
from experiments.partner_adaptation.partner_generation.utils import _create_minibatches_no_time, Transition, unbatchify
from experiments.partner_adaptation.partner_generation.utils import get_stats

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class StageKeys(NamedTuple):
    """Independent PRNG streams for one training stage (one partner)."""
    train: jax.Array
    eval: jax.Array
    importance: jax.Array


def make_stage_keys(seed: int, stage_idx: int) -> StageKeys:
    """Derive reproducible, mutually independent streams for a stage.

    Stage keys depend only on (seed, stage_idx), so stages never replay each other's
    random draws, and evaluation keys are never split from the training stream.
    """
    stage_key = jax.random.fold_in(jax.random.PRNGKey(seed), stage_idx)
    return StageKeys(*(jax.random.fold_in(stage_key, i) for i in range(3)))


def team_episode_summary(returned_episode_soups, returned_episode_returns, returned_episode):
    """Collapse per-agent LogWrapper episode outputs (..., num_agents) to team quantities.

    The Overcooked env reports reward and soup deliveries only for the delivering agent,
    so summing over agents counts each delivery exactly once. Returns (team_soups,
    team_returns, completed); `completed` is False for episodes that have not finished,
    whose returned_* values are not meaningful.
    """
    completed = jnp.all(returned_episode, axis=-1)
    return returned_episode_soups.sum(axis=-1), returned_episode_returns.sum(axis=-1), completed


def should_evaluate(update_steps, eval_every: int, num_updates: int):
    """True after the 1st update, every `eval_every` updates after that, and after the last."""
    return jnp.logical_or(
        jnp.equal(jnp.mod(update_steps - 1, eval_every), 0),
        jnp.equal(update_steps, num_updates),
    )


def group_eval_partners(eval_partner, current=None):
    """Group evaluation partners that share a policy object so each group is one vmapped graph.

    Args:
        eval_partner: list of (population, params with leading axis 1, partner_id).
        current: optional (population, params, partner_id) for the partner currently being
            trained; added only if no entry with the same partner_id exists.
    Returns:
        policies: tuple of policy objects, one per group.
        params: list of parameter pytrees stacked along axis 0, one per group.
        ids: list of int32 arrays of partner ids, one per group.
    Partners with duplicate ids are evaluated once (first occurrence wins).
    """
    entries = list(eval_partner)
    if current is not None and all(idx != current[2] for _, _, idx in entries):
        entries.append(current)

    seen, order, groups = set(), [], {}
    for population, params, idx in entries:
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        key = id(population.policy_cls)
        if key not in groups:
            groups[key] = (population.policy_cls, [], [])
            order.append(key)
        groups[key][1].append(params)
        groups[key][2].append(idx)

    policies, stacked, ids = [], [], []
    for key in order:
        policy, params_list, id_list = groups[key]
        policies.append(policy)
        stacked.append(jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *params_list))
        ids.append(jnp.asarray(id_list, dtype=jnp.int32))
    return tuple(policies), stacked, ids


def build_eval_event(stage_id, update_steps, train_episodes, partner_ids, team_soups, team_returns, completed,
                     num_updates, steps_per_update, max_soup=None):
    """Turn one evaluation event into (record, log_dict).

    team_soups/team_returns/completed have shape (num_partners, num_eval_episodes). Means are taken
    over completed episodes only; a partner with no completed episode is omitted instead of logged as 0.
    The record carries the identities needed to place the event: stage, update, cumulative environment
    steps (training interactions across all stages so far), training episodes completed in this stage,
    partner ids, and the index of each evaluation episode.
    """
    stage_id, update_steps, train_episodes = int(stage_id), int(update_steps), int(train_episodes)
    partner_ids = [int(p) for p in np.asarray(partner_ids)]
    team_soups, team_returns = np.asarray(team_soups), np.asarray(team_returns)
    completed = np.asarray(completed, dtype=bool)
    env_steps = (stage_id * int(num_updates) + update_steps) * int(steps_per_update)

    record = {
        "stage_id": stage_id,
        "update": update_steps,
        "env_steps": env_steps,
        "train_episodes": train_episodes,
        "partner_ids": partner_ids,
        "episode_ids": list(range(team_soups.shape[1])),
        "team_soups": team_soups,
        "team_returns": team_returns,
        "completed": completed,
    }

    log_dict = {
        "train_step": stage_id * int(num_updates) + update_steps - 1,
        "Eval/StageID": stage_id,
        "Eval/EnvSteps": env_steps,
        "Eval/TrainEpisodes": train_episodes,
    }
    for row, pid in enumerate(partner_ids):
        n = int(completed[row].sum())
        log_dict[f"Eval/Episodes_Partner{pid}"] = n
        if n == 0:
            continue
        soup = float(team_soups[row][completed[row]].mean())
        ret = float(team_returns[row][completed[row]].mean())
        log_dict[f"Eval/EgoSoup_Partner{pid}"] = soup
        log_dict[f"Eval/EgoReturn_Partner{pid}"] = ret
        if max_soup:
            log_dict[f"Eval/EgoSoup_scaled_Partner{pid}"] = soup / max_soup
        if pid == stage_id:
            log_dict["Eval/EgoSoup"] = soup
            log_dict["Eval/EgoReturn"] = ret
            if max_soup:
                log_dict["Eval/EgoSoup_scaled"] = soup / max_soup
    return record, log_dict


def build_train_log(stage_id, update_steps, num_updates, n_episodes, team_soup, ego_return, value_loss, actor_loss,
                    entropy_loss, grad_norm, max_soup=None):
    log_dict = {
        "Train/EgoValueLoss": float(value_loss),
        "Train/EgoActorLoss": float(actor_loss),
        "Train/EgoEntropyLoss": float(entropy_loss),
        "Train/EgoGradNorm": float(grad_norm),
        "Train/EpisodesCompleted": int(n_episodes),
        "train_step": int(stage_id) * int(num_updates) + int(update_steps) - 1,
    }
    if int(n_episodes) > 0:
        log_dict["Train/EgoReturn"] = float(ego_return)
        log_dict["Train/EgoSoup"] = float(team_soup)
        if max_soup:
            log_dict["Train/EgoSoup_scaled"] = float(team_soup) / max_soup
    return log_dict


def train_ppo_ego_agent(
        config, env, train_rng,
        ego_policy, init_ego_params, n_ego_train_seeds,
        partner_population: AgentPopulation,
        partner_params, env_id_idx=0, eval_partner=[], cl=None, cl_state=None,
        eval_rng=None, log_fn=None, max_soup=None, compiled_cache=None
):
    '''
    Train PPO ego agent using the given partner checkpoints and initial ego parameters.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        train_rng: jax.random.PRNGKey, random key for training. Never consumed by evaluation.
        ego_policy: AgentPolicy, policy for the ego agent
        init_ego_params: dict, initial parameters for the ego agent
        n_ego_train_seeds: int, number of ego training seeds
        partner_population: AgentPopulation, population of partner agents
        partner_params: pytree of parameters for the population of agents of shape (pop_size, ...).
        env_id_idx: stage / task id of this partner (also selects the ego head)
        eval_partner: list of (population, params, partner_id) evaluated on evaluation events. The
            current partner is evaluated through this list when present, otherwise it is added.
        eval_rng: key for the evaluation stream. Each event uses fold_in(eval_rng, update), so the
            draws depend only on the update index, not on how many events came before.
        log_fn: callable taking a dict; defaults to wandb.log. Called only on actual events.
        max_soup: layout max soup count used for the *_scaled metrics (computed if omitted).
        compiled_cache: optional dict reused across stages of one run. Stages whose static pieces
            (config, env, ego policy, partner population type/policy, CL method, eval groups) match
            share one compiled train function; params, CL state and stage id are runtime arguments.
    '''
    if eval_rng is None:
        eval_rng = jax.random.fold_in(train_rng, 1)

    eval_policies, eval_params, eval_ids = group_eval_partners(
        eval_partner, current=(partner_population, partner_params, int(env_id_idx)))

    # ------------------------------
    # Build the PPO training function
    # ------------------------------
    def make_ppo_train(config):
        '''agent 0 is the ego agent while agent 1 is the confederate'''
        from meal.env.overcooked.max_soup_calculator import calculate_max_soup
        num_agents = env.num_agents
        assert num_agents == 2, "This snippet assumes exactly 2 agents."

        # Max soup for this layout (used for normalization in logging)
        if max_soup is not None:
            max_soup_val = float(max_soup)
        else:
            try:
                max_soup_val = float(calculate_max_soup(config.layout["layout"], config.num_steps, n_agents=num_agents))
            except Exception:
                max_soup_val = None

        is_memory_method = (cl is not None and cl_state is not None and
                            isinstance(cl_state, AGEMMemory))
        num_updates = int(config.num_updates)
        steps_per_update = int(config.num_envs) * int(config.num_steps)
        emit = log_fn if log_fn is not None else (lambda d: wandb.log(d))

        def linear_schedule(count):
            frac = 1.0 - \
                   (count // (config.num_minibatches *
                              config.update_epochs)) / config.num_updates
            return config.lr * frac

        def train(rng, eval_key, init_ego_params, partner_params, cl_state, env_id_idx, eval_params, eval_ids):
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

            eval_every = int(getattr(config, "eval_every", 1))  # evaluate every N updates
            num_ckpts = int(getattr(config, "num_checkpoints", 1))  # 1 = only final

            rew_shaping_horizon = float(getattr(config, "reward_shaping_horizon", 0.0))
            if rew_shaping_horizon > 0:
                rew_shaping_anneal = optax.linear_schedule(
                    init_value=1.0,
                    end_value=0.0,
                    transition_steps=int(rew_shaping_horizon),
                )
            else:
                rew_shaping_anneal = None

            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                train_state, env_state, prev_obs, prev_done, ego_hstate, partner_hstate, partner_indices, rng, update_steps_inner = runner_state
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
                # Apply reward shaping annealing if configured
                shaped_reward_0 = info["shaped_reward"]["agent_0"]
                if rew_shaping_anneal is not None:
                    current_timestep = update_steps_inner * config.num_envs * config.num_steps
                    ego_reward = reward["agent_0"] + rew_shaping_anneal(current_timestep) * shaped_reward_0
                else:
                    ego_reward = reward["agent_0"]

                # Team episode soups (raw deliveries by either agent, summed once) as reported by LogWrapper.
                # Only meaningful where the episode has completed.
                team_ep_soups, _, _ = team_episode_summary(
                    info["returned_episode_soups"], info["returned_episode_returns"], info["returned_episode"])

                # note that num_actors = num_envs * num_agents
                keys_to_drop = {"shaped_reward", "soups"}
                info = {k: v for k, v in info.items() if k not in keys_to_drop}
                info_0 = jax.tree.map(lambda x: x[:, 0], info)

                # Store agent_0 data in transition
                transition = Transition(
                    done=done_next["agent_0"],
                    action=act_0,
                    value=val_0,
                    reward=ego_reward,
                    log_prob=logp_0,
                    obs=prev_obs["agent_0"].reshape(
                        config.num_controlled_actors, -1),
                    info=info_0,
                    avail_actions=avail_actions_0
                )
                new_runner_state = (train_state, env_state_next, obs_next, done_next,
                                    new_ego_hstate, new_partner_hstate, updated_partner_indices, rng,
                                    update_steps_inner)
                return new_runner_state, (transition, team_ep_soups)

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

            def _update_minbatch(carry, batch_info):
                if is_memory_method:
                    train_state, mb_rng = carry
                else:
                    train_state = carry
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

                # ER-ACE: add BC gradient from past task memory (uses initial cl_state via closure)
                if is_memory_method and isinstance(cl, ERACE):
                    mb_rng, er_ace_rng = jax.random.split(mb_rng)
                    past_sizes = cl_state.sizes.at[env_id_idx].set(0)
                    er_ace_grads, _ = compute_er_ace_gradient(
                        ego_policy.network, train_state.params, cl_state,
                        config.agem_sample_size, er_ace_rng, past_sizes,
                    )
                    grads = jax.tree_util.tree_map(
                        lambda g, eg: g + config.er_ace_coef * eg, grads, er_ace_grads
                    )

                train_state = train_state.apply_gradients(grads=grads)

                # compute average grad norm
                grad_l2_norms = jax.tree.map(
                    lambda g: jnp.linalg.norm(g.astype(jnp.float32)), grads)
                sum_of_grad_norms = jax.tree.reduce(
                    lambda x, y: x + y, grad_l2_norms)
                n_elements = len(jax.tree.leaves(grad_l2_norms))
                avg_grad_norm = sum_of_grad_norms / n_elements

                if is_memory_method:
                    return (train_state, mb_rng), (loss_val, aux_vals, avg_grad_norm)
                else:
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
                if is_memory_method:
                    (train_state, _), losses_and_grads = jax.lax.scan(
                        _update_minbatch, (train_state, rng), minibatches
                    )
                else:
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
                if is_memory_method:
                    (train_state, rng, update_steps, cl_carry) = update_runner_state
                else:
                    (train_state, rng, update_steps) = update_runner_state
                # Init envs & partner indices
                rng, reset_rng, p_rng = jax.random.split(rng, 3)
                reset_rngs = jax.random.split(reset_rng, config.num_envs)
                init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config.num_envs), dtype=bool) for k in env.agents + ["__all__"]}
                new_partner_indices = partner_population.sample_agent_indices(config.num_uncontrolled_actors, p_rng)

                # 1) rollout
                runner_state = (train_state, init_env_state, init_obs, init_done,
                                init_ego_hstate, init_partner_hstate, new_partner_indices, rng,
                                update_steps)

                runner_state, (traj_batch, team_ep_soups) = jax.lax.scan(
                    _env_step, runner_state, None, config.num_steps)
                (train_state, env_state, obs, done, ego_hstate, partner_hstate, partner_indices, rng, _) = runner_state

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

                # Update per-task AGEM/ER-ACE memory buffer after all PPO epochs
                if is_memory_method:
                    rng = update_state[5]
                    cl_carry, rng = update_agem_memory(
                        config.agem_sample_size, env_id_idx, advantages,
                        cl_carry, rng, targets, traj_batch,
                    )

                # Compact per-update metrics. Per-step rollout information stays inside this function.
                # Episode statistics use only episodes that completed within the rollout.
                ep_mask = traj_batch.info["returned_episode"]  # (num_steps, num_envs)
                n_episodes = ep_mask.sum()
                denom = jnp.maximum(n_episodes, 1)
                metric = {
                    "update_steps": update_steps,
                    "actor_loss": loss_terms[1].mean(),
                    "value_loss": loss_terms[0].mean(),
                    "entropy_loss": loss_terms[2].mean(),
                    "cl_penalty": loss_terms[3].mean(),
                    "avg_grad_norm": avg_grad_norm.mean(),
                    "n_episodes": n_episodes,
                    "team_soup": jnp.where(ep_mask, team_ep_soups, 0.0).sum() / denom,
                    "ego_return": jnp.where(ep_mask, traj_batch.info["returned_episode_returns"], 0.0).sum() / denom,
                }
                if is_memory_method:
                    new_runner_state = (train_state, rng, update_steps + 1, cl_carry)
                else:
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

            def _run_eval(ego_params, event_key):
                """Evaluate the ego on every evaluation partner; returns compact (P, E) arrays."""
                soups, rets, done, ids = [], [], [], []
                for policy, params, group_ids in zip(eval_policies, eval_params, eval_ids):
                    last_infos = jax.vmap(lambda p, i: run_episodes(
                        event_key, env,
                        agent_0_param=ego_params, agent_0_policy=ego_policy,
                        agent_1_param=p, agent_1_policy=policy,
                        max_episode_steps=max_episode_steps,
                        env_id_idx=i,
                        num_eps=config.num_eval_episodes
                    ))(params, group_ids)
                    s, r, c = team_episode_summary(
                        last_infos["returned_episode_soups"], last_infos["returned_episode_returns"],
                        last_infos["returned_episode"])
                    soups.append(s), rets.append(r), done.append(c), ids.append(group_ids)
                return (jnp.concatenate(ids), jnp.concatenate(soups), jnp.concatenate(rets),
                        jnp.concatenate(done))

            def _on_eval_event(args):
                stage_id, update_steps, train_episodes, partner_ids, soups, rets, completed = args
                _, log_dict = build_eval_event(
                    stage_id, update_steps, train_episodes, partner_ids, soups, rets, completed,
                    num_updates, steps_per_update, max_soup_val)
                emit(log_dict)

            def _on_train_step(args):
                (stage_id, update_steps, n_episodes, team_soup, ego_return,
                 value_loss, actor_loss, entropy_loss, grad_norm) = args
                emit(build_train_log(
                    stage_id, update_steps, num_updates, n_episodes, team_soup, ego_return,
                    value_loss, actor_loss, entropy_loss, grad_norm, max_soup_val))

            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_state, checkpoint_array, ckpt_idx, train_episodes) = state_with_ckpt

                # Single PPO update
                new_update_state, metric = _update_step(
                    update_state,
                    None
                )
                if is_memory_method:
                    (train_state, rng, update_steps, cl_carry) = new_update_state
                else:
                    (train_state, rng, update_steps) = new_update_state
                train_episodes = train_episodes + metric["n_episodes"]

                # To eval or not to eval
                to_eval = should_evaluate(update_steps, eval_every, num_updates)
                # Only store a checkpoint at the very end when num_ckpts == 1
                to_store_ckpt = jnp.equal(update_steps, config.num_updates)

                def do_eval(_):
                    # The evaluation key depends only on the update index; the training rng is untouched.
                    event_key = jax.random.fold_in(eval_key, update_steps)
                    ids, soups, rets, completed = _run_eval(train_state.params, event_key)
                    jax.experimental.io_callback(
                        _on_eval_event, None,
                        (env_id_idx, update_steps, train_episodes, ids, soups, rets, completed),
                        ordered=False,
                    )
                    return jnp.array(True)

                def skip_eval(_):
                    return jnp.array(False)

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

                metric["evaluated"] = jax.lax.cond(to_eval, do_eval, skip_eval, None)

                # --- In-scan logging via io_callback (same approach as ippo.py) ---
                jax.experimental.io_callback(
                    _on_train_step, None,
                    (env_id_idx, update_steps, metric["n_episodes"], metric["team_soup"], metric["ego_return"],
                     metric["value_loss"], metric["actor_loss"], metric["entropy_loss"], metric["avg_grad_norm"]),
                    ordered=False,
                )

                if is_memory_method:
                    runner_state_out = (train_state, rng, update_steps, cl_carry)
                else:
                    runner_state_out = (train_state, rng, update_steps)
                return (runner_state_out, checkpoint_array, ckpt_idx, train_episodes), metric

            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            # initial runner state for scanning
            update_steps = 0

            if is_memory_method:
                update_runner_state = (train_state, rng, update_steps, cl_state)
            else:
                update_runner_state = (train_state, rng, update_steps)
            state_with_ckpt = (update_runner_state, checkpoint_array, ckpt_idx, jnp.zeros((), jnp.int32))

            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config.num_updates
            )
            (final_runner_state, checkpoint_array, final_ckpt_idx, _) = state_with_ckpt
            out = {
                "final_params": final_runner_state[0].params,
                "metrics": metrics,  # compact scalars, shape (NUM_UPDATES,)
                "checkpoints": checkpoint_array,
            }
            if is_memory_method:
                out["final_cl_state"] = final_runner_state[3]
            return out

        return train

    # ------------------------------
    # Actually run the PPO training
    # ------------------------------
    stage_id = jnp.asarray(env_id_idx, jnp.int32)
    rngs = jax.random.split(train_rng, n_ego_train_seeds)
    eval_rngs = jax.random.split(eval_rng, n_ego_train_seeds)
    if n_ego_train_seeds == 1:
        if compiled_cache is None:
            train_fn = jax.jit(make_ppo_train(config))
        else:
            cache_key = (id(config), env, ego_policy, type(partner_population), partner_population.policy_cls,
                         getattr(partner_population, "test_mode", None), cl, type(cl_state), log_fn, max_soup,
                         eval_policies)
            if cache_key not in compiled_cache:
                compiled_cache[cache_key] = jax.jit(make_ppo_train(config))
            train_fn = compiled_cache[cache_key]
        out = train_fn(rngs[0], eval_rngs[0], init_ego_params, partner_params, cl_state, stage_id,
                       eval_params, eval_ids)
    else:
        train_fn = jax.jit(jax.vmap(make_ppo_train(config), in_axes=(0, 0, None, None, None, None, None, None)))
        out = train_fn(rngs, eval_rngs, init_ego_params, partner_params, cl_state, stage_id, eval_params, eval_ids)
    jax.block_until_ready(out)
    jax.effects_barrier()  # flush unordered io_callbacks so every event is logged before returning
    return out


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
