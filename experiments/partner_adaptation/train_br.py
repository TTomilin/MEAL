'''Train an ego agent against a *single* partner agent.
Supports training against both RL and heuristic partner agents.
'''
import logging
import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from experiments.partner_adaptation.partner_agents.population_interface import AgentPopulation
from experiments.partner_adaptation.train_ego import make_stage_keys, train_ppo_ego_agent

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DummyPolicyPopulation(AgentPopulation):
    '''A wrapper around the AgentPopulation class that allows for a single policy to be used.
    The main difference from the AgentPopulation is that the test mode is a class attribute, 
    so it remains static for the lifetime of the object
    '''

    def __init__(self, policy_cls, test_mode=False):
        super().__init__(pop_size=1, policy_cls=policy_cls)
        self.test_mode = test_mode

    def get_actions(self, pop_params, agent_indices, obs, done, avail_actions, hstate, rng,
                    env_state=None, aux_obs=None):
        '''
        Get the actions of the agents specified by agent_indices. Does not support agents that 
        require auxiliary observations.
        Returns:
            actions: actions with shape (num_envs,)
            new_hstate: new hidden state with shape (num_envs, ...) or None
        '''
        gathered_params = self.gather_agent_params(pop_params, agent_indices)
        num_envs = agent_indices.shape[0]
        rngs_batched = jax.random.split(rng, num_envs)
        vmapped_get_action = jax.vmap(partial(self.policy_cls.get_action,
                                              aux_obs=aux_obs,
                                              env_state=env_state,
                                              test_mode=self.test_mode))
        actions, new_hstate = vmapped_get_action(
            gathered_params, obs, done, avail_actions, hstate,
            rngs_batched)
        return actions, new_hstate

    def init_hstate(self, n: int):
        '''Initialize the hidden state for n members of the population.'''
        hstate = self.policy_cls.init_hstate(n)
        return hstate


class HeuristicPolicyPopulation(AgentPopulation):
    '''A wrapper around the AgentPopulation class that allows for a heuristic policy to be used.
    The main difference from the AgentPopulation is that:
    - test mode is not used b/c heuristic agents do not have a test mode
    - get_actions requires the environment state
    - the init_hstate method is overridden to vmap over the hidden state initialization.
    '''

    def __init__(self, policy_cls):
        super().__init__(pop_size=1, policy_cls=policy_cls)

    def get_actions(self, pop_params, agent_indices, obs, done, avail_actions, hstate, rng,
                    env_state, aux_obs=None):
        '''
        Get the actions of the agents specified by agent_indices. Requires env_state. 
        Does not support agents that require auxiliary observations.
        Returns:
            actions: actions with shape (num_envs,)
            new_hstate: new hidden state with shape (num_envs, ...) or None
        '''
        gathered_params = self.gather_agent_params(pop_params, agent_indices)
        num_envs = agent_indices.shape[0]
        rngs_batched = jax.random.split(rng, num_envs)

        def _policy_cls_get_action(params, obs, done, avail_actions, hstate, rng, env_state
                                   ):
            return self.policy_cls.get_action(params=params, obs=obs, done=done,
                                              avail_actions=avail_actions, hstate=hstate,
                                              rng=rng, env_state=env_state,
                                              aux_obs=None, test_mode=False)

        vmapped_get_action = jax.vmap(_policy_cls_get_action)
        actions, new_hstate = vmapped_get_action(
            params=gathered_params,
            obs=obs,
            done=done,
            avail_actions=avail_actions,
            hstate=hstate,
            rng=rngs_batched,
            env_state=env_state)
        return actions, new_hstate

    def init_hstate(self, n: int):
        '''Initialize the hidden state for n members of the population.'''
        # partner agent is always agent 1 in the ppo_ego training code
        vmap_dummy_input = jnp.ones(n)
        return jax.vmap(partial(self.policy_cls.init_hstate, aux_info={"agent_id": 1}))(vmap_dummy_input)


@struct.dataclass
class PartnerRolloutState:
    """Env state plus everything the frozen partner needs to act on the next step."""
    env_state: Any
    partner_obs: Any
    partner_done: Any
    partner_hstate: Any


def make_partner_switches(env, partner_population, partner_params):
    """reset/step switches for single-ego importance rollouts in which the frozen partner acts.

    They have the (reset_switch, step_switch) signature the EWC/MAS importance estimators already use,
    so the estimators are unchanged. The partner is queried through `partner_population.get_actions`,
    the same interface as in training: available actions come from the env, the done flag of the previous
    step drives the partner's own reset handling, and heuristic (planner) partners receive the wrapped env
    state and carry their agent state across steps. The env auto-resets, as in training.

    Args:
        partner_params: partner parameters with a leading population axis of size 1, as in training.
    """
    ego, partner = env.agents[0], env.agents[1]

    def reset_switch(key, task_idx):
        obs, env_state = env.reset(key)
        state = PartnerRolloutState(
            env_state=env_state,
            partner_obs=obs[partner],
            partner_done=jnp.zeros((), dtype=bool),
            partner_hstate=partner_population.init_hstate(1),
        )
        return obs, state

    def step_switch(key, state, actions, task_idx):
        step_key, partner_key = jax.random.split(key)
        avail = env.get_avail_actions(state.env_state.env_state)[partner].astype(jnp.float32)
        act_partner, partner_hstate = partner_population.get_actions(
            partner_params,
            jnp.zeros((1,), dtype=jnp.int32),
            state.partner_obs.reshape(1, 1, -1),
            state.partner_done.reshape(1, 1, 1),
            avail[None],
            state.partner_hstate,
            partner_key,
            env_state=jax.tree.map(lambda x: x[None], state.env_state),
            aux_obs=None,
        )
        joint = {ego: actions[ego], partner: act_partner.reshape(1).astype(actions[ego].dtype)}
        obs, env_state, reward, done, info = env.step(step_key, state.env_state, joint)
        next_state = PartnerRolloutState(
            env_state=env_state,
            partner_obs=obs[partner],
            partner_done=done["__all__"],
            partner_hstate=partner_hstate,
        )
        return obs, next_state, reward, done, info

    return reset_switch, step_switch


def make_partner_importance_fn(cl, env, ego_network, partner_population, config):
    """Jitted `fn(ego_params, env_idx, rng, partner_params)` computing CL importance with the partner acting.

    Partner parameters are an argument, so one compiled function serves every partner that shares a
    policy class and parameter shapes.
    """

    def importance(ego_params, env_idx, rng, partner_params):
        reset_switch, step_switch = make_partner_switches(env, partner_population, partner_params)
        fn = cl.make_importance_fn(
            reset_switch, step_switch, ego_network, [env.agents[0]], config.use_cnn,
            config.importance_episodes, config.importance_steps, config.normalize_importance,
            config.importance_stride)
        return fn(ego_params, env_idx, rng)

    return jax.jit(importance)


def run_br_training(
        config, env, partner_agent_config, ego_policy, ego_params, partner_policy, partner_params=None,
        partner_test_mode=False, env_id_idx=0, eval_partner=[], max_soup_dict=None, cl=None,
        cl_state=None, log_fn=None, compiled_cache=None, record_fn=None, stats=None,
        init_eval=False):
    '''Run ego agent training against a single partner agent.

    Args:
        max_soup_dict: dict, maximum soup counts for each layout (for unified soup metrics)
        log_fn: callable receiving metric dicts; defaults to wandb.log
        compiled_cache: dict shared by all stages of one run so compatible stages reuse compiled functions
        record_fn, stats: see `train_ppo_ego_agent`; `stats` also receives `importance_s` (wall time of the
            post-stage importance computation, including its compilation on first use) and
            `importance_compiled_here` when an importance estimate was computed
    '''
    keys = make_stage_keys(config.seed, env_id_idx)
    max_soup = list(max_soup_dict.values())[0] if max_soup_dict else None

    if partner_params is not None and not getattr(partner_policy, "is_planner", False):  # RL agent
        partner_params = jax.tree.map(
            lambda x: x[jnp.newaxis, ...], partner_params)
        partner_population = DummyPolicyPopulation(
            policy_cls=partner_policy,
            test_mode=partner_test_mode
        )

    else:  # heuristic agent
        if partner_params is not None:  # configurable planner: its params are its configuration
            partner_params = jax.tree.map(lambda x: x[jnp.newaxis, ...], partner_params)
        else:
            # Doesn't matter what we pass for params, since the heuristic agent doesn't use params.
            # We just need to pass something to vmap over.
            partner_params = jax.tree.map(
                lambda x: x[jnp.newaxis, ...], ego_params)
        partner_population = HeuristicPolicyPopulation(
            policy_cls=partner_policy
        )

    log.info("Starting ego agent training...")
    start_time = time.time()

    # Run the training
    out = train_ppo_ego_agent(
        config=config,
        env=env,
        train_rng=keys.train,
        ego_policy=ego_policy,
        init_ego_params=ego_params,
        n_ego_train_seeds=1,
        partner_population=partner_population,
        partner_params=partner_params,
        env_id_idx=env_id_idx,
        eval_partner=eval_partner,
        cl=cl,
        cl_state=cl_state,
        eval_rng=keys.eval,
        log_fn=log_fn,
        max_soup=max_soup,
        compiled_cache=compiled_cache,
        record_fn=record_fn,
        stats=stats,
        init_eval=init_eval,
    )

    log.info(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Update continual learning state after training if CL method is specified
    if cl is not None and cl_state is not None:
        if "final_cl_state" in out:
            # Memory-based methods (AGEM, ER-ACE): cl_state is updated inside the training scan
            cl_state = out["final_cl_state"]
            log.info(f"Updated memory CL state after training on partner {env_id_idx}")
        elif cl.name == "ft":
            # Plain fine-tuning stores nothing, so no importance rollout is needed.
            pass
        else:
            # Importance-based methods (EWC, MAS, L2): states come from rollouts where the frozen partner acts
            cache = compiled_cache if compiled_cache is not None else {}
            cache_key = ("importance", id(config), env, ego_policy, cl, type(partner_population),
                         partner_population.policy_cls, getattr(partner_population, "test_mode", None))
            first_use = cache_key not in cache
            if first_use:
                cache[cache_key] = make_partner_importance_fn(
                    cl, env, ego_policy.network, partner_population, config)
            importance_start = time.perf_counter()
            importance = cache[cache_key](
                out["final_params"], jnp.asarray(env_id_idx, jnp.int32), keys.importance, partner_params)
            jax.block_until_ready(importance)
            if stats is not None:
                stats["importance_s"] = time.perf_counter() - importance_start
                stats["importance_compiled_here"] = first_use
            cl_state = cl.update_state(cl_state, out["final_params"], importance)
            log.info(f"Updated CL state after training on partner {env_id_idx}")

    return out["final_params"], cl_state
