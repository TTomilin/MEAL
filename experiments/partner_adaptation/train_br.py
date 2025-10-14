'''Train an ego agent against a *single* partner agent.
Supports training against both RL and heuristic partner agents.
'''
import logging
import time
from functools import partial

import jax
import jax.numpy as jnp

from experiments.partner_adaptation.partner_agents.population_interface import AgentPopulation
from experiments.partner_adaptation.partner_generation.utils import get_metric_names
from experiments.partner_adaptation.train_ego import train_ppo_ego_agent, log_metrics

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
        num_envs = agent_indices.squeeze().shape[0]
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
        num_envs = agent_indices.squeeze().shape[0]
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


def run_br_training(
        config, env, partner_agent_config, ego_policy, ego_params, partner_policy, partner_params=None,
        partner_test_mode=False, env_id_idx=0, eval_partner=[], max_soup_dict=None, layout_names=None, cl=None,
        cl_state=None):
    '''Run ego agent training against a single partner agent.

    Args:
        max_soup_dict: dict, maximum soup counts for each layout (for unified soup metrics)
        layout_names: list, names of layouts/partners for evaluation metrics
    '''
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, train_rng = jax.random.split(rng, 3)

    if partner_params is not None:  # RL agent
        partner_params = jax.tree.map(
            lambda x: x[jnp.newaxis, ...], partner_params)
        partner_population = DummyPolicyPopulation(
            policy_cls=partner_policy,
            test_mode=partner_test_mode
        )

    else:  # heuristic agent
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
        train_rng=train_rng,
        ego_policy=ego_policy,
        init_ego_params=ego_params,
        n_ego_train_seeds=1,
        partner_population=partner_population,
        partner_params=partner_params,
        env_id_idx=env_id_idx,
        eval_partner=eval_partner,
        cl=cl,
        cl_state=cl_state
    )

    log.info(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Update continual learning state after training if CL method is specified
    if cl is not None and cl_state is not None:
        # Compute importance weights for the parameters after training
        importance = cl.compute_importance(
            out["final_params"], env, ego_policy.network, env_id_idx, train_rng,
            config.use_cnn, config.importance_episodes, config.importance_steps,
            config.normalize_importance
        )

        # Update the CL state with new parameters and importance
        cl_state = cl.update_state(cl_state, out["final_params"], importance)
        log.info(f"Updated CL state after training on partner {env_id_idx}")

    # process and log metrics
    metric_names = get_metric_names("overcooked")
    log_metrics(config, out, metric_names, max_soup_dict, layout_names)

    return out["final_params"], cl_state
