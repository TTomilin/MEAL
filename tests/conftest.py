"""Shared fixtures for the MEAL test suite.

Note: exposed as pytest fixtures rather than importable module-level names,
because a user-site-installed `tests` package (from an unrelated pip
dependency) shadows this directory as `tests.*` for plain imports — pytest's
own conftest discovery is unaffected since it doesn't go through `import
tests.conftest`.
"""
import os

# The suite doesn't need GPU throughput and CI/dev boxes may be running other
# JAX processes concurrently; force CPU so tests don't flake on GPU OOM.
# Must be set before `jax` is imported anywhere.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import pytest

# Discrete action aliases shared by every scripted-rollout test.
ACTIONS = {'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5}


@pytest.fixture
def actions():
    return dict(ACTIONS)


@pytest.fixture
def rollout():
    """Step an env through a scripted rollout.

    Returns a `rollout(env, rng, actions_by_agent, state=None)` callable:
        env: an Overcooked-family env exposing `.step_env(key, state, actions)`.
        rng: a PRNGKey. If `state` is None, it is also used to reset the env.
        actions_by_agent: dict {agent_id: [action, ...]} of equal-length lists.
        state: optional pre-existing state to continue from (skips reset).

    Returns (states, rewards, shaped_rewards, infos) — lists, one entry per
    step (post-reset state included as states[0]).
    """

    def _rollout(env, rng, actions_by_agent, state=None):
        if state is None:
            rng, reset_key = jax.random.split(rng)
            _, state = env.reset(reset_key)

        n_steps = len(next(iter(actions_by_agent.values())))
        states = [state]
        rewards, shaped_rewards, infos = [], [], []

        for t in range(n_steps):
            rng, step_key = jax.random.split(rng)
            step_actions = {agent: jax.numpy.uint32(acts[t]) for agent, acts in actions_by_agent.items()}
            _, state, rew, done, info = env.step_env(step_key, state, step_actions)
            states.append(state)
            rewards.append(rew)
            shaped_rewards.append(info["shaped_reward"])
            infos.append(info)

        return states, rewards, shaped_rewards, infos

    return _rollout
