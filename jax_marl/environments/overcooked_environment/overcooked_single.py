import jax.numpy as jnp
from jax import lax

from jax_marl.environments.overcooked_environment.overcooked_n_agent import (
    Overcooked as BaseOvercooked)


class OvercookedSingle(BaseOvercooked):
    """One-chef flavour of Overcooked."""

    def __init__(self, **kwargs):
        super().__init__(num_agents=1, **kwargs)

        self.agents = ["agent_0"]

    # ------------------------------------------------------------------
    #  ↓ Everything that referenced agent_1 shrinks to scalar logic ↓
    # ------------------------------------------------------------------

    # 1. step_env handles a single action instead of a dict of two
    def step_env(self, key, state, actions):
        # cast to array because BaseOvercooked expects an array
        action = actions["agent_0"] if isinstance(actions, dict) else actions

        state, reward, shaped = self.step_agents(key, state, action)
        state = state.replace(time=state.time + 1, terminal=self.is_terminal(state))

        obs = self.get_obs(state)
        done = state.terminal

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            {"agent_0": reward},
            {"agent_0": done, "__all__": done},
            {"shaped_reward": {"agent_0": shaped}},
        )

    # 2. reset returns only one observation
    def reset(self, key):
        obs, state = super().reset(key)
        # obs is already a dict with {"agent_0": …}; nothing to tweak
        return obs, state

    # 3. get_obs generates one view instead of two
    def get_obs(self, state):
        # build exactly the same 26-layer tensor …
        full = super().get_obs(state)["agent_0"]
        return {"agent_0": full}
