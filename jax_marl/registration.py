from jax_marl.environments import Overcooked
from jax_marl.environments.overcooked.overcooked_n_agent import Overcooked as OvercookedNAgent
from jax_marl.environments.overcooked.overcooked_po import OvercookedPO


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")

    elif env_id == "overcooked":
        env = Overcooked(**env_kwargs)
    elif env_id == "overcooked_po":
        env = OvercookedPO(**env_kwargs)
    elif env_id == "overcooked_n_agent" or env_kwargs.get('num_agents', 2) != 2:
        env = OvercookedNAgent(**env_kwargs)

    return env


registered_envs = ["overcooked", "overcooked_single", "overcooked_po", "overcooked_n_agent"]
