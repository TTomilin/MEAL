from jax_marl.environments import Overcooked
from jax_marl.environments.overcooked_environment.overcooked_n_agent import Overcooked as OvercookedNAgent


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")
    
    elif env_id == "overcooked":
        env = Overcooked(**env_kwargs)
    elif env_id == "overcooked_n_agent":
        env = OvercookedNAgent(**env_kwargs)

    return env

registered_envs = ["overcooked", "overcooked_n_agent"]
