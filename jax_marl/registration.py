from jax_marl.environments import Overcooked, SimpleMPE, SimplePushMPE, SimpleSpreadMPE, InTheGrid, InTheGrid_2p, InTheMatrix, LevelOne
from jax_marl.environments.overcooked_environment.overcooked_single import OvercookedSingle


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")
    
    # mpe
    if env_id == "MPE_simple_v3":
        env = SimpleMPE(**env_kwargs)
    elif env_id == "MPE_simple_push_v3":
        env = SimplePushMPE(**env_kwargs)
    elif env_id == "MPE_simple_spread_v3":
        env = SimpleSpreadMPE(**env_kwargs)
    # Overcooked
    elif env_id == "overcooked":
        env = Overcooked(**env_kwargs)
    elif env_id == "overcooked_single":
        env = OvercookedSingle(**env_kwargs)
    # Storm
    elif env_id == "storm":
        env = InTheGrid(**env_kwargs)
    elif env_id == "storm_2p":
        env = LevelOne(**env_kwargs)
    elif env_id == "storm_np":
        env = InTheMatrix(**env_kwargs)

    return env

registered_envs = ["overcooked", "overcooked_single", "MPE_simple_v3", "MPE_simple_push_v3", "MPE_simple_spread_v3", "storm", "storm_2p", "storm_np"]
