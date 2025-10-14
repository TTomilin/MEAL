from meal.env import Overcooked, OvercookedPO, OvercookedLegacy


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off of Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered.")
    elif env_id == "overcooked":
        env = Overcooked(**env_kwargs)
    elif env_id == "overcooked_po":
        env = OvercookedPO(**env_kwargs)
    elif env_id == "overcooked_legacy":
        env = OvercookedLegacy(**env_kwargs)
    return env


registered_envs = ["overcooked", "overcooked_po", "overcooked_legacy"]
