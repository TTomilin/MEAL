"""Env selection: `BaseConfig.env` is a tyro-subcommand union of the 4 EnvConfigs below,
so `python -m experiments.train <algo> env:overcooked --env.difficulty easy ...` (or
`env:mpe`/`env:smax`/`env:jaxnav`) picks both the environment and its own CLI fields.
`ENV_ADAPTERS` maps each EnvConfig type to the adapter that knows how to build/run it.
"""
from typing import Annotated, Union

import tyro

from experiments.envs.jaxnav import JaxNavAdapter, JaxNavEnvConfig
from experiments.envs.mpe import MPEAdapter, MPEEnvConfig
from experiments.envs.overcooked import OvercookedAdapter, OvercookedEnvConfig
from experiments.envs.smax import SMAXAdapter, SMAXEnvConfig

EnvConfig = Union[
    Annotated[OvercookedEnvConfig, tyro.conf.subcommand("overcooked")],
    Annotated[MPEEnvConfig, tyro.conf.subcommand("mpe")],
    Annotated[SMAXEnvConfig, tyro.conf.subcommand("smax")],
    Annotated[JaxNavEnvConfig, tyro.conf.subcommand("jaxnav")],
]

ENV_ADAPTERS = {
    OvercookedEnvConfig: OvercookedAdapter,
    MPEEnvConfig: MPEAdapter,
    SMAXEnvConfig: SMAXAdapter,
    JaxNavEnvConfig: JaxNavAdapter,
}


def build_env_adapter(env_cfg):
    return ENV_ADAPTERS[type(env_cfg)]()
