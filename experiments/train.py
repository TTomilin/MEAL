from typing import Union

import tyro
from typing_extensions import Annotated

from experiments.algos.happo import HAPPO, HAPPOConfig
from experiments.algos.ippo import IPPO, IPPOConfig
from experiments.algos.mappo import MAPPO, MAPPOConfig
from experiments.algos.qmix import QMIX, QMIXConfig
from experiments.algos.vdn import VDN, VDNConfig

ALGOS = {
    IPPOConfig: IPPO,
    MAPPOConfig: MAPPO,
    HAPPOConfig: HAPPO,
    VDNConfig: VDN,
    QMIXConfig: QMIX,
}

AlgoConfig = Union[
    Annotated[IPPOConfig, tyro.conf.subcommand("ippo")],
    Annotated[MAPPOConfig, tyro.conf.subcommand("mappo")],
    Annotated[HAPPOConfig, tyro.conf.subcommand("happo")],
    Annotated[VDNConfig, tyro.conf.subcommand("vdn")],
    Annotated[QMIXConfig, tyro.conf.subcommand("qmix")],
]


def main():
    cfg = tyro.cli(AlgoConfig, default=IPPOConfig())
    ALGOS[type(cfg)](cfg).run()


if __name__ == "__main__":
    print("Running main...")
    main()
