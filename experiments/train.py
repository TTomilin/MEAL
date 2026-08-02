import tyro

from experiments.algos.happo import HAPPO, HAPPOConfig
from experiments.algos.ippo import IPPO, IPPOConfig
from experiments.algos.mappo import MAPPO, MAPPOConfig
from experiments.algos.qmix import QMIX, QMIXConfig
from experiments.algos.vdn import VDN, VDNConfig

ALGOS = {
    "ippo": (IPPO, IPPOConfig),
    "mappo": (MAPPO, MAPPOConfig),
    "happo": (HAPPO, HAPPOConfig),
    "vdn": (VDN, VDNConfig),
    "qmix": (QMIX, QMIXConfig),
}


def main():
    cfg = tyro.extras.subcommand_cli_from_dict(
        {name: config_cls for name, (_, config_cls) in ALGOS.items()}
    )
    algo_cls = {config_cls: algo_cls for algo_cls, config_cls in ALGOS.values()}[type(cfg)]
    algo_cls(cfg).run()


if __name__ == "__main__":
    print("Running main...")
    main()
