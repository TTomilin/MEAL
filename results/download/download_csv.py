import argparse
import os
from pathlib import Path

import wandb
from wandb.apis.public import Run

from results.download.common import want, cli

FORBIDDEN_TAGS = ['TEST', 'LOCAL']
TAG_TO_FOLDER = {}


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        if want(run, args):
            store_data(run, args)


def store_data(run: Run, args: argparse.Namespace) -> None:
    cfg = run.config

    algo = cfg.get("alg_name")
    cl_method = cfg.get("cl_method", "UNKNOWN_CL")

    # --- Temporary replacements because old runs are still using the old names --- #
    if algo == 'ippo_cbp':
        algo = 'ippo'
        cl_method = 'CBP'
    if 'EWC' in run.name:
        cl_method = 'EWC'
        if cfg.get("ewc_mode") == "online":
            cl_method = "Online EWC"
    elif 'MAS' in run.name:
        cl_method = 'MAS'
    elif cl_method is None:
        cl_method = "FT"
    # --- Temporary replacements because old runs are still using the old names --- #

    seq_length = cfg['seq_length']
    seed = cfg['seed']
    strategy = cfg['strategy']
    arch = "CNN" if cfg.get("use_cnn") else "MLP"
    # tag = cfg['wandb_tags'][0]

    # Construct folder path for each configuration
    # folder_path = os.path.join(args.output, f"{TAG_TO_FOLDER[tag]}", algo, cl_method, f"{strategy}_{seq_length}")
    base_dir = Path(__file__).resolve().parent.parent
    folder_path = os.path.join(base_dir, args.output, algo, cl_method, arch, f"{strategy}_{seq_length}", f"seed_{seed}")
    os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists

    # Filename based on metric
    file_name = f"summary.csv"
    file_path = os.path.join(folder_path, file_name)

    # If the file already exists and we don't want to overwrite, skip
    if not args.overwrite and os.path.exists(file_path):
        print(f"Skipping already existing: {file_path}")
        return

    # Attempt to retrieve and save the data
    try:
        df = run.history()
        df.to_csv(file_path, index=False)
        print(f"Successfully stored run: {run.name} to {file_path}")
    except Exception as e:
        print(f"Error downloading data for run {run.name} to {file_path}", e)


if __name__ == "__main__":
    main(cli())
