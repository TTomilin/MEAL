#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time 1:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=overcooked
#SBATCH -o /home/lvdenboogaard/slurm/%j.out

module load 2022
module load 2023
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0

python --version

nvidia-sminvcc --version
. /etc/bashrc
. ~/.bashrc

source ~/venv/bin/activate

PYTHONPATH=$HOME/JAXOvercooked python3 ~/JAXOvercooked/baselines/IPPO_MLP.py \
    --seq_length=2 \
    --anneal_lr \
    --lr=2.5e-4 \
    --seed=0 \
    --tags "2.5e-4" "anneal_lr" "successful drop reward" \