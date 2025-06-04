#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time 12:00:00
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
#pip install -r ~/JAXOvercooked/requirements.txt
python --version

echo $$

seeds=(0 1 2 3 4)
learning_rates=(1e-4 3e-4 5e-4 7e-4)

for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    for lr in "${learning_rates[@]}"; do
        echo "Running with learning rate: $lr"
        PYTHONPATH=$HOME/JAXOvercooked python3 ~/JAXOvercooked/baselines/IPPO_MLP.py --seq_length=5 --lr="$lr" --seed="$seed" --no-anneal_lr --tags "$lr"  "constant_lr"
        PYTHONPATH=$HOME/JAXOvercooked python3 ~/JAXOvercooked/baselines/IPPO_MLP.py --seq_length=5 --lr="$lr" --seed="$seed" --anneal_lr --tags "$lr" "anneal_lr"
    done
done
