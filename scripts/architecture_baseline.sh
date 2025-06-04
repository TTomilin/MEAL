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

seeds=(0 1)
for seed in "${seeds[@]}"; do
    echo "Running with seed ${seed}"
    PYTHONPATH=$HOME/JAXOvercooked python3 ~/JAXOvercooked/baselines/IPPO_MLP.py --seq_length=5  --seed="$seed" --no-anneal_lr --tags "MLP baseline" "seed $seed" "experiment 2"
    PYTHONPATH=$HOME/JAXOvercooked python3 ~/JAXOvercooked/baselines/IPPO_CNN.py --seq_length=5  --seed="$seed" --no-anneal_lr --tags "CNN baseline" "seed $seed" "experiment 2"
    PYTHONPATH=$HOME/JAXOvercooked python3 ~/JAXOvercooked/baselines/IPPO_decoupled_MLP.py --seq_length=5  --seed="$seed" --no-anneal_lr --tags "decoupled MLP baseline" "seed $seed" "experiment 2"
    PYTHONPATH=$HOME/JAXOvercooked python3 ~/JAXOvercooked/baselines/IPPO_shared_MLP.py --seq_length=5  --seed="$seed" --no-anneal_lr --tags "shared MLP baseline" "seed $seed" "experiment 2"
done



