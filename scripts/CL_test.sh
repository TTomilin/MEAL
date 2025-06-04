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
seq_lengths=(5 10 25)
for seed in "${seeds[@]}"; do
    echo "Running with seed ${seed}"
    for seq_length in "${seq_lengths[@]}"; do
        echo "Running with seq_length ${seq_length}"
        # Run the training script with the specified seed and seq_length
        PYTHONPATH=$HOME/JAXOvercooked python3 ~/JAXOvercooked/baselines/IPPO_MLP.py --seq_length="$seq_length"  --seed="$seed" --no-anneal_lr --tags "MLP baseline" "seed $seed" "experiment 3"
        PYTHONPATH=$HOME/JAXOvercooked python3 ~/JAXOvercooked/baselines/IPPO_CNN.py --seq_length="$seq_length"  --seed="$seed" --no-anneal_lr --tags "CNN baseline" "seed $seed" "experiment 3"
        PYTHONPATH=$HOME/JAXOvercooked python3 ~/JAXOvercooked/baselines/IPPO_shared_MLP_AGEM.py --seq_length="$seq_length"  --seed="$seed" --no-anneal_lr --tags "MLP AGEM" "seed $seed" "experiment 3"
        PYTHONPATH=$HOME/JAXOvercooked python3 ~/JAXOvercooked/baselines/IPPO_shared_MLP_EWC.py --seq_length="$seq_length"  --seed="$seed" --no-anneal_lr --tags "MLP EWC" "seed $seed" "experiment 3"
        PYTHONPATH=$HOME/JAXOvercooked python3 ~/JAXOvercooked/baselines/IPPO_shared_MLP_MAS.py --seq_length="$seq_length"  --seed="$seed" --no-anneal_lr --tags "MLP MAS" "seed $seed" "experiment 3"
        PYTHONPATH=$HOME/JAXOvercooked python3 ~/JAXOvercooked/baselines/IPPO_decoupled_MLP_Packnet.py --seq_length="$seq_length"  --seed="$seed" --no-anneal_lr --tags "MLP MAS" "seed $seed" "experiment 3"
    done
done



 