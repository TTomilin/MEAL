#!/bin/bash

seeds=(0 1 2 3 4)
architectures=(IPPO_MLP IPPO_CNN IPPO_decoupled_MLP IPPO_shared_MLP)
tags=(
  "MLP baseline"
  "CNN baseline"
  "decoupled MLP baseline"
  "shared MLP baseline"
)

for idx in "${!architectures[@]}"; do
  architecture="${architectures[$idx]}"
  tag="${tags[$idx]}"

  for seed in "${seeds[@]}"; do
    echo "Submitting $architecture with seed=$seed"

    cat <<EOF | sbatch

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
source /etc/bashrc
source ~/.bashrc
source ~/venv/bin/activate

PYTHONPATH=\$HOME/JAXOvercooked python \$HOME/JAXOvercooked/baselines/${architecture}.py \
  --seq_length 5 \
  --seed ${seed} \
  --no-anneal_lr \
  --tags "${tag}" "seed ${seed}" "experiment 2"

EOF

    done
  done
done