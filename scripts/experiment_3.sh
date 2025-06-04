#!/bin/bash

seeds=(0 1 2 3 4)
models=(IPPO_shared_MLP IPPO_multihead_L2 IPPO_shared_MLP_AGEM IPPO_shared_MLP_EWC IPPO_shared_MLP_MAS IPPO_MLP_CBP)
tags=(
  "MLP shared baseline"
  "L2 regularization"
  "AGEM"
  "EWC"
  "MAS"
  "CBP"
)

for idx in "${!models[@]}"; do
  model="${models[$idx]}"
  tag="${tags[$idx]}"

  for seed in "${seeds[@]}"; do
    echo "Submitting $model with seed=$seed"

    sbatch <<EOF
#!/bin/bash
#SBATCH -p gpu_mig
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

PYTHONPATH=\$HOME/JAXOvercooked python \$HOME/JAXOvercooked/baselines/${model}.py \
  --seq_length 5 \
  --seed ${seed} \
  --no-anneal_lr \
  --tags "${tag}" "seed ${seed}" "experiment 3"
EOF
  done
done