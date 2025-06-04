#!/bin/bash

seeds=(0 1 2 3 4)
learning_rates=(3e-4 5e-4 7e-4)

# the two mutually–exclusive CLI flags tyro understands
anneal_flags=(--anneal_lr --no-anneal_lr)

for seed in "${seeds[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for anneal_flag in "${anneal_flags[@]}"; do

      # readable tag for wandb, taken from the flag that’s in use
      if [[ "$anneal_flag" == "--anneal_lr" ]]; then
        tag="anneal_lr"
      else
        tag="constant_lr"
      fi

      echo "submitting job: seed=$seed  lr=$lr  $anneal_flag"

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

PYTHONPATH=\$HOME/JAXOvercooked python3 \$HOME/JAXOvercooked/baselines/IPPO_MLP.py \
  --seq_length=5 \
  --lr $lr \
  --seed $seed \
  $anneal_flag \
  --tags $lr $tag "experiment_1"
EOF

    done
  done
done