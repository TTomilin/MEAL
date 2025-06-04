#!/bin/bash

seeds=(0 1 2 3 4)
layouts=(
  cramped_room
  asymm_advantages
  coord_ring
  forced_coord
  counter_circuit
  square_arena
  split_kitchen
  basic_kitchen_large
  basic_kitchen_small
  shared_wall
  smallest_kitchen
  easy_layout
  big_kitchen
  no_cooperation
  forced_coord_2
  basic_cooperative
  corridor_challenge
  split_work
  resource_sharing
  efficiency_test
  c_kitchen
  most_efficient
  most_efficient_horizontal
  bottleneck_small
  bottleneck_large
)

for seed in "${seeds[@]}"; do
  for layout in "${layouts[@]}"; do
    echo "Submitting $layout with seed $seed"

    sbatch <<EOF
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

PYTHONPATH=\$HOME/JAXOvercooked python3 \$HOME/JAXOvercooked/baselines/reward_normalization.py\
  --seq_length=1 \
  --lr=2.5e-4 \
  --seed $seed \
  --anneal_lr\
  --layouts $layout\
  --tags $layout "single_env" "reward_normalization"
EOF

  done
done