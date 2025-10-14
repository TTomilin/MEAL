#!/bin/bash

algo=( IPPO MAPPO )
methods=( L2 EWC MAS AGEM )
strategies=( generate )
models=( no-use-cnn )
task_ids=( use-task-id )
multiheads=( use-multihead )
layer_norms=( use-layer-norm )
agents=( 1 2 3 4 )
seeds=( 1 2 3 4 5 )
difficulties=( easy medium hard )
env_name=overcooked_n_agent
seq_length=20
ewc_mode=multi

for method in "${methods[@]}"; do
    # Assign regularization coefficient per method
    case "$method" in
        EWC) coef=1e11 ;;
        MAS) coef=1e9  ;;
        L2)  coef=1e7  ;;
        *)   coef=1    ;;  # doesn’t matter for others
    esac

    # Build the reg_coef argument (omit for FT)
    if [ -n "$coef" ]; then
        reg_arg="--reg_coef $coef"
    else
        reg_arg=""
    fi
    for strategy in "${strategies[@]}"; do
        for model in "${models[@]}"; do
            for task_id in "${task_ids[@]}"; do
                for multihead in "${multiheads[@]}"; do
                    for layer_norm in "${layer_norms[@]}"; do
                        for num_agents in "${agents[@]}"; do
			                      for diff in "${difficulties[@]}"; do
                                for seed in "${seeds[@]}"; do
                                    echo "Submitting job for combination: $algo, $method, Agents=$num_agents, Seed=$seed, $strategy, $model, $task_id, $multihead, coef=$coef"

                                    # Create an SBATCH script
                                    cat <<EOF | sbatch
#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time 3:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=MEAL_${method}_${model}_${strategy}_Seed_${seed}
#SBATCH -o /home/ttomilin/slurm/%j_"${algo}"_"${method}"_"${model}"_"${num_agents}"agents_Seed_"${seed}"_$(date +%Y-%m-%d-%H-%M-%S).out
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH=~/MEAL:$PYTHONPATH
conda activate meal
python3 ~/MEAL/baselines/${algo}_CL.py \
        --no_regularize_heads \
        --anneal_lr \
	      --record_gif \
	      --random-reset \
        --cl_method $method \
        --reg_coef $coef \
        --ewc_mode $ewc_mode \
        --seq_length $seq_length \
	      --env_name $env_name \
	      --difficulty $diff \
        --$layer_norm \
        --$task_id \
        --$multihead \
        --strategy $strategy \
        --$model \
        --seed $seed \
        --tags BASELINE_COMPARISON
conda deactivate
EOF
                                    sleep 0.1
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
