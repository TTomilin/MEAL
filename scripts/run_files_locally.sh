#!/bin/bash
cd /home/luka/repo/JAXOvercooked

models=(IPPO_CNN
        IPPO_multihead_L2
        IPPO_decoupled_MLP_Packnet
        IPPO_decoupled_MLP
        IPPO_shared_MLP_AGEM 
        IPPO_shared_MLP_EWC 
        IPPO_shared_MLP_MAS 
        IPPO_MLP_CBP
        IPPO_shared_MLP
        vdn_cnn
        )

for model in "${models[@]}"; do
  echo "Running $model"
  # Run the model with the specified parameters
   python -m baselines.${model} --seq_length=2 --anneal_lr --evaluation --seed=0 --group="test"
done
