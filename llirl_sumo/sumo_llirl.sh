#!/bin/bash
# Script to run LLIRL for SUMO single intersection environment

# Step 1: Cluster the environments using the Dirichlet mixture model with CRP
echo "Step 1: Environment Clustering..."
python env_clustering.py --sumo_config ../nets/single-intersection/run_morning_6to10.sumocfg \
    --model_path saves/sumo_single_intersection \
    --et_length 1 --num_periods 30 --device cpu --seed 1009 \
    --batch_size 8 --env_num_layers 2 --env_hidden_size 200 --H 4

# Step 2: Train the policies according to built library
echo "Step 2: Policy Training..."
python policy_training.py --sumo_config ../nets/single-intersection/run_morning_6to10.sumocfg \
    --model_path saves/sumo_single_intersection \
    --output output/sumo_single_intersection --algorithm reinforce --opt sgd --lr 0.01 \
    --num_iter 50 --num_periods 30 --device cpu --seed 1009 \
    --batch_size 8 --hidden_size 200 --num_layers 2

echo "Training completed!"

