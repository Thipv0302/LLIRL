"""
Train LLIRL only
"""

import sys
import os
import subprocess

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    llirl_dir = os.path.join(base_dir, "llirl_sumo")
    
    # Step 1: Environment clustering
    clustering_cmd = [
        sys.executable, "env_clustering.py",
        "--sumo_config", "../nets/single-intersection/run_morning_6to10_10k.sumocfg",
        "--model_path", "saves/sumo_single_intersection",
        "--et_length", "1",
        "--num_periods", "30",
        "--device", "cuda",
        "--seed", "1009",
        "--batch_size", "8",
        "--env_num_layers", "2",
        "--env_hidden_size", "200",
        "--H", "4"
    ]
    
    print("Step 1: Environment Clustering...")
    subprocess.run(clustering_cmd, cwd=llirl_dir, check=True)
    
    # Step 2: Policy training
    policy_cmd = [
        sys.executable, "policy_training.py",
        "--sumo_config", "../nets/single-intersection/run_morning_6to10_10k.sumocfg",
        "--model_path", "saves/sumo_single_intersection",
        "--output", "output/sumo_single_intersection",
        "--algorithm", "reinforce",
        "--opt", "sgd",
        "--lr", "0.01",
        "--num_iter", "50",
        "--num_periods", "30",
        "--device", "cuda",
        "--seed", "1009",
        "--batch_size", "8",
        "--hidden_size", "200",
        "--num_layers", "2",
        "--use_general_policy",
        "--num_test_episodes", "3",
        "--policy_eval_weight", "0.5"
    ]
    
    print("\nStep 2: Policy Training (with General Policy)...")
    subprocess.run(policy_cmd, cwd=llirl_dir, check=True)
    
    print("\nLLIRL training completed!")
