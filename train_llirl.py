"""
Train LLIRL only
"""

import sys
import os
import subprocess

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    llirl_dir = os.path.join(base_dir, "llirl_sumo")
    
    # Environment configuration
    sumo_config = "../nets/120p4k/run_120p4k.sumocfg"
    model_path = "saves/120p4k"
    output_path = "output/120p4k"
    
    # Step 1: Environment clustering
    clustering_cmd = [
        sys.executable, "env_clustering.py",
        "--sumo_config", sumo_config,
        "--model_path", model_path,
        "--et_length", "1",
        "--num_periods", "5",
        "--device", "cuda",  # Using GPU
        "--seed", "1009",
        "--batch_size", "8",
        "--env_num_layers", "2",
        "--env_hidden_size", "200",
        "--H", "4",
        "--max_steps", "7200"
    ]
    
    print("=" * 60)
    print("Step 1: Environment Clustering (GPU, 5 periods, 120p4k network)...")
    print(f"SUMO Config: {sumo_config}")
    print("=" * 60)
    subprocess.run(clustering_cmd, cwd=llirl_dir, check=True)
    
    # Step 2: Policy training (with optimizations)
    policy_cmd = [
        sys.executable, "policy_training.py",
        "--sumo_config", sumo_config,
        "--model_path", model_path,
        "--output", output_path,
        "--algorithm", "reinforce",
        "--opt", "adam",  # Changed to Adam for better convergence
        "--lr", "0.003",  # Lower initial LR for stability
        "--num_iter", "50",
        "--num_periods", "5",
        "--device", "cuda",  # Using GPU
        "--seed", "1009",
        "--batch_size", "8",
        "--hidden_size", "200",
        "--num_layers", "2",
        "--use_general_policy",
        "--num_test_episodes", "3",
        "--policy_eval_weight", "0.5",
        "--max_steps", "7200",
        "--use_baseline",  # Enable baseline for variance reduction
        "--lr_decay", "0.98",  # Learning rate decay per period
        "--lr_min", "1e-5",  # Minimum learning rate
        "--early_stop_patience", "10",  # Early stopping
        "--early_stop_threshold", "0.01",
        "--grad_clip", "0.5"  # Gradient clipping
    ]
    
    print("\n" + "=" * 60)
    print("Step 2: Policy Training (GPU, 5 periods, 120p4k network, with General Policy)...")
    print(f"SUMO Config: {sumo_config}")
    print("=" * 60)
    subprocess.run(policy_cmd, cwd=llirl_dir, check=True)
    
    print("\n" + "=" * 60)
    print("LLIRL training completed!")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  - Models: {llirl_dir}/{model_path}/")
    print(f"  - Output: {llirl_dir}/{output_path}/")
