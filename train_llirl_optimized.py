"""
Train LLIRL với các cải thiện tối ưu để đạt kết quả tốt nhất
Optimized LLIRL training script with best practices
"""

import sys
import os
import subprocess

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    llirl_dir = os.path.join(base_dir, "llirl_sumo")
    
    # Environment configuration
    sumo_config = "../nets/120p4k/run_120p4k.sumocfg"
    model_path = "saves/120p4k_optimized"
    output_path = "output/120p4k_optimized"
    
    print("=" * 80)
    print("LLIRL OPTIMIZED TRAINING")
    print("=" * 80)
    print(f"SUMO Config: {sumo_config}")
    print(f"Model Path: {model_path}")
    print(f"Output Path: {output_path}")
    print("=" * 80)
    
    # Step 1: Environment Clustering với parameters tối ưu
    print("\n" + "=" * 80)
    print("Step 1: Environment Clustering (OPTIMIZED)")
    print("=" * 80)
    print("Optimizations:")
    print("  - ZETA = 0.5 (dễ tạo cluster mới hơn)")
    print("  - SIGMA = 0.1 (tăng độ nhạy)")
    print("  - TAU1 = 0.5, TAU2 = 0.5 (temperature thấp hơn)")
    print("  - EM_STEPS = 5 (nhiều iterations hơn)")
    print("  - env_hidden_size = 256 (capacity lớn hơn)")
    print("  - env_num_layers = 3 (sâu hơn)")
    print("=" * 80)
    
    clustering_cmd = [
        sys.executable, "env_clustering.py",
        "--sumo_config", sumo_config,
        "--model_path", model_path,
        "--et_length", "1",
        "--num_periods", "5",
        "--device", "cuda",
        "--seed", "1009",
        "--batch_size", "8",
        "--env_num_layers", "3",  # Tăng từ 2 → 3
        "--env_hidden_size", "256",  # Tăng từ 200 → 256
        "--H", "4",
        "--max_steps", "7200",
        "--zeta", "0.5",  # Giảm từ 1.0 → 0.5
        "--sigma", "0.1",  # Giảm từ 0.25 → 0.1
        "--tau1", "0.5",  # Giảm từ 1.0 → 0.5
        "--tau2", "0.5",  # Giảm từ 1.0 → 0.5
        "--em_steps", "5"  # Tăng từ 1 → 5
    ]
    
    subprocess.run(clustering_cmd, cwd=llirl_dir, check=True)
    
    # Step 2: Policy Training với PPO và optimizations
    print("\n" + "=" * 80)
    print("Step 2: Policy Training (OPTIMIZED with PPO)")
    print("=" * 80)
    print("Optimizations:")
    print("  - Algorithm: PPO (thay vì REINFORCE)")
    print("  - Learning rate: 1e-3 (ổn định hơn)")
    print("  - num_iter: 100 (nhiều iterations hơn)")
    print("  - Baseline: linear (giảm variance)")
    print("  - Gradient clipping: 0.5")
    print("  - Learning rate decay: 0.95")
    print("  - policy_eval_weight: 0.7 (ưu tiên performance)")
    print("  - num_test_episodes: 5 (đánh giá chính xác hơn)")
    print("=" * 80)
    
    policy_cmd = [
        sys.executable, "policy_training.py",
        "--sumo_config", sumo_config,
        "--model_path", model_path,
        "--output", output_path,
        "--algorithm", "ppo",  # PPO thay vì reinforce
        "--opt", "adam",
        "--lr", "1e-3",  # Giảm từ 0.003 → 0.001
        "--num_iter", "100",  # Tăng từ 50 → 100
        "--num_periods", "5",
        "--device", "cuda",
        "--seed", "1009",
        "--batch_size", "8",
        "--hidden_size", "200",
        "--num_layers", "2",
        "--use_general_policy",
        "--num_test_episodes", "5",  # Tăng từ 3 → 5
        "--policy_eval_weight", "0.7",  # Tăng từ 0.5 → 0.7
        "--max_steps", "7200",
        "--use_baseline",  # Enable baseline
        "--baseline", "linear",
        "--lr_decay", "0.95",
        "--lr_min", "1e-5",
        "--early_stop_patience", "10",
        "--early_stop_threshold", "0.01",
        "--grad_clip", "0.5",
        "--clip", "0.2",  # PPO clip parameter
        "--epochs", "5",  # PPO epochs
        "--tau", "1.0"  # GAE tau
    ]
    
    subprocess.run(policy_cmd, cwd=llirl_dir, check=True)
    
    print("\n" + "=" * 80)
    print("LLIRL OPTIMIZED TRAINING COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - Models: {llirl_dir}/{model_path}/")
    print(f"  - Output: {llirl_dir}/{output_path}/")
    print("\nNext steps:")
    print(f"  1. Check clustering results: {model_path}/clustering_summary.json")
    print(f"  2. Check training summary: {model_path}/training_summary.json")
    print(f"  3. Compare with DDQN using compare_3_models_120p4k.py")
    print("=" * 80)


