"""
Train LLIRL với TẤT CẢ optimizations để đảm bảo vượt DDQN
Ultimate LLIRL training with all advanced improvements
"""

import sys
import os
import subprocess

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    llirl_dir = os.path.join(base_dir, "llirl_sumo")
    
    # Environment configuration
    sumo_config = "../nets/120p4k/run_120p4k.sumocfg"
    model_path = "saves/120p4k_ultimate"
    output_path = "output/120p4k_ultimate"
    
    print("=" * 80)
    print("LLIRL ULTIMATE TRAINING - ĐẢM BẢO VƯỢT DDQN")
    print("=" * 80)
    print(f"SUMO Config: {sumo_config}")
    print(f"Model Path: {model_path}")
    print(f"Output Path: {output_path}")
    print("=" * 80)
    
    # Step 1: Environment Clustering với parameters tối ưu nhất
    print("\n" + "=" * 80)
    print("Step 1: Environment Clustering (ULTIMATE)")
    print("=" * 80)
    print("Optimizations:")
    print("  - ZETA = 0.5 (dễ tạo cluster mới)")
    print("  - SIGMA = 0.1 (tăng độ nhạy)")
    print("  - TAU1 = 0.5, TAU2 = 0.5 (temperature thấp)")
    print("  - EM_STEPS = 5 (nhiều iterations)")
    print("  - env_hidden_size = 256 (capacity lớn)")
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
        "--env_num_layers", "3",
        "--env_hidden_size", "256",
        "--H", "4",
        "--max_steps", "7200",
        "--zeta", "0.5",
        "--sigma", "0.1",
        "--tau1", "0.5",
        "--tau2", "0.5",
        "--em_steps", "5"
    ]
    
    subprocess.run(clustering_cmd, cwd=llirl_dir, check=True)
    
    # Step 2: Policy Training với TẤT CẢ optimizations
    print("\n" + "=" * 80)
    print("Step 2: Policy Training (ULTIMATE with PPO)")
    print("=" * 80)
    print("Optimizations:")
    print("  - Algorithm: PPO (ổn định)")
    print("  - Learning rate: 1e-3 (ổn định)")
    print("  - num_iter: 150 (NHIỀU HƠN - từ 100 → 150)")
    print("  - batch_size: 16 (NHIỀU HƠN - từ 8 → 16)")
    print("  - Baseline: linear (giảm variance)")
    print("  - Gradient clipping: 0.5")
    print("  - Learning rate decay: 0.95")
    print("  - policy_eval_weight: 0.7 (ưu tiên performance)")
    print("  - num_test_episodes: 5 (đánh giá chính xác)")
    print("  - PPO clip: 0.2, epochs: 5")
    print("=" * 80)
    
    policy_cmd = [
        sys.executable, "policy_training.py",
        "--sumo_config", sumo_config,
        "--model_path", model_path,
        "--output", output_path,
        "--algorithm", "ppo",
        "--opt", "adam",
        "--lr", "1e-3",
        "--num_iter", "150",  # TĂNG từ 100 → 150
        "--num_periods", "5",
        "--device", "cuda",
        "--seed", "1009",
        "--batch_size", "16",  # TĂNG từ 8 → 16
        "--hidden_size", "200",
        "--num_layers", "2",
        "--use_general_policy",
        "--num_test_episodes", "5",
        "--policy_eval_weight", "0.7",
        "--max_steps", "7200",
        "--use_baseline",
        "--baseline", "linear",
        "--lr_decay", "0.95",
        "--lr_min", "1e-5",
        "--early_stop_patience", "10",
        "--early_stop_threshold", "0.01",
        "--grad_clip", "0.5",
        "--clip", "0.2",
        "--epochs", "5",
        "--tau", "1.0",
        "--ddqn_init_path", "ddqn_sumo/output/120p4k/ddqn_model_final.pth"  # Transfer learning từ DDQN
    ]
    
    subprocess.run(policy_cmd, cwd=llirl_dir, check=True)
    
    print("\n" + "=" * 80)
    print("LLIRL ULTIMATE TRAINING COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - Models: {llirl_dir}/{model_path}/")
    print(f"  - Output: {llirl_dir}/{output_path}/")
    print("\nKey Improvements:")
    print("  ✓ Tăng num_iter: 100 → 150 (+50% data)")
    print("  ✓ Tăng batch_size: 8 → 16 (+100% samples per iteration)")
    print("  ✓ PPO với tất cả optimizations")
    print("  ✓ Clustering parameters tối ưu")
    print("\nExpected Results:")
    print("  - Clustering: 2-3 clusters (thay vì 1)")
    print("  - Performance: -60,000 đến -80,000 (gần bằng hoặc tốt hơn DDQN)")
    print("  - Stability: Performance ổn định, không giảm dần")
    print("\nNext steps:")
    print(f"  1. Check clustering: {model_path}/clustering_summary.json")
    print(f"  2. Check training: {model_path}/training_summary.json")
    print(f"  3. Compare with DDQN:")
    print(f"     python compare_3_models_120p4k.py \\")
    print(f"         --sumo_config nets/120p4k/run_120p4k.sumocfg \\")
    print(f"         --ddqn_model_path ddqn_sumo/output/120p4k/ddqn_model_final.pth \\")
    print(f"         --llirl_model_path llirl_sumo/saves/120p4k_ultimate \\")
    print(f"         --output output/comparison_120p4k_ultimate \\")
    print(f"         --num_episodes 20")
    print("=" * 80)

