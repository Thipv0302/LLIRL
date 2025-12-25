# ================================
# train_llirl.py (DDQN version)
# ================================
# This version calls:
#   1) env_clustering.py
#   2) policy_ddqn_training.py  (instead of PPO)
# -------------------------------

import sys
import os
import subprocess

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    llirl_dir = os.path.join(base_dir, "llirl_sumo")
    
    # Environment configuration
    sumo_config = "../nets/60p2k/run_60p2k.sumocfg"
    model_path = "saves/sumo_single_intersection"
    output_path = "output/sumo_single_intersection"
    
    print("=" * 80)
    print("LLIRL ULTIMATE TRAINING (DDQN VERSION)")
    print("=" * 80)
    print(f"SUMO Config: {sumo_config}")
    print(f"Model Path: {model_path}")
    print(f"Output Path: {output_path}")
    print("=" * 80)
    
    # ============================================================
    # STEP 1 — ENVIRONMENT CLUSTERING
    # ============================================================
    print("\n" + "=" * 80)
    print("Step 1: Environment Clustering (ULTIMATE)")
    print("=" * 80)

  
    # clustering_cmd = [
    #     sys.executable, "env_clustering.py",
    #     "--sumo_config", sumo_config,
    #     "--model_path", model_path,
    #     "--et_length", "1",
    #     "--num_periods", "50",
    #     "--device", "cuda",
    #     "--seed", "2025",
    #     "--batch_size", "1",
    #     "--env_num_layers", "3",
    #     "--env_hidden_size", "256",
    #     "--H", "4",
    #     "--max_steps", "4090",
    #     "--zeta", "1",
    #     "--sigma", "0.25",
    #     "--tau1", "0.5",
    #     "--tau2", "0.5",
    #     "--em_steps", "10"
    # ]
    # subprocess.run(clustering_cmd, cwd=llirl_dir, check=True)
    

    # ============================================================
    # STEP 2 — POLICY TRAINING (DDQN + LLIRL)
    # ============================================================
    print("\n" + "=" * 80)
    print("Step 2: Policy Training (ULTIMATE with DDQN + LLIRL)")
    print("=" * 80)

    # IMPORTANT: use your DDQN LLIRL training file name here
    ddqn_train_file = "policy_ddqn_training.py"

    policy_cmd = [
        sys.executable, ddqn_train_file,
        "--sumo_config", sumo_config,
        "--model_path", model_path,
        "--output", output_path,

        # LLIRL parameters
        "--num_periods", "40",
        "--max_steps", "4090",

        # Device + seed
        "--device", "cuda",
        "--seed", "2025",

        # DDQN hyperparameters
        "--lr", "1e-4",
        "--gamma", "0.95",
        "--batch_size", "32",
        "--replay_buffer_size", "20000",

        "--epsilon_start", "1.0",
        "--epsilon_end", "0.02",
        "--epsilon_decay", "0.96",

        "--hidden_sizes", "200", "200",

        # Transfer learning (init first policy)
        "--ddqn_init_path", r"../llirl_sumo/myrllib/transfer_model/ddqn_model_final.pth"
    ]

    subprocess.run(policy_cmd, cwd=llirl_dir, check=True)

    print("\n" + "=" * 80)
    print("LLIRL ULTIMATE TRAINING COMPLETED (DDQN VERSION)!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - Models: {llirl_dir}/{model_path}/")
    print(f"  - Output: {llirl_dir}/{output_path}/")
