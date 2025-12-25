# ================================
# train_llirl_ddqn.py (NEW VERSION)
# ================================
# Pipeline:
#   1) Run env_clustering_apply.py to classify new environment tasks
#   2) Run policy_ddqn_apply.py to ONLINE-LEARN ONLY the chosen periods
# --------------------------------

import sys
import os
import subprocess

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    llirl_dir = os.path.join(base_dir, "llirl_sumo")

    # ============================
    # CONFIGURATION
    # ============================

    sumo_config = "../nets/demo/run_60p2k.sumocfg"

    # place to store CRP + env_models + tasks_info.json
    model_path = "saves/sumo_single_intersection"

    # place to store summary + xml of DDQN apply
    output_path = "../result_test/h"

    # choose which periods to apply (1-based indexing)
    # -1 means "last period"
    apply_periods = ["-1"]   # You can change e.g. ["28", "29", "30"]

    python_exe = sys.executable

    print("=" * 80)
    print(" LLIRL PIPELINE (DDQN APPLY VERSION) ")
    print("=" * 80)
    print(f"SUMO Config: {sumo_config}")
    print(f"Model Path:  {model_path}")
    print(f"Output Path: {output_path}")
    print("=" * 80)

    # =============================================================
    # STEP 1 — ENVIRONMENT CLUSTERING (RUN ONLY ONCE PER NEW ENV)
    # =============================================================
    print("\n" + "=" * 80)
    print(" STEP 1: Environment Clustering APPLY")
    print("=" * 80)

    clustering_cmd = [
        python_exe, "env_clustering_apply.py",
        "--sumo_config", sumo_config,
        "--model_path", model_path,
        "--device", "cuda",
        "--seed", "2025",
        "--batch_size", "1",
        "--env_num_layers", "3",
        "--env_hidden_size", "256",
        "--H", "4",
        "--max_steps", "8500",
        "--zeta", "1",
        "--sigma", "0.25",
        "--em_steps", "10",
        "--no_save_state",
    ]

    subprocess.run(clustering_cmd, cwd=llirl_dir, check=True)

    # =============================================================
    # STEP 2 — POLICY APPLY (ONLINE LEARNING ON SPECIFIC PERIODS)
    # =============================================================
    print("\n" + "=" * 80)
    print(" STEP 2: DDQN POLICY APPLY (ONLINE LEARNING ONLY ON SELECTED PERIODS)")
    print("=" * 80)

    # build apply_period args
    apply_period_args = []
    for p in apply_periods:
        apply_period_args += ["--apply_periods", p]

    policy_apply_cmd = [
        python_exe, "policy_ddqn_apply.py",
        "--sumo_config", sumo_config,
        "--model_path", model_path,
        "--output", output_path,
        "--device", "cuda",
        "--seed", "2025",
        "--num_episodes", "1",
        "--max_steps", "4090",
        "--lr", "1e-4",
        "--gamma", "0.95",
        "--batch_size", "32",
        "--replay_buffer_size", "20000",
        "--epsilon_start", "1.0",
        "--epsilon_end", "0.05",
        "--epsilon_decay", "0.995",
        "--hidden_sizes", "200", "200",
        "--use_general_policy",
        "--eval_use_gui",
        "--gui_delay_ms", "150",
        "--no_save_models",
    ] + apply_period_args

    subprocess.run(policy_apply_cmd, cwd=llirl_dir, check=True)

    print("\n" + "=" * 80)
    print(" LLIRL DDQN APPLY COMPLETED! ")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - Models: {llirl_dir}/{model_path}/")
    print(f"  - Outputs: {llirl_dir}/{output_path}/")
