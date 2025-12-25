"""
Train DDQN only
"""

import sys
import os
import subprocess

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ddqn_dir = os.path.join(base_dir, "ddqn_sumo")
    
    cmd = [
        sys.executable, "ddqn_training.py",
        "--sumo_config", "../nets/60p2k/run_60p2k.sumocfg",
        "--output", "output/sumo_single_intersection",
        "--model_path", "saves/sumo_single_intersection",
        "--num_periods", "15",
        "--num_episodes", "80",
        "--max_steps", "4090",
        "--lr", "0.001",
        "--gamma", "0.95",
        "--batch_size", "32",
        "--replay_buffer_size", "20000",
        "--hidden_sizes", "200", "200",
        "--device", "cuda",
        "--seed", "1009"
    ]
    
    subprocess.run(cmd, cwd=ddqn_dir, check=True)

