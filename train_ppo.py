"""
Train PPO only
"""

import sys
import os
import subprocess

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ppo_dir = os.path.join(base_dir, "ppo_sumo")
    
    cmd = [
        sys.executable, "ppo_training.py",
        "--sumo_config", "../nets/single-intersection/run_morning_6to10.sumocfg",
        "--output", "output/sumo_single_intersection",
        "--model_path", "saves/sumo_single_intersection",
        "--batch_size", "8",
        "--hidden_size", "200",
        "--num_layers", "2",
        "--num_iter", "50",
        "--num_periods", "30",
        "--lr", "3e-4",
        "--algorithm", "ppo",
        "--opt", "adam",
        "--baseline", "linear",
        "--clip", "0.2",
        "--epochs", "5",
        "--tau", "1.0",
        "--device", "cuda",
        "--seed", "1009"
    ]
    
    subprocess.run(cmd, cwd=ppo_dir, check=True)

