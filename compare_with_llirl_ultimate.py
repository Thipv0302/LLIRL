"""
Script để so sánh DDQN và PPO với LLIRL Ultimate (đã cải thiện)
Compare DDQN and PPO with optimized LLIRL Ultimate model
"""

import sys
import os
import gym
import numpy as np
import argparse
import torch
from tqdm import tqdm
import time
import json
import random
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ddqn_sumo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'llirl_sumo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ppo_sumo'))

# Register environments
from ddqn_sumo.myrllib.envs import sumo_env as ddqn_env
from llirl_sumo.myrllib.envs import sumo_env as llirl_env
from ppo_sumo.myrllib.envs import sumo_env as ppo_env
import gym

# Import traci for connection management
try:
    import traci
    import traci.exceptions
except ImportError:
    traci = None
    traci.exceptions = None

gym.register(
    'SUMO-SingleIntersection-DDQN-v1',
    entry_point='ddqn_sumo.myrllib.envs.sumo_env:SUMOEnv',
    max_episode_steps=3600
)

gym.register(
    'SUMO-SingleIntersection-LLIRL-v1',
    entry_point='llirl_sumo.myrllib.envs.sumo_env:SUMOEnv',
    max_episode_steps=3600
)

gym.register(
    'SUMO-SingleIntersection-PPO-v1',
    entry_point='ppo_sumo.myrllib.envs.sumo_env:SUMOEnv',
    max_episode_steps=3600
)

# Import models
from ddqn_sumo.myrllib.algorithms.ddqn import DDQN
from llirl_sumo.load_models import load_policies, load_crp_state, load_task_info
from llirl_sumo.myrllib.utils.policy_utils import (
    create_general_policy, 
    evaluate_policies, 
    evaluate_policy_performance
)
from llirl_sumo.myrllib.samplers.sampler import BatchSampler
from llirl_sumo.myrllib.policies import NormalMLPPolicy as LLIRL_NormalMLPPolicy
from ppo_sumo.myrllib.policies import NormalMLPPolicy as PPO_NormalMLPPolicy

start_time = time.time()

######################## Arguments ############################################
parser = argparse.ArgumentParser(description='Compare DDQN and PPO with LLIRL Ultimate')
parser.add_argument('--sumo_config', type=str, 
        default='nets/120p4k/run_120p4k.sumocfg',
        help='path to SUMO configuration file')
parser.add_argument('--ddqn_model_path', type=str, 
        default='ddqn_sumo/saves/120p4k/ddqn_model_final.pth',
        help='path to DDQN model')
parser.add_argument('--ddqn_output_path', type=str,
        default=None,
        help='path to DDQN output directory (for training rewards). If None, will auto-detect from model path')
parser.add_argument('--llirl_ultimate_path', type=str, 
        default='llirl_sumo/saves/120p4k_ultimate',
        help='path to LLIRL Ultimate model directory')
parser.add_argument('--llirl_ultimate_output_path', type=str,
        default=None,
        help='path to LLIRL Ultimate output directory (for training rewards). If None, will auto-detect from model path')
parser.add_argument('--ppo_model_path', type=str, 
        default='ppo_sumo/saves/120p4k/ppo_policy_final.pth',
        help='path to PPO model')
parser.add_argument('--ppo_output_path', type=str,
        default=None,
        help='path to PPO output directory (for training rewards). If None, will auto-detect from model path')
parser.add_argument('--ppo_saves_path', type=str,
        default=None,
        help='alternative path to PPO model in saves directory')
parser.add_argument('--output', type=str, default='output/comparison_with_llirl_ultimate',
        help='output folder for saving comparison results')
parser.add_argument('--num_episodes', type=int, default=20,
        help='number of test episodes per model')
parser.add_argument('--max_steps', type=int, default=3600,
        help='maximum steps per episode')
parser.add_argument('--device', type=str, default='cpu',
        help='device (cpu or cuda)')
parser.add_argument('--seed', type=int, default=42,
        help='random seed for reproducibility')
parser.add_argument('--llirl_policy_selection', type=str, default='best',
        choices=['cluster', 'performance', 'combined', 'general', 'best', 'first'],
        help='How to select LLIRL policy')
parser.add_argument('--policy_eval_weight', type=float, default=0.7,
        help='Weight for combined selection (optimized for ultimate)')
parser.add_argument('--num_test_episodes', type=int, default=5,
        help='Number of test episodes for policy evaluation')
args = parser.parse_args()

print("=" * 80)
print("MODEL COMPARISON: DDQN vs PPO vs LLIRL ULTIMATE")
print("=" * 80)
print(f"SUMO Config: {args.sumo_config}")
print(f"DDQN Model: {args.ddqn_model_path}")
print(f"LLIRL Ultimate: {args.llirl_ultimate_path}")
print(f"PPO Model: {args.ppo_model_path}")
print(f"Number of episodes: {args.num_episodes}")
print(f"Device: {args.device}")
print("=" * 80)

# Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
os.makedirs(args.output, exist_ok=True)

######################## Load Models ##########################################

print("\n" + "=" * 80)
print("LOADING MODELS")
print("=" * 80)

sumo_config_path = os.path.abspath(args.sumo_config)

# Load DDQN model
print("\n[1/3] Loading DDQN model...")
ddqn_env_test = gym.make('SUMO-SingleIntersection-DDQN-v1', 
                         sumo_config_path=sumo_config_path, 
                         max_steps=args.max_steps)

state_dim = ddqn_env_test.observation_space.shape[0]
action_dim = ddqn_env_test.action_space.n
print(f"State dim: {state_dim}, Action dim: {action_dim}")

ddqn = DDQN(
    state_dim=state_dim,
    action_dim=action_dim,
    device=device
)

if not os.path.exists(args.ddqn_model_path):
    raise FileNotFoundError(f"DDQN model not found at {args.ddqn_model_path}")

ddqn.load(args.ddqn_model_path)
ddqn.q_network.eval()
ddqn.target_network.eval()
print(f"✓ DDQN model loaded from {args.ddqn_model_path}")
ddqn_env_test.close()

# Load LLIRL Ultimate model
print("\n[2/3] Loading LLIRL Ultimate model...")
if not os.path.exists(args.llirl_ultimate_path):
    raise FileNotFoundError(f"LLIRL Ultimate model directory not found at {args.llirl_ultimate_path}")

llirl_policies, llirl_checkpoint = load_policies(args.llirl_ultimate_path, device=device)
print(f"✓ Loaded {len(llirl_policies)} LLIRL Ultimate policies")

# Load task information and cluster mapping
tasks, task_ids = load_task_info(args.llirl_ultimate_path)
crp = load_crp_state(args.llirl_ultimate_path)

if tasks is not None:
    print(f"✓ Loaded task info: {len(tasks)} periods, {len(np.unique(task_ids))} unique clusters")
    # Check clustering results
    clustering_summary_path = os.path.join(args.llirl_ultimate_path, 'clustering_summary.json')
    if os.path.exists(clustering_summary_path):
        import json
        with open(clustering_summary_path, 'r') as f:
            clustering_summary = json.load(f)
        print(f"✓ Clustering: {clustering_summary.get('num_clusters', 'N/A')} clusters created")
else:
    print("[WARNING] No task info found, will use default task")

# Auto-detect output paths if not provided
def auto_detect_output_path(model_path, folder_name):
    """Auto-detect output path from model path"""
    if 'saves' in model_path:
        return model_path.replace('saves', 'output')
    elif 'output' in model_path:
        return model_path.replace('output', 'output')  # Already output
    else:
        # Try common pattern
        model_dir_name = os.path.basename(model_path)
        if folder_name == 'ddqn_sumo':
            return os.path.join('ddqn_sumo', 'output', model_dir_name)
        elif folder_name == 'ppo_sumo':
            return os.path.join('ppo_sumo', 'output', model_dir_name)
        elif folder_name == 'llirl_sumo':
            return os.path.join('llirl_sumo', 'output', model_dir_name)
    return None

# DDQN output path
ddqn_output_path = args.ddqn_output_path
if ddqn_output_path is None:
    ddqn_output_path = auto_detect_output_path(args.ddqn_model_path, 'ddqn_sumo')
    if ddqn_output_path is None:
        ddqn_output_path = 'ddqn_sumo/output/120p4k'

# PPO output path
ppo_output_path = args.ppo_output_path
if ppo_output_path is None:
    ppo_output_path = auto_detect_output_path(args.ppo_model_path, 'ppo_sumo')
    if ppo_output_path is None:
        ppo_output_path = 'ppo_sumo/output/120p4k'

# LLIRL output path
llirl_output_path = args.llirl_ultimate_output_path
if llirl_output_path is None:
    llirl_output_path = auto_detect_output_path(args.llirl_ultimate_path, 'llirl_sumo')
    if llirl_output_path is None:
        llirl_output_path = 'llirl_sumo/output/120p4k_ultimate'

# Load training rewards from all three models
def load_training_rewards(output_path, rewards_filename, model_name):
    """Load training rewards from output directory"""
    if output_path and os.path.exists(output_path):
        rewards_file = os.path.join(output_path, rewards_filename)
        # Try .npy first, then .tmp.npy
        if not os.path.exists(rewards_file):
            rewards_file_tmp = rewards_file + '.tmp.npy'
            if os.path.exists(rewards_file_tmp):
                rewards_file = rewards_file_tmp
        
        if os.path.exists(rewards_file):
            try:
                rewards = np.load(rewards_file)
                print(f"✓ Loaded {model_name} training rewards from {rewards_file}")
                print(f"  Training rewards shape: {rewards.shape}")
                if len(rewards.shape) > 1:
                    print(f"  Training rewards: {rewards.shape[0]} periods, {rewards.shape[1]} {'episodes' if 'ddqn' in model_name.lower() else 'iterations'} per period")
                    avg_per_period = [np.mean(period_rewards) for period_rewards in rewards]
                    print(f"  Average training reward per period: {[f'{x:.2f}' for x in avg_per_period]}")
                else:
                    print(f"  Average training reward: {np.mean(rewards):.2f}")
                return rewards
            except Exception as e:
                print(f"[WARNING] Could not load {model_name} training rewards from {rewards_file}: {e}")
        else:
            print(f"[INFO] {model_name} training rewards not found at {rewards_file} (optional)")
    else:
        print(f"[INFO] {model_name} output directory not found at {output_path} (training rewards not available)")
    return None

ddqn_training_rewards = load_training_rewards(ddqn_output_path, 'rews_ddqn.npy', 'DDQN')
ppo_training_rewards = load_training_rewards(ppo_output_path, 'rews_ppo.npy', 'PPO')
llirl_training_rewards = load_training_rewards(llirl_output_path, 'rews_llirl.npy', 'LLIRL Ultimate')

# Load PPO model
print("\n[3/3] Loading PPO model...")
ppo_env_test = gym.make('SUMO-SingleIntersection-PPO-v1', 
                        sumo_config_path=sumo_config_path, 
                        max_steps=args.max_steps)

ppo_state_dim = int(np.prod(ppo_env_test.observation_space.shape))
ppo_action_dim = int(np.prod(ppo_env_test.action_space.shape))
print(f"PPO State dim: {ppo_state_dim}, Action dim: {ppo_action_dim}")

# Try to find PPO model
ppo_model_path = args.ppo_model_path
ppo_model_found = False

# Check if the specified path exists and is a valid PPO model
if os.path.exists(ppo_model_path):
    try:
        checkpoint = torch.load(ppo_model_path, map_location='cpu')
        # Check if it's a DDQN model (has 'q_network' key)
        if isinstance(checkpoint, dict) and 'q_network' in checkpoint:
            print(f"[WARNING] File {ppo_model_path} contains DDQN model, not PPO model!")
            ppo_model_path = None
        else:
            # It might be a PPO model, will verify later when loading
            ppo_model_found = True
            print(f"Found potential PPO model at: {ppo_model_path}")
    except Exception as e:
        print(f"[WARNING] Could not check file {ppo_model_path}: {e}")
        ppo_model_path = None

# If not found, try alternative locations
if not ppo_model_found:
    print("Trying alternative locations...")
    alt_paths = [
        args.ppo_saves_path if args.ppo_saves_path else None,
        'ppo_sumo/saves/120p4k/ppo_policy_final.pth',
        'ppo_sumo/saves/sumo_single_intersection/ppo_policy_final.pth'
    ]
    for alt_path in alt_paths:
        if alt_path and os.path.exists(alt_path):
            try:
                checkpoint = torch.load(alt_path, map_location='cpu')
                # Check if it's a DDQN model
                if isinstance(checkpoint, dict) and 'q_network' in checkpoint:
                    print(f"[WARNING] File {alt_path} contains DDQN model, skipping...")
                    continue
                # It's a valid PPO model
                ppo_model_path = alt_path
                ppo_model_found = True
                print(f"Found PPO model at: {ppo_model_path}")
                break
            except Exception as e:
                print(f"[WARNING] Could not check file {alt_path}: {e}")
                continue
    
    if not ppo_model_found:
        print("[WARNING] PPO model not found. Will skip PPO evaluation.")
        ppo_model_path = None

ppo_policy = None
if ppo_model_path and os.path.exists(ppo_model_path):
    try:
        # Load the checkpoint to check its structure
        checkpoint = torch.load(ppo_model_path, map_location=device)
        
        # Check if it's a DDQN model (has 'q_network' key) or PPO model (direct state_dict)
        if isinstance(checkpoint, dict) and 'q_network' in checkpoint:
            print(f"[ERROR] File {ppo_model_path} contains DDQN model, not PPO model!")
            print("[WARNING] PPO model not found. Skipping PPO evaluation.")
            ppo_model_path = None
        else:
            # It's a PPO model (direct state_dict or dict with policy keys)
            # Default architecture (same as training)
            hidden_size = 200
            num_layers = 2
            ppo_policy = PPO_NormalMLPPolicy(
                ppo_state_dim, 
                ppo_action_dim,
                hidden_sizes=(hidden_size,) * num_layers
            ).to(device)
            
            # If checkpoint is a dict with 'policy' key, extract it
            if isinstance(checkpoint, dict) and 'policy' in checkpoint:
                state_dict = checkpoint['policy']
            elif isinstance(checkpoint, dict) and any(k.startswith('layer') or k.startswith('mu') or k.startswith('sigma') for k in checkpoint.keys()):
                # Direct state_dict
                state_dict = checkpoint
            else:
                # Assume it's a direct state_dict
                state_dict = checkpoint
            
            ppo_policy.load_state_dict(state_dict)
            ppo_policy.eval()
            print(f"✓ PPO model loaded from {ppo_model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load PPO model from {ppo_model_path}: {e}")
        print("[WARNING] Skipping PPO evaluation.")
        ppo_model_path = None
        ppo_policy = None
else:
    print("[WARNING] PPO model not found. Skipping PPO evaluation.")

ppo_env_test.close()

# Function to select LLIRL policy
def select_llirl_policy(task, task_id, policies, crp, checkpoint, sampler, selection_method, 
                       policy_eval_weight=0.7, num_test_episodes=5, device='cpu'):
    """Select LLIRL policy according to algorithm"""
    state_dim = checkpoint.get('state_dim')
    action_dim = checkpoint.get('action_dim')
    hidden_size = checkpoint.get('hidden_size', 200)
    num_layers = checkpoint.get('num_layers', 2)
    
    if state_dim is None or action_dim is None:
        raise ValueError(f"Checkpoint missing required keys: state_dim={state_dim}, action_dim={action_dim}")
    
    # Cluster-based selection
    cluster_policy = None
    if task_id is not None and task_id <= len(policies):
        if policies[task_id - 1] is not None:
            cluster_policy = policies[task_id - 1]
    
    # Performance-based selection
    performance_policy = None
    performance_reward = None
    general_policy = None
    
    if selection_method in ['performance', 'combined', 'best', 'general']:
        valid_policies = [p for p in policies if p is not None]
        if len(valid_policies) > 0:
            if crp is not None and len(crp._prior) >= len(valid_policies):
                prior_slice = crp._prior[:len(valid_policies)]
                if isinstance(prior_slice, np.ndarray):
                    priors = prior_slice.tolist()
                else:
                    priors = list(prior_slice)
            else:
                priors = [1.0 / len(valid_policies)] * len(valid_policies)
            
            general_policy = create_general_policy(
                valid_policies, priors, state_dim, action_dim,
                (hidden_size,) * num_layers, device=device
            )
            
            all_policies_to_test = valid_policies.copy()
            if general_policy is not None and selection_method != 'general':
                all_policies_to_test = all_policies_to_test + [general_policy]
            
            if len(all_policies_to_test) > 0:
                best_policy, best_reward, policy_rewards = evaluate_policies(
                    all_policies_to_test, sampler, num_test_episodes, device
                )
                if best_policy is not None:
                    performance_policy = best_policy
                    performance_reward = best_reward
    
    # Select final policy
    if selection_method == 'cluster':
        selected_policy = cluster_policy if cluster_policy is not None else (policies[0] if len(policies) > 0 else None)
    elif selection_method == 'performance':
        selected_policy = performance_policy if performance_policy is not None else (policies[0] if len(policies) > 0 else None)
    elif selection_method == 'combined':
        if cluster_policy is not None and performance_policy is not None:
            cluster_reward, _ = evaluate_policy_performance(cluster_policy, sampler, num_test_episodes, device)
            combined_score_cluster = (1 - policy_eval_weight) * cluster_reward
            combined_score_perf = policy_eval_weight * performance_reward
            selected_policy = performance_policy if combined_score_perf > combined_score_cluster else cluster_policy
        elif performance_policy is not None:
            selected_policy = performance_policy
        elif cluster_policy is not None:
            selected_policy = cluster_policy
        else:
            selected_policy = policies[0] if len(policies) > 0 else None
    elif selection_method == 'general':
        selected_policy = general_policy if general_policy is not None else (policies[0] if len(policies) > 0 else None)
    elif selection_method == 'best':
        selected_policy = performance_policy if performance_policy is not None else (policies[0] if len(policies) > 0 else None)
    else:  # first
        selected_policy = policies[0] if len(policies) > 0 else None
    
    if selected_policy is None:
        raise ValueError("No policy available to select!")
    
    return selected_policy

######################## Evaluation Functions ##################################

def evaluate_ddqn(env, model, num_episodes, max_steps):
    """Evaluate DDQN model"""
    metrics = {
        'rewards': [],
        'episode_lengths': [],
        'waiting_times': [],
        'queue_lengths': [],
        'vehicle_counts': [],
        'speeds': []
    }
    
    print(f"\nEvaluating DDQN for {num_episodes} episodes...")
    for episode in tqdm(range(num_episodes), desc="DDQN episodes"):
        state = env.reset()
        episode_reward = 0
        episode_waiting = []
        episode_queues = []
        episode_vehicles = []
        episode_speeds = []
        
        for step in range(max_steps):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model.q_network(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            if 'waiting_time' in info:
                episode_waiting.append(info['waiting_time'])
            if 'queue_length' in info:
                episode_queues.append(info['queue_length'])
            if 'vehicle_count' in info:
                episode_vehicles.append(info['vehicle_count'])
            if 'speed' in info:
                episode_speeds.append(info['speed'])
            
            state = next_state
            
            if done:
                break
        
        metrics['rewards'].append(episode_reward)
        metrics['episode_lengths'].append(step + 1)
        if episode_waiting:
            metrics['waiting_times'].append(np.mean(episode_waiting))
        if episode_queues:
            metrics['queue_lengths'].append(np.mean(episode_queues))
        if episode_vehicles:
            metrics['vehicle_counts'].append(np.mean(episode_vehicles))
        if episode_speeds:
            metrics['speeds'].append(np.mean(episode_speeds))
    
    return metrics


def evaluate_llirl(env, policy, num_episodes, max_steps, device):
    """Evaluate LLIRL policy"""
    metrics = {
        'rewards': [],
        'episode_lengths': [],
        'waiting_times': [],
        'queue_lengths': [],
        'vehicle_counts': [],
        'speeds': []
    }
    
    print(f"\nEvaluating LLIRL Ultimate for {num_episodes} episodes...")
    for episode in tqdm(range(num_episodes), desc="LLIRL Ultimate episodes"):
        obs = env.reset()
        episode_reward = 0
        episode_waiting = []
        episode_queues = []
        episode_vehicles = []
        episode_speeds = []
        
        for step in range(max_steps):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().to(device)
                if len(obs_tensor.shape) == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                pi = policy(obs_tensor)
                action_tensor = pi.mean
                action = action_tensor.cpu().numpy()
                if len(action.shape) > 1:
                    action = action[0] if action.shape[0] == 1 else action.flatten()
                if len(action) != 4:
                    green_time_min = env.unwrapped.green_time_min if hasattr(env, 'unwrapped') else 10
                    if len(action) == 1:
                        action = np.repeat(action, 4)
                    else:
                        action = action[:4] if len(action) > 4 else np.pad(action, (0, 4 - len(action)), 'constant', constant_values=green_time_min)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            if 'waiting_time' in info:
                episode_waiting.append(info['waiting_time'])
            if 'queue_length' in info:
                episode_queues.append(info['queue_length'])
            if 'vehicle_count' in info:
                episode_vehicles.append(info['vehicle_count'])
            if 'speed' in info:
                episode_speeds.append(info['speed'])
            
            obs = next_obs
            
            if done:
                break
        
        metrics['rewards'].append(episode_reward)
        metrics['episode_lengths'].append(step + 1)
        if episode_waiting:
            metrics['waiting_times'].append(np.mean(episode_waiting))
        if episode_queues:
            metrics['queue_lengths'].append(np.mean(episode_queues))
        if episode_vehicles:
            metrics['vehicle_counts'].append(np.mean(episode_vehicles))
        if episode_speeds:
            metrics['speeds'].append(np.mean(episode_speeds))
    
    return metrics


def evaluate_ppo(env, policy, num_episodes, max_steps, device):
    """Evaluate PPO policy"""
    metrics = {
        'rewards': [],
        'episode_lengths': [],
        'waiting_times': [],
        'queue_lengths': [],
        'vehicle_counts': [],
        'speeds': []
    }
    
    print(f"\nEvaluating PPO for {num_episodes} episodes...")
    for episode in tqdm(range(num_episodes), desc="PPO episodes"):
        obs = env.reset()
        episode_reward = 0
        episode_waiting = []
        episode_queues = []
        episode_vehicles = []
        episode_speeds = []
        
        for step in range(max_steps):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().to(device)
                if len(obs_tensor.shape) == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                pi = policy(obs_tensor)
                action_tensor = pi.mean
                action = action_tensor.cpu().numpy()
                if len(action.shape) > 1:
                    action = action[0] if action.shape[0] == 1 else action.flatten()
                if len(action) != 4:
                    green_time_min = env.unwrapped.green_time_min if hasattr(env, 'unwrapped') else 10
                    if len(action) == 1:
                        action = np.repeat(action, 4)
                    else:
                        action = action[:4] if len(action) > 4 else np.pad(action, (0, 4 - len(action)), 'constant', constant_values=green_time_min)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            if 'waiting_time' in info:
                episode_waiting.append(info['waiting_time'])
            if 'queue_length' in info:
                episode_queues.append(info['queue_length'])
            if 'vehicle_count' in info:
                episode_vehicles.append(info['vehicle_count'])
            if 'speed' in info:
                episode_speeds.append(info['speed'])
            
            obs = next_obs
            
            if done:
                break
        
        metrics['rewards'].append(episode_reward)
        metrics['episode_lengths'].append(step + 1)
        if episode_waiting:
            metrics['waiting_times'].append(np.mean(episode_waiting))
        if episode_queues:
            metrics['queue_lengths'].append(np.mean(episode_queues))
        if episode_vehicles:
            metrics['vehicle_counts'].append(np.mean(episode_vehicles))
        if episode_speeds:
            metrics['speeds'].append(np.mean(episode_speeds))
    
    return metrics

######################## Run Comparison #######################################

print("\n" + "=" * 80)
print("RUNNING COMPARISON")
print("=" * 80)

# Determine tasks to test
test_tasks_list = []
if tasks is not None and len(tasks) > 0:
    # Use first task as default
    test_tasks_list = [(tasks[0], int(task_ids[0]) if task_ids is not None else None)]
    print(f"Testing with task from training: {test_tasks_list[0][0]}")
else:
    test_tasks_list = [(None, None)]
    print("No task info available, using default task")

# Create sampler for LLIRL policy selection
llirl_sampler = BatchSampler(
    'SUMO-SingleIntersection-LLIRL-v1',
    batch_size=1,
    num_workers=0,
    seed=args.seed,
    sumo_config_path=sumo_config_path
)

# Select LLIRL policy
task, task_id = test_tasks_list[0]
if task is not None:
    llirl_sampler.reset_task(task)

llirl_policy = select_llirl_policy(
    task, task_id, llirl_policies, crp, llirl_checkpoint,
    llirl_sampler, args.llirl_policy_selection,
    args.policy_eval_weight, args.num_test_episodes, device
)
llirl_policy.eval()
llirl_sampler._env.close()

# Evaluate DDQN
print("\n" + "=" * 80)
print("[1/3] Evaluating DDQN model...")
print("=" * 80)

ddqn_env = gym.make('SUMO-SingleIntersection-DDQN-v1', 
                    sumo_config_path=sumo_config_path, 
                    max_steps=args.max_steps)

if task is not None and hasattr(ddqn_env.unwrapped, 'reset_task'):
    ddqn_env.unwrapped.reset_task(task)

ddqn_metrics = evaluate_ddqn(ddqn_env, ddqn, args.num_episodes, args.max_steps)
ddqn_env.close()

# Close SUMO connection
if traci is not None:
    try:
        try:
            traci.getVersion()
            traci.close()
        except:
            pass
    except:
        pass
time.sleep(1.0)

# Evaluate LLIRL Ultimate
print("\n" + "=" * 80)
print("[2/3] Evaluating LLIRL Ultimate model...")
print("=" * 80)

llirl_env = gym.make('SUMO-SingleIntersection-LLIRL-v1', 
                     sumo_config_path=sumo_config_path, 
                     max_steps=args.max_steps)

if task is not None:
    llirl_env.unwrapped.reset_task(task)

llirl_metrics = evaluate_llirl(llirl_env, llirl_policy, args.num_episodes, args.max_steps, device)
llirl_env.close()

# Close SUMO connection
if traci is not None:
    try:
        try:
            traci.getVersion()
            traci.close()
        except:
            pass
    except:
        pass
time.sleep(1.0)

# Evaluate PPO
ppo_metrics = None
if ppo_policy is not None:
    print("\n" + "=" * 80)
    print("[3/3] Evaluating PPO model...")
    print("=" * 80)
    
    ppo_env = gym.make('SUMO-SingleIntersection-PPO-v1', 
                       sumo_config_path=sumo_config_path, 
                       max_steps=args.max_steps)
    
    if task is not None and hasattr(ppo_env.unwrapped, 'reset_task'):
        ppo_env.unwrapped.reset_task(task)
    
    ppo_metrics = evaluate_ppo(ppo_env, ppo_policy, args.num_episodes, args.max_steps, device)
    ppo_env.close()
    
    # Close SUMO connection
    if traci is not None:
        try:
            try:
                traci.getVersion()
                traci.close()
            except:
                pass
        except:
            pass
else:
    print("\n[3/3] Skipping PPO evaluation (model not found)")

# Final cleanup
if traci is not None:
    try:
        try:
            traci.getVersion()
            traci.close()
        except:
            pass
    except:
        pass

######################## Results Analysis ######################################

print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

def print_metrics(name, metrics):
    """Print metrics summary"""
    if metrics is None:
        print(f"\n{name}: Not evaluated (model not found)")
        return
    
    print(f"\n{name}:")
    print(f"  Average Reward: {np.mean(metrics['rewards']):.2f} ± {np.std(metrics['rewards']):.2f}")
    print(f"  Average Episode Length: {np.mean(metrics['episode_lengths']):.1f} ± {np.std(metrics['episode_lengths']):.1f}")
    if metrics['waiting_times']:
        print(f"  Average Waiting Time: {np.mean(metrics['waiting_times']):.2f} ± {np.std(metrics['waiting_times']):.2f} seconds")
    if metrics['queue_lengths']:
        print(f"  Average Queue Length: {np.mean(metrics['queue_lengths']):.2f} ± {np.std(metrics['queue_lengths']):.2f} vehicles")
    if metrics['vehicle_counts']:
        print(f"  Average Vehicle Count: {np.mean(metrics['vehicle_counts']):.2f} ± {np.std(metrics['vehicle_counts']):.2f}")
    if metrics['speeds']:
        print(f"  Average Speed: {np.mean(metrics['speeds']):.2f} ± {np.std(metrics['speeds']):.2f} m/s")

print_metrics("DDQN", ddqn_metrics)
print_metrics("LLIRL Ultimate", llirl_metrics)
print_metrics("PPO", ppo_metrics)

# Comparison
print("\n" + "-" * 80)
print("COMPARISON SUMMARY:")
print("-" * 80)

models = {
    'DDQN': ddqn_metrics,
    'LLIRL Ultimate': llirl_metrics,
    'PPO': ppo_metrics
}

# Compare rewards
valid_models = {k: v for k, v in models.items() if v is not None}
if len(valid_models) > 0:
    avg_rewards = {k: np.mean(v['rewards']) for k, v in valid_models.items()}
    best_model = max(avg_rewards, key=avg_rewards.get)
    worst_model = min(avg_rewards, key=avg_rewards.get)
    
    print(f"\n🏆 Best Model (by Average Reward): {best_model}")
    print(f"  Average Reward: {avg_rewards[best_model]:.2f}")
    
    print(f"\n📊 Ranking:")
    for model_name, reward in sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True):
        improvement = ((reward - avg_rewards[worst_model]) / abs(avg_rewards[worst_model])) * 100 if avg_rewards[worst_model] != 0 else 0
        vs_ddqn = ((reward - avg_rewards.get('DDQN', reward)) / abs(avg_rewards.get('DDQN', reward))) * 100 if avg_rewards.get('DDQN') else 0
        print(f"  {model_name}: {reward:.2f}")
        if model_name != 'DDQN' and 'DDQN' in avg_rewards:
            print(f"    → vs DDQN: {vs_ddqn:+.1f}%")

# Compare waiting times (lower is better for traffic congestion)
if all(m is not None and m['waiting_times'] for m in valid_models.values()):
    print(f"\n⏱️  Waiting Time Comparison (lower is better):")
    avg_waiting = {k: np.mean(v['waiting_times']) for k, v in valid_models.items()}
    best_waiting = min(avg_waiting, key=avg_waiting.get)
    worst_waiting = max(avg_waiting, key=avg_waiting.get)
    
    for model_name, waiting in sorted(avg_waiting.items(), key=lambda x: x[1]):
        improvement = ((avg_waiting[worst_waiting] - waiting) / avg_waiting[worst_waiting]) * 100 if avg_waiting[worst_waiting] != 0 else 0
        print(f"  {model_name}: {waiting:.2f}s ({improvement:+.1f}% improvement vs worst)")

# Compare queue lengths (lower is better)
if all(m is not None and m['queue_lengths'] for m in valid_models.values()):
    print(f"\n🚗 Queue Length Comparison (lower is better):")
    avg_queues = {k: np.mean(v['queue_lengths']) for k, v in valid_models.items()}
    best_queues = min(avg_queues, key=avg_queues.get)
    worst_queues = max(avg_queues, key=avg_queues.get)
    
    for model_name, queues in sorted(avg_queues.items(), key=lambda x: x[1]):
        improvement = ((avg_queues[worst_queues] - queues) / avg_queues[worst_queues]) * 100 if avg_queues[worst_queues] != 0 else 0
        print(f"  {model_name}: {queues:.2f} vehicles ({improvement:+.1f}% improvement vs worst)")

# Display training rewards comparison if available
print(f"\n📈 Training Progress Comparison:")
training_data = []
if ddqn_training_rewards is not None:
    if len(ddqn_training_rewards.shape) > 1:
        final_avg = np.mean([period_rewards[-1] for period_rewards in ddqn_training_rewards if len(period_rewards) > 0])
        overall_avg = np.mean(ddqn_training_rewards)
    else:
        final_avg = ddqn_training_rewards[-1] if len(ddqn_training_rewards) > 0 else 0
        overall_avg = np.mean(ddqn_training_rewards)
    training_data.append(('DDQN', final_avg, overall_avg, ddqn_training_rewards.shape))

if ppo_training_rewards is not None:
    if len(ppo_training_rewards.shape) > 1:
        final_avg = np.mean([period_rewards[-1] for period_rewards in ppo_training_rewards if len(period_rewards) > 0])
        overall_avg = np.mean(ppo_training_rewards)
    else:
        final_avg = ppo_training_rewards[-1] if len(ppo_training_rewards) > 0 else 0
        overall_avg = np.mean(ppo_training_rewards)
    training_data.append(('PPO', final_avg, overall_avg, ppo_training_rewards.shape))

if llirl_training_rewards is not None:
    if len(llirl_training_rewards.shape) > 1:
        final_avg = np.mean([period_rewards[-1] for period_rewards in llirl_training_rewards if len(period_rewards) > 0])
        overall_avg = np.mean(llirl_training_rewards)
    else:
        final_avg = llirl_training_rewards[-1] if len(llirl_training_rewards) > 0 else 0
        overall_avg = np.mean(llirl_training_rewards)
    training_data.append(('LLIRL Ultimate', final_avg, overall_avg, llirl_training_rewards.shape))

if training_data:
    print(f"\n  Final Training Rewards (last iteration/episode):")
    for model_name, final_avg, overall_avg, shape in training_data:
        print(f"    {model_name}: {final_avg:.2f} (shape: {shape})")
    
    print(f"\n  Overall Average Training Rewards:")
    for model_name, final_avg, overall_avg, shape in training_data:
        print(f"    {model_name}: {overall_avg:.2f}")
    
    # Compare with evaluation rewards
    print(f"\n  Training vs Evaluation Comparison:")
    eval_rewards = {}
    if ddqn_metrics:
        eval_rewards['DDQN'] = np.mean(ddqn_metrics['rewards'])
    if ppo_metrics:
        eval_rewards['PPO'] = np.mean(ppo_metrics['rewards'])
    if llirl_metrics:
        eval_rewards['LLIRL Ultimate'] = np.mean(llirl_metrics['rewards'])
    
    for model_name, final_avg, overall_avg, shape in training_data:
        if model_name in eval_rewards:
            eval_avg = eval_rewards[model_name]
            diff = eval_avg - final_avg
            pct_diff = (diff / abs(final_avg)) * 100 if final_avg != 0 else 0
            print(f"    {model_name}:")
            print(f"      Final Training: {final_avg:.2f}")
            print(f"      Evaluation: {eval_avg:.2f}")
            print(f"      Difference: {diff:+.2f} ({pct_diff:+.1f}%)")

# Save results
results = {
    'config': {
        'sumo_config': args.sumo_config,
        'num_episodes': args.num_episodes,
        'max_steps': args.max_steps,
        'seed': args.seed,
        'device': str(device),
        'llirl_policy_selection': args.llirl_policy_selection,
        'llirl_ultimate_path': args.llirl_ultimate_path,
    },
    'ddqn': {
        'model_path': args.ddqn_model_path,
        'output_path': ddqn_output_path if ddqn_output_path else None,
        'training_rewards_available': ddqn_training_rewards is not None,
        'rewards': ddqn_metrics['rewards'] if ddqn_metrics else None,
        'episode_lengths': ddqn_metrics['episode_lengths'] if ddqn_metrics else None,
        'waiting_times': ddqn_metrics['waiting_times'] if ddqn_metrics else None,
        'queue_lengths': ddqn_metrics['queue_lengths'] if ddqn_metrics else None,
        'vehicle_counts': ddqn_metrics['vehicle_counts'] if ddqn_metrics else None,
        'speeds': ddqn_metrics['speeds'] if ddqn_metrics else None,
        'avg_reward': float(np.mean(ddqn_metrics['rewards'])) if ddqn_metrics else None,
        'avg_waiting_time': float(np.mean(ddqn_metrics['waiting_times'])) if ddqn_metrics and ddqn_metrics['waiting_times'] else None,
        'avg_queue_length': float(np.mean(ddqn_metrics['queue_lengths'])) if ddqn_metrics and ddqn_metrics['queue_lengths'] else None,
        'training_rewards': ddqn_training_rewards.tolist() if ddqn_training_rewards is not None else None,
        'training_rewards_shape': list(ddqn_training_rewards.shape) if ddqn_training_rewards is not None else None,
    },
    'llirl_ultimate': {
        'model_path': args.llirl_ultimate_path,
        'output_path': llirl_output_path if llirl_output_path else None,
        'training_rewards_available': llirl_training_rewards is not None,
        'rewards': llirl_metrics['rewards'] if llirl_metrics else None,
        'episode_lengths': llirl_metrics['episode_lengths'] if llirl_metrics else None,
        'waiting_times': llirl_metrics['waiting_times'] if llirl_metrics else None,
        'queue_lengths': llirl_metrics['queue_lengths'] if llirl_metrics else None,
        'vehicle_counts': llirl_metrics['vehicle_counts'] if llirl_metrics else None,
        'speeds': llirl_metrics['speeds'] if llirl_metrics else None,
        'avg_reward': float(np.mean(llirl_metrics['rewards'])) if llirl_metrics else None,
        'avg_waiting_time': float(np.mean(llirl_metrics['waiting_times'])) if llirl_metrics and llirl_metrics['waiting_times'] else None,
        'avg_queue_length': float(np.mean(llirl_metrics['queue_lengths'])) if llirl_metrics and llirl_metrics['queue_lengths'] else None,
        'training_rewards': llirl_training_rewards.tolist() if llirl_training_rewards is not None else None,
        'training_rewards_shape': list(llirl_training_rewards.shape) if llirl_training_rewards is not None else None,
    },
    'ppo': {
        'model_path': ppo_model_path if ppo_policy is not None else None,
        'output_path': ppo_output_path if ppo_output_path else None,
        'training_rewards_available': ppo_training_rewards is not None,
        'rewards': ppo_metrics['rewards'] if ppo_metrics else None,
        'episode_lengths': ppo_metrics['episode_lengths'] if ppo_metrics else None,
        'waiting_times': ppo_metrics['waiting_times'] if ppo_metrics else None,
        'queue_lengths': ppo_metrics['queue_lengths'] if ppo_metrics else None,
        'vehicle_counts': ppo_metrics['vehicle_counts'] if ppo_metrics else None,
        'speeds': ppo_metrics['speeds'] if ppo_metrics else None,
        'avg_reward': float(np.mean(ppo_metrics['rewards'])) if ppo_metrics else None,
        'avg_waiting_time': float(np.mean(ppo_metrics['waiting_times'])) if ppo_metrics and ppo_metrics['waiting_times'] else None,
        'avg_queue_length': float(np.mean(ppo_metrics['queue_lengths'])) if ppo_metrics and ppo_metrics['queue_lengths'] else None,
        'training_rewards': ppo_training_rewards.tolist() if ppo_training_rewards is not None else None,
        'training_rewards_shape': list(ppo_training_rewards.shape) if ppo_training_rewards is not None else None,
    },
    'comparison': {
        'best_model_by_reward': best_model if len(valid_models) > 0 else None,
        'avg_rewards': {k: float(v) for k, v in avg_rewards.items()} if len(valid_models) > 0 else None,
    }
}

# Save to JSON
results_path = os.path.join(args.output, 'comparison_with_llirl_ultimate.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to {results_path}")

# Save reward arrays (evaluation)
if ddqn_metrics:
    np.save(os.path.join(args.output, 'ddqn_rewards.npy'), np.array(ddqn_metrics['rewards']))
if llirl_metrics:
    np.save(os.path.join(args.output, 'llirl_ultimate_rewards.npy'), np.array(llirl_metrics['rewards']))
if ppo_metrics:
    np.save(os.path.join(args.output, 'ppo_rewards.npy'), np.array(ppo_metrics['rewards']))

# Save training rewards if available
if ddqn_training_rewards is not None:
    np.save(os.path.join(args.output, 'ddqn_training_rewards.npy'), ddqn_training_rewards)
    print(f"✓ DDQN training rewards saved to {args.output}/ddqn_training_rewards.npy")
if ppo_training_rewards is not None:
    np.save(os.path.join(args.output, 'ppo_training_rewards.npy'), ppo_training_rewards)
    print(f"✓ PPO training rewards saved to {args.output}/ppo_training_rewards.npy")
if llirl_training_rewards is not None:
    np.save(os.path.join(args.output, 'llirl_ultimate_training_rewards.npy'), llirl_training_rewards)
    print(f"✓ LLIRL Ultimate training rewards saved to {args.output}/llirl_ultimate_training_rewards.npy")
print(f"✓ Reward arrays saved to {args.output}")

######################## Plot Comparison Charts ################################

print("\n" + "=" * 80)
print("GENERATING COMPARISON CHARTS")
print("=" * 80)

def plot_comparison_charts(results, output_dir):
    """Generate comprehensive comparison charts"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set style
    if sns is not None:
        sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 10
    
    fig = plt.figure(figsize=(20, 14))
    
    # 1. Evaluation Rewards Comparison (Bar Chart)
    ax1 = plt.subplot(3, 3, 1)
    models = []
    rewards = []
    errors = []
    
    if results['ddqn']['avg_reward'] is not None:
        models.append('DDQN')
        rewards.append(results['ddqn']['avg_reward'])
        if results['ddqn']['rewards']:
            errors.append(np.std(results['ddqn']['rewards']))
        else:
            errors.append(0)
    
    if results['llirl_ultimate']['avg_reward'] is not None:
        models.append('LLIRL\nUltimate')
        rewards.append(results['llirl_ultimate']['avg_reward'])
        if results['llirl_ultimate']['rewards']:
            errors.append(np.std(results['llirl_ultimate']['rewards']))
        else:
            errors.append(0)
    
    if results['ppo']['avg_reward'] is not None:
        models.append('PPO')
        rewards.append(results['ppo']['avg_reward'])
        if results['ppo']['rewards']:
            errors.append(np.std(results['ppo']['rewards']))
        else:
            errors.append(0)
    
    if models:
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = ax1.bar(models, rewards, yerr=errors, capsize=5, 
                       color=colors[:len(models)], alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
        ax1.set_title('Evaluation Rewards Comparison', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, reward in zip(bars, rewards):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{reward:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Evaluation Rewards Distribution (Box Plot)
    ax2 = plt.subplot(3, 3, 2)
    data_to_plot = []
    labels = []
    
    if results['ddqn']['rewards']:
        data_to_plot.append(results['ddqn']['rewards'])
        labels.append('DDQN')
    if results['llirl_ultimate']['rewards']:
        data_to_plot.append(results['llirl_ultimate']['rewards'])
        labels.append('LLIRL Ultimate')
    if results['ppo']['rewards']:
        data_to_plot.append(results['ppo']['rewards'])
        labels.append('PPO')
    
    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax2.set_ylabel('Reward', fontsize=12, fontweight='bold')
        ax2.set_title('Reward Distribution (Box Plot)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Waiting Time Comparison
    ax3 = plt.subplot(3, 3, 3)
    models_wt = []
    waiting_times = []
    errors_wt = []
    
    if results['ddqn']['avg_waiting_time'] is not None:
        models_wt.append('DDQN')
        waiting_times.append(results['ddqn']['avg_waiting_time'])
        if results['ddqn']['waiting_times']:
            errors_wt.append(np.std(results['ddqn']['waiting_times']))
        else:
            errors_wt.append(0)
    
    if results['llirl_ultimate']['avg_waiting_time'] is not None:
        models_wt.append('LLIRL\nUltimate')
        waiting_times.append(results['llirl_ultimate']['avg_waiting_time'])
        if results['llirl_ultimate']['waiting_times']:
            errors_wt.append(np.std(results['llirl_ultimate']['waiting_times']))
        else:
            errors_wt.append(0)
    
    if results['ppo']['avg_waiting_time'] is not None:
        models_wt.append('PPO')
        waiting_times.append(results['ppo']['avg_waiting_time'])
        if results['ppo']['waiting_times']:
            errors_wt.append(np.std(results['ppo']['waiting_times']))
        else:
            errors_wt.append(0)
    
    if models_wt:
        bars = ax3.bar(models_wt, waiting_times, yerr=errors_wt, capsize=5,
                      color=colors[:len(models_wt)], alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Average Waiting Time (seconds)', fontsize=12, fontweight='bold')
        ax3.set_title('Waiting Time Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, wt in zip(bars, waiting_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{wt:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # 4. Queue Length Comparison
    ax4 = plt.subplot(3, 3, 4)
    models_ql = []
    queue_lengths = []
    errors_ql = []
    
    if results['ddqn']['avg_queue_length'] is not None:
        models_ql.append('DDQN')
        queue_lengths.append(results['ddqn']['avg_queue_length'])
        if results['ddqn']['queue_lengths']:
            errors_ql.append(np.std(results['ddqn']['queue_lengths']))
        else:
            errors_ql.append(0)
    
    if results['llirl_ultimate']['avg_queue_length'] is not None:
        models_ql.append('LLIRL\nUltimate')
        queue_lengths.append(results['llirl_ultimate']['avg_queue_length'])
        if results['llirl_ultimate']['queue_lengths']:
            errors_ql.append(np.std(results['llirl_ultimate']['queue_lengths']))
        else:
            errors_ql.append(0)
    
    if results['ppo']['avg_queue_length'] is not None:
        models_ql.append('PPO')
        queue_lengths.append(results['ppo']['avg_queue_length'])
        if results['ppo']['queue_lengths']:
            errors_ql.append(np.std(results['ppo']['queue_lengths']))
        else:
            errors_ql.append(0)
    
    if models_ql:
        bars = ax4.bar(models_ql, queue_lengths, yerr=errors_ql, capsize=5,
                      color=colors[:len(models_ql)], alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Average Queue Length (vehicles)', fontsize=12, fontweight='bold')
        ax4.set_title('Queue Length Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, ql in zip(bars, queue_lengths):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ql:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Training Rewards Comparison (if available)
    ax5 = plt.subplot(3, 3, 5)
    has_training_data = False
    
    if results['ddqn']['training_rewards'] is not None:
        has_training_data = True
        ddqn_train = np.array(results['ddqn']['training_rewards'])
        if len(ddqn_train.shape) > 1:
            # Average across periods, then take final values
            final_rewards = [period_rewards[-1] if len(period_rewards) > 0 else 0 
                           for period_rewards in ddqn_train]
            periods = range(1, len(final_rewards) + 1)
            ax5.plot(periods, final_rewards, 'o-', label='DDQN', linewidth=2, markersize=6, color='#3498db')
    
    if results['llirl_ultimate']['training_rewards'] is not None:
        has_training_data = True
        llirl_train = np.array(results['llirl_ultimate']['training_rewards'])
        if len(llirl_train.shape) > 1:
            final_rewards = [period_rewards[-1] if len(period_rewards) > 0 else 0 
                           for period_rewards in llirl_train]
            periods = range(1, len(final_rewards) + 1)
            ax5.plot(periods, final_rewards, 's-', label='LLIRL Ultimate', linewidth=2, markersize=6, color='#2ecc71')
    
    if results['ppo']['training_rewards'] is not None:
        has_training_data = True
        ppo_train = np.array(results['ppo']['training_rewards'])
        if len(ppo_train.shape) > 1:
            final_rewards = [period_rewards[-1] if len(period_rewards) > 0 else 0 
                           for period_rewards in ppo_train]
            periods = range(1, len(final_rewards) + 1)
            ax5.plot(periods, final_rewards, '^-', label='PPO', linewidth=2, markersize=6, color='#e74c3c')
    
    if has_training_data:
        ax5.set_xlabel('Period', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Final Training Reward', fontsize=12, fontweight='bold')
        ax5.set_title('Training Progress Comparison', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Training data not available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Training Progress Comparison', fontsize=14, fontweight='bold')
    
    # 6. Episode Length Comparison
    ax6 = plt.subplot(3, 3, 6)
    models_el = []
    episode_lengths = []
    errors_el = []
    
    if results['ddqn']['episode_lengths']:
        models_el.append('DDQN')
        episode_lengths.append(np.mean(results['ddqn']['episode_lengths']))
        errors_el.append(np.std(results['ddqn']['episode_lengths']))
    
    if results['llirl_ultimate']['episode_lengths']:
        models_el.append('LLIRL\nUltimate')
        episode_lengths.append(np.mean(results['llirl_ultimate']['episode_lengths']))
        errors_el.append(np.std(results['llirl_ultimate']['episode_lengths']))
    
    if results['ppo']['episode_lengths']:
        models_el.append('PPO')
        episode_lengths.append(np.mean(results['ppo']['episode_lengths']))
        errors_el.append(np.std(results['ppo']['episode_lengths']))
    
    if models_el:
        bars = ax6.bar(models_el, episode_lengths, yerr=errors_el, capsize=5,
                      color=colors[:len(models_el)], alpha=0.8, edgecolor='black', linewidth=1.5)
        ax6.set_ylabel('Average Episode Length (steps)', fontsize=12, fontweight='bold')
        ax6.set_title('Episode Length Comparison', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        for bar, el in zip(bars, episode_lengths):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{el:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Speed Comparison
    ax7 = plt.subplot(3, 3, 7)
    models_sp = []
    speeds = []
    errors_sp = []
    
    if results['ddqn']['speeds']:
        models_sp.append('DDQN')
        speeds.append(np.mean(results['ddqn']['speeds']))
        errors_sp.append(np.std(results['ddqn']['speeds']))
    
    if results['llirl_ultimate']['speeds']:
        models_sp.append('LLIRL\nUltimate')
        speeds.append(np.mean(results['llirl_ultimate']['speeds']))
        errors_sp.append(np.std(results['llirl_ultimate']['speeds']))
    
    if results['ppo']['speeds']:
        models_sp.append('PPO')
        speeds.append(np.mean(results['ppo']['speeds']))
        errors_sp.append(np.std(results['ppo']['speeds']))
    
    if models_sp:
        bars = ax7.bar(models_sp, speeds, yerr=errors_sp, capsize=5,
                      color=colors[:len(models_sp)], alpha=0.8, edgecolor='black', linewidth=1.5)
        ax7.set_ylabel('Average Speed (m/s)', fontsize=12, fontweight='bold')
        ax7.set_title('Average Speed Comparison (Higher is Better)', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        for bar, sp in zip(bars, speeds):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{sp:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Vehicle Count Comparison
    ax8 = plt.subplot(3, 3, 8)
    models_vc = []
    vehicle_counts = []
    errors_vc = []
    
    if results['ddqn']['vehicle_counts']:
        models_vc.append('DDQN')
        vehicle_counts.append(np.mean(results['ddqn']['vehicle_counts']))
        errors_vc.append(np.std(results['ddqn']['vehicle_counts']))
    
    if results['llirl_ultimate']['vehicle_counts']:
        models_vc.append('LLIRL\nUltimate')
        vehicle_counts.append(np.mean(results['llirl_ultimate']['vehicle_counts']))
        errors_vc.append(np.std(results['llirl_ultimate']['vehicle_counts']))
    
    if results['ppo']['vehicle_counts']:
        models_vc.append('PPO')
        vehicle_counts.append(np.mean(results['ppo']['vehicle_counts']))
        errors_vc.append(np.std(results['ppo']['vehicle_counts']))
    
    if models_vc:
        bars = ax8.bar(models_vc, vehicle_counts, yerr=errors_vc, capsize=5,
                      color=colors[:len(models_vc)], alpha=0.8, edgecolor='black', linewidth=1.5)
        ax8.set_ylabel('Average Vehicle Count', fontsize=12, fontweight='bold')
        ax8.set_title('Vehicle Count Comparison', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        
        for bar, vc in zip(bars, vehicle_counts):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{vc:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 9. Overall Performance Radar Chart (simplified as bar chart)
    ax9 = plt.subplot(3, 3, 9)
    # Normalize metrics for comparison (inverse for lower-is-better metrics)
    metrics_data = {}
    
    if results['ddqn']['avg_reward'] is not None:
        metrics_data['DDQN'] = {
            'Reward': results['ddqn']['avg_reward'],
            'Waiting Time': -results['ddqn']['avg_waiting_time'] if results['ddqn']['avg_waiting_time'] else 0,
            'Queue Length': -results['ddqn']['avg_queue_length'] if results['ddqn']['avg_queue_length'] else 0,
            'Speed': results['ddqn']['speeds'][0] if results['ddqn']['speeds'] else 0
        }
    
    if results['llirl_ultimate']['avg_reward'] is not None:
        metrics_data['LLIRL Ultimate'] = {
            'Reward': results['llirl_ultimate']['avg_reward'],
            'Waiting Time': -results['llirl_ultimate']['avg_waiting_time'] if results['llirl_ultimate']['avg_waiting_time'] else 0,
            'Queue Length': -results['llirl_ultimate']['avg_queue_length'] if results['llirl_ultimate']['avg_queue_length'] else 0,
            'Speed': results['llirl_ultimate']['speeds'][0] if results['llirl_ultimate']['speeds'] else 0
        }
    
    if results['ppo']['avg_reward'] is not None:
        metrics_data['PPO'] = {
            'Reward': results['ppo']['avg_reward'],
            'Waiting Time': -results['ppo']['avg_waiting_time'] if results['ppo']['avg_waiting_time'] else 0,
            'Queue Length': -results['ppo']['avg_queue_length'] if results['ppo']['avg_queue_length'] else 0,
            'Speed': results['ppo']['speeds'][0] if results['ppo']['speeds'] else 0
        }
    
    if metrics_data:
        # Simple bar chart showing best model
        best_model = results['comparison']['best_model_by_reward']
        ax9.text(0.5, 0.7, f'Best Model:', ha='center', va='center', 
                transform=ax9.transAxes, fontsize=16, fontweight='bold')
        ax9.text(0.5, 0.5, best_model, ha='center', va='center', 
                transform=ax9.transAxes, fontsize=20, fontweight='bold', 
                color='#2ecc71' if best_model == 'LLIRL Ultimate' else '#3498db')
        ax9.text(0.5, 0.3, f'Reward: {results["comparison"]["avg_rewards"][best_model]:.0f}', 
                ha='center', va='center', transform=ax9.transAxes, fontsize=14)
        ax9.axis('off')
        ax9.set_title('Overall Winner', fontsize=14, fontweight='bold')
    
    plt.suptitle('Model Comparison: DDQN vs PPO vs LLIRL Ultimate', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    chart_path = os.path.join(output_dir, 'comparison_charts.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison charts saved to {chart_path}")
    plt.close()
    
    # Create a separate training progress chart if training data is available
    if (results['ddqn']['training_rewards'] is not None or 
        results['llirl_ultimate']['training_rewards'] is not None or 
        results['ppo']['training_rewards'] is not None):
        
        fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # DDQN training progress
        if results['ddqn']['training_rewards'] is not None:
            ddqn_train = np.array(results['ddqn']['training_rewards'])
            if len(ddqn_train.shape) > 1:
                for period_idx, period_rewards in enumerate(ddqn_train):
                    axes[0].plot(period_rewards, alpha=0.6, linewidth=1.5, 
                               label=f'Period {period_idx+1}' if period_idx < 5 else '')
                axes[0].set_xlabel('Episode', fontsize=12, fontweight='bold')
                axes[0].set_ylabel('Reward', fontsize=12, fontweight='bold')
                axes[0].set_title('DDQN Training Progress', fontsize=14, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                if len(ddqn_train) <= 5:
                    axes[0].legend(fontsize=8)
        
        # LLIRL training progress
        if results['llirl_ultimate']['training_rewards'] is not None:
            llirl_train = np.array(results['llirl_ultimate']['training_rewards'])
            if len(llirl_train.shape) > 1:
                for period_idx, period_rewards in enumerate(llirl_train):
                    axes[1].plot(period_rewards, alpha=0.6, linewidth=1.5,
                               label=f'Period {period_idx+1}' if period_idx < 5 else '')
                axes[1].set_xlabel('Iteration', fontsize=12, fontweight='bold')
                axes[1].set_ylabel('Reward', fontsize=12, fontweight='bold')
                axes[1].set_title('LLIRL Ultimate Training Progress', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                if len(llirl_train) <= 5:
                    axes[1].legend(fontsize=8)
        
        # PPO training progress
        if results['ppo']['training_rewards'] is not None:
            ppo_train = np.array(results['ppo']['training_rewards'])
            if len(ppo_train.shape) > 1:
                for period_idx, period_rewards in enumerate(ppo_train):
                    axes[2].plot(period_rewards, alpha=0.6, linewidth=1.5,
                               label=f'Period {period_idx+1}' if period_idx < 5 else '')
                axes[2].set_xlabel('Iteration', fontsize=12, fontweight='bold')
                axes[2].set_ylabel('Reward', fontsize=12, fontweight='bold')
                axes[2].set_title('PPO Training Progress', fontsize=14, fontweight='bold')
                axes[2].grid(True, alpha=0.3)
                if len(ppo_train) <= 5:
                    axes[2].legend(fontsize=8)
        
        plt.tight_layout()
        training_chart_path = os.path.join(output_dir, 'training_progress_charts.png')
        plt.savefig(training_chart_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training progress charts saved to {training_chart_path}")
        plt.close()

# Generate charts
plot_comparison_charts(results, args.output)

print("\n" + "=" * 80)
print(f"Comparison completed in {(time.time() - start_time) / 60:.2f} minutes")
print("=" * 80)

