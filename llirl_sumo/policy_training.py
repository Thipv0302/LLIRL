"""
This the code for the paper:
[1] Zhi Wang, Chunlin Chen, and Daoyi Dong, "Lifelong Incremental Reinforcement Learning with 
Online Bayesian Inference", IEEE Trasactions on Neural Networks and Learning Systems, 2021.
https://github.com/HeyuanMingong/llinrl.git

This file is the implementation of the policy learning part of the proposed LLIRL algorithm
"""

### common lib
import sys
import os
import numpy as np
import argparse 
import torch
from tqdm import tqdm
import time 
from torch.optim import Adam, SGD 
import pickle
import random
import signal
import atexit
import gc
import json
import platform

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Register environment before importing
from myrllib.envs import sumo_env
import gym
gym.register(
    'SUMO-SingleIntersection-v1',
    entry_point='myrllib.envs.sumo_env:SUMOEnv',
    max_episode_steps=14400  # Thay đổi từ 3600 thành 14400 (4 giờ)
)

### personal lib
from myrllib.episodes.episode import BatchEpisodes 
from myrllib.samplers.sampler import BatchSampler 
from myrllib.policies import NormalMLPPolicy  
from myrllib.baselines.baseline import LinearFeatureBaseline
from myrllib.algorithms.reinforce import REINFORCE 
from myrllib.algorithms.trpo import TRPO 
from myrllib.algorithms.ppo import PPO
from myrllib.mixture.inference import CRP
from myrllib.utils.policy_utils import (
    create_general_policy, 
    evaluate_policy_performance,
    evaluate_policies
)


start_time = time.time()
######################## Arguments ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, 
        help='number of rollouts/learning episodes in one policy iteration')
parser.add_argument('--hidden_size', type=int, default=200,
        help='hidden size of the policy network')
parser.add_argument('--num_layers', type=int, default=2,
        help='number of hidden layers of the policy network')
parser.add_argument('--num_iter', type=int, default=50,
        help='number of policy iterations')
parser.add_argument('--lr', type=float, default=1e-3,
        help='learning rate, if REINFORCE algorithm is used')
parser.add_argument('--output', type=str, default='output/sumo_single_intersection',
        help='output folder for saving the experimental results')
parser.add_argument('--model_path', type=str, default='saves/sumo_single_intersection',
        help='the folder for saving and loading the pretrained model')
parser.add_argument('--sumo_config', type=str, 
        default='../nets/single-intersection/run_morning_6to10.sumocfg',
        help='path to SUMO configuration file')
parser.add_argument('--algorithm', type=str, default='reinforce',
        help='reinforce or trpo, the base algorithm for policy gradient')
parser.add_argument('--opt', type=str, default='sgd',
        help='sgd or adam, if using the reinforce algorithm')
parser.add_argument('--baseline', type=str, default=None,
        help='linear or None, baseline for policy gradient step')
parser.add_argument('--num_periods', type=int, default=30)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use_general_policy', action='store_true', default=True,
        help='Use general policy for evaluation and selection')
parser.add_argument('--num_test_episodes', type=int, default=3,
        help='Number of episodes to test policies')
parser.add_argument('--policy_eval_weight', type=float, default=0.5,
        help='Weight for policy evaluation vs cluster selection (0=cluster only, 1=eval only)')
parser.add_argument('--max_steps', type=int, default=7200,
        help='Maximum number of steps per episode')
parser.add_argument('--use_baseline', action='store_true', default=False,
        help='Whether to use a baseline for variance reduction')
parser.add_argument('--lr_decay', type=float, default=0.98,
        help='Learning rate decay factor per period')
parser.add_argument('--lr_min', type=float, default=1e-5,
        help='Minimum learning rate')
parser.add_argument('--early_stop_patience', type=int, default=10,
        help='Number of iterations to wait before early stopping')
parser.add_argument('--early_stop_threshold', type=float, default=0.01,
        help='Threshold for early stopping based on performance improvement')
parser.add_argument('--grad_clip', type=float, default=0.5,
        help='Gradient clipping value')
args = parser.parse_args()
print(args)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Print device information
print(f"\n{'='*60}")
print(f"Device Configuration:")
print(f"  Requested device: {args.device}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Actual device: {device}")
if torch.cuda.is_available():
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  [OK] Training will use GPU")
else:
    print(f"  [WARNING] CUDA not available - Training will use CPU")
print(f"{'='*60}\n")

np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)
np.set_printoptions(precision=3)

######################## Small functions ######################################
### build a learner given a policy network
def generate_learner(policy):
    if args.algorithm == 'trpo':
        learner = TRPO(policy, baseline=args.baseline, device=device)
    elif args.algorithm == 'ppo':
        learner = PPO(policy, baseline=args.baseline, lr=args.lr, opt=args.opt, device=device)
    else:
        learner = REINFORCE(policy, baseline=args.baseline, lr=args.lr, opt=args.opt, device=device)
    return learner 

### train a policy using a learner
def inner_train(policy, learner, period=0, track_metrics=True):
    rews = np.zeros(args.num_iter)
    iteration_metrics = {
        'rewards': [],
        'gradient_norms': [],
        'losses': [],
        'learning_rates': []
    }
    
    for idx in tqdm(range(args.num_iter)):
        episodes = sampler.sample(policy, device=device)
        reward = episodes.evaluate()
        rews[idx] = reward
        
        if track_metrics:
            iteration_metrics['rewards'].append(float(reward))
            
            # Track gradient norms before update
            if hasattr(learner, 'opt') and learner.opt is not None:
                total_norm = 0.0
                for p in policy.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                iteration_metrics['gradient_norms'].append(float(total_norm))
                
                # Track learning rate
                if hasattr(learner.opt, 'param_groups') and len(learner.opt.param_groups) > 0:
                    iteration_metrics['learning_rates'].append(float(learner.opt.param_groups[0]['lr']))
        
        learner.step(episodes, clip=True)
    
    return rews, iteration_metrics


######################## Main Functions #######################################
### build a sampler given an environment
env_name = 'SUMO-SingleIntersection-v1'
sumo_config_path = os.path.abspath(args.sumo_config)
# Use num_workers=0 on Windows to avoid multiprocessing issues
num_workers = 0 if platform.system() == 'Windows' else 1
sampler = BatchSampler(env_name, args.batch_size, num_workers=num_workers, seed=args.seed,
                      sumo_config_path=sumo_config_path, max_steps=args.max_steps) 
# Handle num_workers=0 case (single env)
if hasattr(sampler, 'envs') and sampler.envs is not None:
    state_dim = int(np.prod(sampler.envs.observation_space.shape))
    action_dim = int(np.prod(sampler.envs.action_space.shape))
else:
    state_dim = int(np.prod(sampler._env.observation_space.shape))
    action_dim = int(np.prod(sampler._env.action_space.shape))
print('state dim: %d; action dim: %d'%(state_dim,action_dim))

### get the task ids that are computed using env_clustering.py
task_info_path = os.path.join(args.model_path, 'task_info.npy')
if not os.path.exists(task_info_path):
    raise FileNotFoundError(
        f'Task info file not found at {task_info_path}. '
        f'Please run env_clustering.py first to generate task_info.npy'
    )
task_info = np.load(task_info_path)
tasks = task_info[:, :-1]
task_ids = task_info[:, -1]

# Validate task_info
if len(tasks) < args.num_periods:
    print(f'[WARNING] task_info has {len(tasks)} periods but num_periods={args.num_periods}')
    print(f'  -> Will use {len(tasks)} periods available')
    args.num_periods = len(tasks)

# Load CRP state để có priors cho general policy
crp_path = os.path.join(args.model_path, 'crp_state.pkl')
crp = None
if os.path.exists(crp_path):
    with open(crp_path, 'rb') as f:
        crp_data = pickle.load(f)
    crp = CRP(zeta=crp_data.get('zeta', 1.0))
    crp._L = crp_data.get('L', 1)
    crp._t = crp_data.get('t', 1)
    crp._prior = np.array(crp_data.get('prior', [0.5, 0.5]))
    print(f'Loaded CRP state: L={crp._L}, t={crp._t}, prior shape={crp._prior.shape}')
else:
    print(f'[WARNING] CRP state not found at {crp_path}. General policy will use uniform priors.')

print('====== Lifelong Incremental Reinforcement Learning (LLIRL) =======')

# Ensure output directories exist
os.makedirs(args.output, exist_ok=True)
os.makedirs(args.model_path, exist_ok=True)

# Global flag to track if we should save on exit
_save_on_exit = True
_completed_periods = 0

def save_intermediate_results():
    """Save intermediate results (called on exit or interrupt)"""
    global _save_on_exit, _completed_periods
    if not _save_on_exit:
        return
    
    try:
        print('\n' + '='*60)
        print('Saving intermediate results...')
        print('='*60)
        
        # Save current rewards
        if 'rews_llirl' in globals() and rews_llirl.sum() != 0:
            np.save(os.path.join(args.output, 'rews_llirl.npy'), rews_llirl)
            print(f'[OK] Saved rewards to {args.output}/rews_llirl.npy')
        
        # Save current policies if available
        if 'policies' in globals() and len(policies) > 0:
            try:
                intermediate_policies_path = os.path.join(args.model_path, 'policies_intermediate.pth')
                torch.save({
                    'policies': [policy.state_dict() for policy in policies],
                    'num_policies': len(policies),
                    'state_dim': state_dim,
                    'action_dim': action_dim,
                    'hidden_size': args.hidden_size,
                    'num_layers': args.num_layers,
                    'completed_periods': _completed_periods
                }, intermediate_policies_path)
                print(f'[OK] Saved intermediate policies to {intermediate_policies_path}')
            except Exception as e:
                print(f'[WARNING] Could not save intermediate policies: {e}')
        
        print('Intermediate results saved!')
    except Exception as e:
        print(f'[WARNING] Error saving intermediate results: {e}')

def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C)"""
    print('\n\n[INTERRUPT] Training interrupted by user (Ctrl+C)')
    save_intermediate_results()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(save_intermediate_results)

### at the initial time period, nominal model
print('The nominal task:', tasks[0]) 
sampler.reset_task(tasks[0])

### generate the nominal policy model at the first time period
policy_init = NormalMLPPolicy(state_dim, action_dim, 
        hidden_sizes=(args.hidden_size,) * args.num_layers)
learner_init = generate_learner(policy_init)

### record the performance 
rews_llirl = np.zeros((args.num_periods, args.num_iter))

# Track optimal parameters and performance for each period
optimal_period_data = {
    'periods': [],
    'optimal_rewards': [],
    'optimal_policy_ids': [],
    'cluster_ids': [],
    'task_params': [],
    'optimal_iterations': []
}

# Track detailed training metrics
training_metrics = {
    'periods': [],
    'iterations': [],
    'rewards': [],
    'rewards_mean': [],
    'rewards_std': [],
    'rewards_min': [],
    'rewards_max': [],
    'policy_gradients_norm': [],
    'losses': [],
    'learning_rates': [],
    'convergence': []
}

### training the nominal policy model 
print('Train the nominal model...')
rews_init, init_metrics = inner_train(policy_init, learner_init, period=0, track_metrics=True)

# Store initial period metrics
training_metrics['periods'].append(0)
training_metrics['iterations'].append(list(range(args.num_iter)))
training_metrics['rewards'].append(rews_init.tolist())
training_metrics['rewards_mean'].append(float(rews_init.mean()))
training_metrics['rewards_std'].append(float(rews_init.std()))
training_metrics['rewards_min'].append(float(rews_init.min()))
training_metrics['rewards_max'].append(float(rews_init.max()))
training_metrics['policy_gradients_norm'].append(init_metrics.get('gradient_norms', []))
training_metrics['learning_rates'].append(init_metrics.get('learning_rates', []))

### initialize the Dirichlet mixture model
policies = [policy_init]; learners = [learner_init]
num_policies = 1

rews_llirl[0] = rews_init

# Track optimal for period 0 (nominal) and save snapshot
optimal_reward_init = float(rews_init.max())
optimal_iter_init = int(rews_init.argmax())
optimal_period_data['periods'].append(1)
optimal_period_data['optimal_rewards'].append(optimal_reward_init)
optimal_period_data['optimal_policy_ids'].append(1)
optimal_period_data['cluster_ids'].append(1)
optimal_period_data['task_params'].append(tasks[0].tolist())
optimal_period_data['optimal_iterations'].append(optimal_iter_init)

# Save optimal policy snapshot for period 0
optimal_policy_path_0 = os.path.join(args.model_path, 'optimal_policy_period_1.pth')
torch.save({
    'policy': policy_init.state_dict(),
    'period': 1,
    'task_id': 1,
    'cluster_id': 1,
    'optimal_reward': optimal_reward_init,
    'optimal_iter': optimal_iter_init,
    'task_params': tasks[0].tolist(),
    'state_dim': state_dim,
    'action_dim': action_dim,
    'hidden_size': args.hidden_size,
    'num_layers': args.num_layers
}, optimal_policy_path_0)
print(f'Saved optimal policy snapshot for period 1')

### in the following time periods, dynamic environments
# Track policy selection method
policy_selection_history = {
    'periods': [],
    'cluster_based_policy': [],
    'performance_based_policy': [],
    'selected_policy': [],
    'selection_method': [],  # 'cluster', 'performance', 'combined'
    'cluster_reward': [],
    'performance_reward': [],
    'final_reward': []
}

# Wrap training loop in try-except to ensure results are saved even if crash
try:
    for period in range(1, args.num_periods):
        print('\n----------- Time period %d------'%(period+1))
        
        # Validate period index
        if period >= len(tasks):
            print(f'[WARNING] Period {period+1} exceeds available tasks ({len(tasks)}). Stopping.')
            break
            
        task = tasks[period]
        print('The task information:', task) 
        sampler.reset_task(task)

        task_id = int(task_ids[period])
        
        # Validate task_id
        if task_id < 1:
            raise ValueError(f'Invalid task_id: {task_id} (must be >= 1) at period {period+1}')
        if task_id > num_policies + 1:
            print(f'[WARNING] task_id {task_id} > num_policies + 1 ({num_policies + 1}) at period {period+1}')
            print(f'  -> Treating as new cluster, creating policy...')
            task_id = num_policies + 1
        
        # Step 1: Cluster-based selection (cách cũ)
        cluster_policy = None
        cluster_reward = None
        if task_id <= num_policies:
            cluster_policy = policies[task_id-1]
            print(f'Cluster-based selection: Policy {task_id}')
        elif task_id == num_policies + 1:
            print('New cluster detected, will create new policy')
        
        # Step 2: Performance-based selection (nếu enabled)
        performance_policy = None
        performance_reward = None
        selected_policy = None
        selection_method = 'cluster'
        general_policy = None
        
        if args.use_general_policy and num_policies > 0:
            print('\n--- Policy Evaluation Phase ---')
            
            # Tạo general policy từ weighted average
            # Normalize priors: prior[policy] / (tổng prior) * tham số của policy
            if crp is not None and len(crp._prior) >= num_policies:
                priors_raw = crp._prior[:num_policies]
                # Normalize: prior[i] / sum(priors)
                priors_sum = priors_raw.sum()
                if priors_sum > 0:
                    priors = (priors_raw / priors_sum).tolist()
                else:
                    priors = [1.0 / num_policies] * num_policies
            else:
                priors = [1.0 / num_policies] * num_policies
            
            general_policy = create_general_policy(
                policies, priors, state_dim, action_dim, 
                (args.hidden_size,) * args.num_layers,
                device=device
            )
            print('Created general policy from weighted average')
            
            # Evaluate tất cả policies (bao gồm general policy)
            all_policies_to_test = policies + [general_policy]
            best_policy, best_reward, policy_rewards = evaluate_policies(
                all_policies_to_test, sampler, args.num_test_episodes, device
            )
            
            if best_policy is not None:
                performance_policy = best_policy
                performance_reward = best_reward
                print(f'Performance-based selection: Best reward = {best_reward:.2f}')
                print(f'Policy rewards: {policy_rewards}')
            
            # Step 3: Kết hợp 2 cách chọn
            if cluster_policy is not None and performance_policy is not None:
                # Evaluate cluster policy
                cluster_reward, _ = evaluate_policy_performance(
                    cluster_policy, sampler, args.num_test_episodes, device
                )
                
                # Weighted combination
                combined_score_cluster = (1 - args.policy_eval_weight) * cluster_reward
                combined_score_perf = args.policy_eval_weight * performance_reward
                
                if combined_score_perf > combined_score_cluster:
                    selected_policy = performance_policy
                    selection_method = 'performance'
                    print(f'Selected: Performance-based (score: {combined_score_perf:.2f} > {combined_score_cluster:.2f})')
                else:
                    selected_policy = cluster_policy
                    selection_method = 'cluster'
                    print(f'Selected: Cluster-based (score: {combined_score_cluster:.2f} >= {combined_score_perf:.2f})')
            elif performance_policy is not None:
                selected_policy = performance_policy
                selection_method = 'performance'
                print('Selected: Performance-based (no cluster policy)')
            elif cluster_policy is not None:
                selected_policy = cluster_policy
                selection_method = 'cluster'
                print('Selected: Cluster-based (no performance evaluation)')
        
        # Step 4: Handle policy creation/selection
        if task_id == num_policies + 1:
            # Tạo policy mới cho cluster mới
            print('Generate a new policy for new cluster...')
            policy = NormalMLPPolicy(state_dim, action_dim, 
                    hidden_sizes=(args.hidden_size,) * args.num_layers)
            
            # Initialize từ best policy (nếu có performance evaluation)
            if performance_policy is not None and general_policy is not None and performance_policy != general_policy:
                # Initialize từ best performance policy (không phải general)
                policy.load_state_dict(performance_policy.state_dict())
                print('Initialized new policy from best performance policy')
            elif general_policy is not None:
                # Initialize từ general policy
                policy.load_state_dict(general_policy.state_dict())
                print('Initialized new policy from general policy')
            else:
                # Fallback: random existing policy
                index = np.random.choice(num_policies)
                policy.load_state_dict(policies[index].state_dict())
                print(f'Initialized new policy from random existing policy {index}')
            
            learner = generate_learner(policy)
            rews, period_metrics = inner_train(policy, learner, period=period, track_metrics=True)
            policies.append(policy)
            learners.append(learner)
            num_policies += 1
            
        elif task_id <= num_policies:
            # Lưu policy gốc của cluster để không mất lịch sử
            original_cluster_policy = policies[task_id-1]
            original_cluster_learner = learners[task_id-1]
            
            # Chọn policy dựa trên selection method
            policy_idx = None  # Index của policy được chọn trong policies list
            is_using_general_policy = False
            is_using_own_cluster_policy = False
            
            if selected_policy is not None and selection_method == 'performance':
                # Sử dụng performance-based policy
                # Tìm learner tương ứng
                for idx, p in enumerate(policies):
                    if p is selected_policy:
                        policy_idx = idx
                        break
                
                if policy_idx is not None:
                    # Policy từ library được chọn
                    is_using_own_cluster_policy = (policy_idx == task_id - 1)
                    
                    if is_using_own_cluster_policy:
                        # Dùng chính policy của cluster này → không cần clone
                        policy = selected_policy
                        learner = learners[policy_idx]
                        print(f'Using performance-selected policy (index {policy_idx}, cluster {policy_idx+1})')
                        print(f'  -> Selected policy belongs to current cluster {task_id}')
                    else:
                        # Policy từ cluster khác → cần clone để không ảnh hưởng policy gốc
                        policy = NormalMLPPolicy(state_dim, action_dim, 
                                hidden_sizes=(args.hidden_size,) * args.num_layers)
                        policy.load_state_dict(selected_policy.state_dict())
                        learner = generate_learner(policy)
                        print(f'Using performance-selected policy (index {policy_idx}, cluster {policy_idx+1})')
                        print(f'  -> Selected policy belongs to different cluster {policy_idx+1}')
                        print(f'  -> Cloned policy to avoid in-place updates')
                elif general_policy is not None and selected_policy == general_policy:
                    # Policy là general policy → tạo policy mới từ general policy
                    policy = NormalMLPPolicy(state_dim, action_dim, 
                            hidden_sizes=(args.hidden_size,) * args.num_layers)
                    policy.load_state_dict(general_policy.state_dict())
                    learner = generate_learner(policy)
                    is_using_general_policy = True
                    print('Using general policy, created new policy and learner')
                else:
                    # Fallback to cluster policy
                    policy = original_cluster_policy
                    learner = original_cluster_learner
                    is_using_own_cluster_policy = True
                    print(f'Fallback to cluster-based policy {task_id}')
            else:
                # Sử dụng cluster-based policy (default)
                policy = original_cluster_policy
                learner = original_cluster_learner
                is_using_own_cluster_policy = True
                print(f'Using cluster-based policy {task_id}')
            
            # Train policy (policy có thể là clone hoặc original)
            rews, period_metrics = inner_train(policy, learner, period=period, track_metrics=True)
            
            # Update policy trong library - Update đúng vị trí theo flowchart
            # Strategy theo flowchart: θ_{t+1}^(l*) ← θ_t* (update policy đã được chọn và train)
            if policy_idx is not None:
                # Case 1: Policy từ library được chọn → update vào cluster của policy đó
                policies[policy_idx] = policy
                learners[policy_idx] = learner
                print(f'[OK] Updated cluster {policy_idx+1} policy after training (l*={policy_idx+1})')
                
                # Nếu không phải cluster hiện tại, thông báo
                if policy_idx != task_id - 1:
                    print(f'  -> Cluster {task_id} policy remains unchanged')
                    
            elif is_using_general_policy:
                # Case 2: General policy được chọn → update vào cluster hiện tại
                # (theo flowchart, general policy được coi như policy của cluster hiện tại)
                policies[task_id-1] = policy
                learners[task_id-1] = learner
                print(f'[OK] Updated cluster {task_id} with trained general policy (l*={task_id})')
                
            elif is_using_own_cluster_policy:
                # Case 3: Cluster-based selection → update cluster hiện tại
                policies[task_id-1] = policy
                learners[task_id-1] = learner
                print(f'[OK] Updated cluster {task_id} policy after training (l*={task_id})')
                
            else:
                # Fallback: không nên xảy ra
                print(f'[WARNING] No policy update performed for cluster {task_id}')

        else:
            raise ValueError(f'Error on task id: {task_id}, num_policies: {num_policies}')
        
        # Track selection history
        policy_selection_history['periods'].append(period + 1)
        policy_selection_history['cluster_based_policy'].append(task_id if task_id <= num_policies else None)
        
        # Find performance policy index
        perf_policy_idx = None
        if selected_policy is not None:
            if general_policy is not None and selected_policy == general_policy:
                perf_policy_idx = 'general'
            else:
                for idx, p in enumerate(policies):
                    # Compare by checking if same object or same parameters
                    if p is selected_policy:
                        perf_policy_idx = idx + 1
                        break
        
        policy_selection_history['performance_based_policy'].append(perf_policy_idx)
        
        # Find selected policy index
        selected_policy_idx = None
        if selection_method == 'cluster':
            selected_policy_idx = task_id if task_id <= num_policies else None
        else:
            if general_policy is not None and selected_policy == general_policy:
                selected_policy_idx = 'general'
            else:
                for idx, p in enumerate(policies):
                    if p is selected_policy:
                        selected_policy_idx = idx + 1
                        break
        
        policy_selection_history['selected_policy'].append(selected_policy_idx)
        policy_selection_history['selection_method'].append(selection_method)
        policy_selection_history['cluster_reward'].append(float(cluster_reward) if cluster_reward is not None else None)
        policy_selection_history['performance_reward'].append(float(performance_reward) if performance_reward is not None else None)
        policy_selection_history['final_reward'].append(float(rews.mean()))
        rews_llirl[period] = rews

        # Track optimal performance for this period
        optimal_reward = rews.max()  # Best reward in this period
        optimal_iter = int(rews.argmax())  # Iteration with best reward
        optimal_period_data['periods'].append(period + 1)
        optimal_period_data['optimal_rewards'].append(float(optimal_reward))
        optimal_period_data['optimal_policy_ids'].append(task_id)
        optimal_period_data['cluster_ids'].append(int(task_ids[period]))
        optimal_period_data['task_params'].append(task.tolist())
        optimal_period_data['optimal_iterations'].append(optimal_iter)

        # Store detailed metrics for this period
        training_metrics['periods'].append(period + 1)
        training_metrics['iterations'].append(list(range(args.num_iter)))
        training_metrics['rewards'].append(rews.tolist())
        training_metrics['rewards_mean'].append(float(rews.mean()))
        training_metrics['rewards_std'].append(float(rews.std()))
        training_metrics['rewards_min'].append(float(rews.min()))
        training_metrics['rewards_max'].append(float(rews.max()))
        training_metrics['policy_gradients_norm'].append(period_metrics.get('gradient_norms', []))
        training_metrics['learning_rates'].append(period_metrics.get('learning_rates', []))
        
        # Compute convergence metric (improvement rate)
        if len(rews) > 10:
            early_reward = rews[:len(rews)//3].mean()
            late_reward = rews[-len(rews)//3:].mean()
            improvement = (late_reward - early_reward) / (abs(early_reward) + 1e-8)
            training_metrics['convergence'].append(float(improvement))
        else:
            training_metrics['convergence'].append(0.0)
        
        print('Average return: %.2f'%rews.mean())
        print('Best return: %.2f (at iteration %d)'%(optimal_reward, optimal_iter))
        print('Std return: %.2f'%rews.std())
        print('Improvement: %.2f%%'%(training_metrics['convergence'][-1] * 100))
        
        # Save rewards after each period (atomic write) - CRITICAL for recovery
        _completed_periods = period + 1
        try:
            # Atomic write: write to temp file then rename
            final_path = os.path.join(args.output, 'rews_llirl.npy')
            temp_path = os.path.join(args.output, 'rews_llirl.npy.tmp')
            # Ensure directory exists
            os.makedirs(args.output, exist_ok=True)
            # Save to temp file
            np.save(temp_path, rews_llirl)
            # On Windows, need to remove target first if it exists
            if os.path.exists(final_path):
                os.remove(final_path)
            # Rename temp to final
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)
            print(f'[OK] Saved rewards (period {period+1}/{args.num_periods})')
        except Exception as e:
            print(f'[WARNING] Could not save rewards: {e}')
            # Fallback to direct save
            try:
                os.makedirs(args.output, exist_ok=True)
                np.save(os.path.join(args.output, 'rews_llirl.npy'), rews_llirl)
            except Exception as e2:
                print(f'[WARNING] Fallback save also failed: {e2}')
        
        # Force garbage collection to free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save optimal policy snapshot for this period
        optimal_policy_path = os.path.join(args.model_path, f'optimal_policy_period_{period+1}.pth')
        torch.save({
            'policy': policy.state_dict(),
            'period': period + 1,
            'task_id': task_id,
            'cluster_id': int(task_ids[period]),
            'optimal_reward': float(optimal_reward),
            'optimal_iter': optimal_iter,
            'task_params': task.tolist(),
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers
        }, optimal_policy_path)
        
        # Save checkpoint every 10 periods AND after each period (for safety)
        if (period + 1) % 10 == 0:
            try:
                checkpoint_path = os.path.join(args.model_path, f'llirl_checkpoint_period_{period+1}.pth')
                temp_checkpoint_path = checkpoint_path + '.tmp'
                torch.save({
                    'policies': [policy.state_dict() for policy in policies],
                    'num_policies': num_policies,
                    'state_dim': state_dim,
                    'action_dim': action_dim,
                    'hidden_size': args.hidden_size,
                    'num_layers': args.num_layers,
                    'period': period + 1,
                    'task_ids': task_ids[:period+1],
                    'rews_llirl': rews_llirl[:period+1],
                    'training_metrics': {k: v[:period+1] if isinstance(v, list) else v for k, v in training_metrics.items()}
                }, temp_checkpoint_path)
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                os.rename(temp_checkpoint_path, checkpoint_path)
                print(f'[OK] Saved checkpoint to {checkpoint_path}')
            except Exception as e:
                print(f'[WARNING] Could not save checkpoint: {e}')
        
        # Also save lightweight checkpoint after EVERY period (for recovery)
        try:
            lightweight_checkpoint = os.path.join(args.model_path, 'llirl_checkpoint_latest.pth')
            temp_lightweight = lightweight_checkpoint + '.tmp'
            torch.save({
                'policies': [policy.state_dict() for policy in policies],
                'num_policies': num_policies,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'period': period + 1,
                'task_ids': task_ids[:period+1],
                'completed_periods': period + 1
            }, temp_lightweight)
            if os.path.exists(lightweight_checkpoint):
                os.remove(lightweight_checkpoint)
            os.rename(temp_lightweight, lightweight_checkpoint)
        except Exception as e:
            print(f'[WARNING] Could not save lightweight checkpoint: {e}')
        
        # Cleanup SUMO connections periodically to prevent memory leaks
        if (period + 1) % 5 == 0:
            try:
                if hasattr(sampler, '_env') and hasattr(sampler._env, 'sumo_running'):
                    if sampler._env.sumo_running:
                        try:
                            import traci
                            traci.close()
                            sampler._env.sumo_running = False
                        except:
                            pass
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

except Exception as e:
    print(f'\n[ERROR] Training crashed: {str(e)}')
    import traceback
    traceback.print_exc()
    print('\nAttempting to save partial results...')

finally:
    # Always save results, even if training crashed
    _save_on_exit = False  # Prevent duplicate saves
    
    print('\n' + '='*60)
    print('Saving results (partial or complete)...')
    print('='*60)
    
    # Determine how many periods were completed
    try:
        completed_periods = _completed_periods if _completed_periods > 0 else len([p for p in range(args.num_periods) if p < len(rews_llirl) and rews_llirl[p].sum() != 0])
        if completed_periods == 0:
            completed_periods = 1  # At least period 0 was done
        print(f'Completed periods: {completed_periods}/{args.num_periods}')
    except:
        completed_periods = 1
        print(f'Completed periods: at least 1/{args.num_periods}')
    
    # Ensure directories exist
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    
    # Cleanup SUMO connections before saving
    try:
        if 'sampler' in globals() and hasattr(sampler, '_env'):
            if hasattr(sampler._env, 'sumo_running') and sampler._env.sumo_running:
                try:
                    import traci
                    traci.close()
                    sampler._env.sumo_running = False
                except:
                    pass
    except:
        pass
    
    # Force garbage collection before saving
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save final policy library (atomic write)
    try:
        final_policies_path = os.path.join(args.model_path, 'policies_final.pth')
        temp_policies_path = final_policies_path + '.tmp'
        torch.save({
            'policies': [policy.state_dict() for policy in policies],
            'num_policies': num_policies,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'task_ids': task_ids,
            'completed_periods': completed_periods
        }, temp_policies_path)
        # Atomic rename
        if os.path.exists(final_policies_path):
            os.remove(final_policies_path)
        os.rename(temp_policies_path, final_policies_path)
        print(f'[OK] Saved {num_policies} policies to {final_policies_path}')
    except Exception as e:
        print(f'[WARNING] Error saving final policies: {e}')

    # Save learners state (if needed for resuming) - atomic write
    try:
        learners_path = os.path.join(args.model_path, 'learners_state.pth')
        temp_learners_path = learners_path + '.tmp'
        learners_state = {}
        for idx, learner in enumerate(learners):
            if hasattr(learner, 'opt'):
                learners_state[f'learner_{idx}_optimizer'] = learner.opt.state_dict()
            if hasattr(learner, 'baseline') and learner.baseline is not None:
                learners_state[f'learner_{idx}_baseline'] = learner.baseline.state_dict()
        torch.save(learners_state, temp_learners_path)
        if os.path.exists(learners_path):
            os.remove(learners_path)
        os.rename(temp_learners_path, learners_path)
        print(f'[OK] Saved learners state to {learners_path}')
    except Exception as e:
        print(f'[WARNING] Error saving learners state: {e}')

    # Save optimal period data - atomic write
    try:
        optimal_data_path = os.path.join(args.model_path, 'optimal_period_data.json')
        temp_optimal_path = optimal_data_path + '.tmp'
        with open(temp_optimal_path, 'w') as f:
            json.dump(optimal_period_data, f, indent=2)
        if os.path.exists(optimal_data_path):
            os.remove(optimal_data_path)
        os.rename(temp_optimal_path, optimal_data_path)
        print(f'[OK] Saved optimal period data to {optimal_data_path}')
    except Exception as e:
        print(f'[WARNING] Error saving optimal period data: {e}')

    # Save period-to-cluster mapping - atomic write
    try:
        period_cluster_mapping = {
            'periods': list(range(1, args.num_periods + 1)),
            'cluster_ids': [int(task_ids[i]) for i in range(args.num_periods)],
            'task_params': [tasks[i].tolist() for i in range(args.num_periods)],
            'num_clusters': num_policies
        }
        mapping_path = os.path.join(args.model_path, 'period_cluster_mapping.json')
        temp_mapping_path = mapping_path + '.tmp'
        with open(temp_mapping_path, 'w') as f:
            json.dump(period_cluster_mapping, f, indent=2)
        if os.path.exists(mapping_path):
            os.remove(mapping_path)
        os.rename(temp_mapping_path, mapping_path)
        print(f'[OK] Saved period-cluster mapping to {mapping_path}')
    except Exception as e:
        print(f'[WARNING] Error saving period-cluster mapping: {e}')

    # Save training summary - atomic write
    try:
        training_summary = {
    'total_periods': args.num_periods,
    'num_clusters': num_policies,
    'num_iterations_per_period': args.num_iter,
    'algorithm': args.algorithm,
    'learning_rate': args.lr,
    'optimizer': args.opt,
    'hidden_size': args.hidden_size,
    'num_layers': args.num_layers,
    'final_average_reward': float(rews_llirl[-1].mean()) if len(rews_llirl) > 0 and rews_llirl[-1].sum() != 0 else 0.0,
    'best_period_reward': float(rews_llirl.max()) if rews_llirl.sum() != 0 else 0.0,
    'best_period': int(rews_llirl.max(axis=1).argmax()) + 1 if rews_llirl.sum() != 0 else 1,
            'training_time_minutes': float((time.time() - start_time) / 60.0)
        }
        summary_path = os.path.join(args.model_path, 'training_summary.json')
        temp_summary_path = summary_path + '.tmp'
        with open(temp_summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        if os.path.exists(summary_path):
            os.remove(summary_path)
        os.rename(temp_summary_path, summary_path)
        print(f'[OK] Saved training summary to {summary_path}')
    except Exception as e:
        print(f'[WARNING] Error saving training summary: {e}')

    # Save detailed training metrics - atomic write
    try:
        metrics_path = os.path.join(args.model_path, 'training_metrics.json')
        temp_metrics_path = metrics_path + '.tmp'
        with open(temp_metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
        os.rename(temp_metrics_path, metrics_path)
        print(f'[OK] Saved detailed training metrics to {metrics_path}')
    except Exception as e:
        print(f'[WARNING] Error saving training metrics: {e}')

    # Save experiment configuration - atomic write
    try:
        experiment_config = {
    'algorithm': args.algorithm,
    'optimizer': args.opt,
    'learning_rate': args.lr,
    'batch_size': args.batch_size,
    'num_iterations': args.num_iter,
    'num_periods': args.num_periods,
    'hidden_size': args.hidden_size,
    'num_layers': args.num_layers,
    'baseline': args.baseline,
    'device': str(device),
    'seed': args.seed,
    'sumo_config': args.sumo_config,
    'model_path': args.model_path,
            'output_path': args.output
        }
        config_path = os.path.join(args.model_path, 'experiment_config.json')
        temp_config_path = config_path + '.tmp'
        with open(temp_config_path, 'w') as f:
            json.dump(experiment_config, f, indent=2)
        if os.path.exists(config_path):
            os.remove(config_path)
        os.rename(temp_config_path, config_path)
        print(f'[OK] Saved experiment configuration to {config_path}')
    except Exception as e:
        print(f'[WARNING] Error saving experiment config: {e}')

    # Compute and save performance statistics - atomic write
    try:
        # Check if we have data
        has_data = False
        try:
            has_data = rews_llirl.sum() != 0 and len(rews_llirl) > 0
        except:
            has_data = False

        performance_stats = {
            'overall_mean_reward': float(rews_llirl.mean()) if has_data else 0.0,
            'overall_std_reward': float(rews_llirl.std()) if has_data else 0.0,
            'overall_max_reward': float(rews_llirl.max()) if has_data else 0.0,
            'overall_min_reward': float(rews_llirl.min()) if has_data else 0.0,
            'final_period_mean': float(rews_llirl[-1].mean()) if len(rews_llirl) > 0 and rews_llirl[-1].sum() != 0 else 0.0,
            'best_period_mean': float(rews_llirl.mean(axis=1).max()) if rews_llirl.sum() != 0 else 0.0,
            'worst_period_mean': float(rews_llirl.mean(axis=1).min()) if rews_llirl.sum() != 0 else 0.0,
            'learning_trajectory': {
                'early_periods_mean': float(rews_llirl[:args.num_periods//3].mean()) if rews_llirl.sum() != 0 else 0.0,
                'middle_periods_mean': float(rews_llirl[args.num_periods//3:2*args.num_periods//3].mean()) if rews_llirl.sum() != 0 else 0.0,
                'late_periods_mean': float(rews_llirl[2*args.num_periods//3:].mean()) if rews_llirl.sum() != 0 else 0.0
            },
            'consistency': {
                'period_std_mean': float(rews_llirl.std(axis=1).mean()) if rews_llirl.sum() != 0 else 0.0,
                'iteration_std_mean': float(rews_llirl.std(axis=0).mean()) if rews_llirl.sum() != 0 else 0.0
            }
        }
        stats_path = os.path.join(args.model_path, 'performance_statistics.json')
        temp_stats_path = stats_path + '.tmp'
        with open(temp_stats_path, 'w') as f:
            json.dump(performance_stats, f, indent=2)
        if os.path.exists(stats_path):
            os.remove(stats_path)
        os.rename(temp_stats_path, stats_path)
        print(f'[OK] Saved performance statistics to {stats_path}')
    except Exception as e:
        print(f'[WARNING] Error saving performance statistics: {e}')

    # Save policy selection history - atomic write
    try:
        selection_history_path = os.path.join(args.model_path, 'policy_selection_history.json')
        temp_selection_path = selection_history_path + '.tmp'
        with open(temp_selection_path, 'w') as f:
            json.dump(policy_selection_history, f, indent=2)
        if os.path.exists(selection_history_path):
            os.remove(selection_history_path)
        os.rename(temp_selection_path, selection_history_path)
        print(f'[OK] Saved policy selection history to {selection_history_path}')
    except Exception as e:
        print(f'[WARNING] Error saving policy selection history: {e}')

    print('\nRunning time: %.2f min'%((time.time()-start_time)/60.0))
    print('='*60)
    print('Results saved successfully!')
    print('='*60)

