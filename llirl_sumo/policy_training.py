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
import gym
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Register environment before importing
from myrllib.envs import sumo_env
import gym
gym.register(
    'SUMO-SingleIntersection-v1',
    entry_point='myrllib.envs.sumo_env:SUMOEnv',
    max_episode_steps=3600  # Default, can be overridden by max_steps parameter
)

### personal lib
from myrllib.episodes.episode import BatchEpisodes 
from myrllib.samplers.sampler import BatchSampler 
from myrllib.policies import NormalMLPPolicy, UniformPolicy  
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
parser.add_argument('--algorithm', type=str, default='ppo',
        help='reinforce, trpo, or ppo (ppo recommended for stability)')
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
parser.add_argument('--max_steps', type=int, default=3600,
        help='maximum steps per episode')
parser.add_argument('--lr_decay', type=float, default=0.95,
        help='Learning rate decay factor per period (1.0 = no decay)')
parser.add_argument('--lr_min', type=float, default=1e-5,
        help='Minimum learning rate')
parser.add_argument('--early_stop_patience', type=int, default=10,
        help='Early stopping patience (0 = disabled)')
parser.add_argument('--early_stop_threshold', type=float, default=0.01,
        help='Early stopping threshold (improvement < threshold)')
parser.add_argument('--use_baseline', action='store_true', default=False,
        help='Use linear baseline for variance reduction')
parser.add_argument('--grad_clip', type=float, default=0.5,
        help='Gradient clipping value (0 = disabled)')
parser.add_argument('--clip', type=float, default=0.2,
        help='PPO clip parameter (only for PPO algorithm)')
parser.add_argument('--epochs', type=int, default=5,
        help='PPO epochs per update (only for PPO algorithm)')
parser.add_argument('--tau', type=float, default=1.0,
        help='GAE tau parameter (only for PPO/REINFORCE with baseline)')
parser.add_argument('--ddqn_init_path', type=str, default=None,
        help='Path to DDQN model for policy initialization (transfer learning)')
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
    # Use baseline if enabled
    baseline_type = 'linear' if args.use_baseline else (args.baseline if args.baseline else None)
    
    if args.algorithm == 'trpo':
        learner = TRPO(policy, baseline=baseline_type, device=device)
    elif args.algorithm == 'ppo':
        # PPO with proper parameters
        learner = PPO(policy, baseline=baseline_type, lr=args.lr, opt=args.opt, 
                     clip=args.clip, epochs=args.epochs, tau=args.tau, device=device)
    else:
        learner = REINFORCE(policy, baseline=baseline_type, lr=args.lr, opt=args.opt, device=device)
    return learner 

### train a policy using a learner with optimizations
def inner_train(policy, learner, period=0, track_metrics=True, initial_lr=None):
    rews = np.zeros(args.num_iter)
    iteration_metrics = {
        'rewards': [],
        'gradient_norms': [],
        'losses': [],
        'learning_rates': []
    }
    
    # Early stopping variables
    best_reward = float('-inf')
    patience_counter = 0
    best_iter = 0
    
    # Learning rate scheduling
    if initial_lr is None and hasattr(learner, 'opt') and learner.opt is not None:
        if len(learner.opt.param_groups) > 0:
            initial_lr = learner.opt.param_groups[0]['lr']
    
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
                if len(learner.opt.param_groups) > 0:
                    iteration_metrics['learning_rates'].append(float(learner.opt.param_groups[0]['lr']))
        
        # Apply gradient clipping if enabled
        clip_value = args.grad_clip if args.grad_clip > 0 else None
        learner.step(episodes, clip=(clip_value is not None))
        # Additional gradient clipping after step if needed (some algorithms do it internally)
        if clip_value and hasattr(learner, 'opt') and learner.opt is not None:
            # Check if gradients exist and clip them
            if any(p.grad is not None for p in policy.parameters()):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_value)
        
        # Learning rate decay (exponential decay)
        if initial_lr and hasattr(learner, 'opt') and learner.opt is not None:
            if len(learner.opt.param_groups) > 0:
                current_lr = initial_lr * (args.lr_decay ** (idx / max(1, args.num_iter // 10)))
                current_lr = max(current_lr, args.lr_min)
                for param_group in learner.opt.param_groups:
                    param_group['lr'] = current_lr
        
        # Early stopping
        if args.early_stop_patience > 0:
            if reward > best_reward + args.early_stop_threshold:
                best_reward = reward
                patience_counter = 0
                best_iter = idx
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print(f'\n[Early Stop] No improvement for {args.early_stop_patience} iterations. '
                          f'Best reward: {best_reward:.4f} at iteration {best_iter}')
                    # Truncate arrays
                    rews = rews[:idx+1]
                    for key in iteration_metrics:
                        if len(iteration_metrics[key]) > idx+1:
                            iteration_metrics[key] = iteration_metrics[key][:idx+1]
                    break
    
    # Ensure rews has correct length (pad with last value if early stopped)
    # This ensures consistent array shape for rews_llirl
    if len(rews) < args.num_iter:
        # Pad with last value for consistency (replicates final performance)
        last_value = rews[-1] if len(rews) > 0 else 0.0
        rews = np.pad(rews, (0, args.num_iter - len(rews)), mode='constant', constant_values=last_value)
    
    return rews, iteration_metrics


######################## Main Functions #######################################
### build a sampler given an environment
env_name = 'SUMO-SingleIntersection-v1'
sumo_config_path = os.path.abspath(args.sumo_config)
# Use num_workers=0 on Windows to avoid multiprocessing issues
import platform
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
    raise FileNotFoundError(f'task_info.npy not found at {task_info_path}. Please run env_clustering.py first.')

task_info = np.load(task_info_path)
if task_info.shape[1] < 2:
    raise ValueError(f'Invalid task_info.npy format: expected at least 2 columns, got {task_info.shape[1]}')

tasks = task_info[:, :-1]
task_ids = task_info[:, -1]

# Validate task_ids
if len(tasks) != len(task_ids):
    raise ValueError(f'Task dimension mismatch: {len(tasks)} tasks but {len(task_ids)} task_ids')
if len(tasks) == 0:
    raise ValueError('No tasks found in task_info.npy')
if task_ids[0] != 1:
    raise ValueError(f'First task_id must be 1, got {task_ids[0]}')

# Validate num_periods consistency
num_periods_from_task_info = len(tasks)
if args.num_periods != num_periods_from_task_info:
    print(f'\n[WARNING] num_periods mismatch!')
    print(f'  Command line: {args.num_periods}')
    print(f'  From task_info.npy: {num_periods_from_task_info}')
    print(f'  Using {num_periods_from_task_info} from task_info.npy\n')
    args.num_periods = num_periods_from_task_info

# Load mixture library (env_models.pth) to get both env_models and policies
env_models_path = os.path.join(args.model_path, 'env_models.pth')
mixture_library = None
if os.path.exists(env_models_path):
    try:
        mixture_library = torch.load(env_models_path, map_location=device)
        if not isinstance(mixture_library, dict):
            raise ValueError(f'mixture_library must be a dict, got {type(mixture_library)}')
        if 'num_models' not in mixture_library:
            print('[WARNING] num_models not found in mixture_library, assuming 0')
            mixture_library['num_models'] = 0
        print(f'Loaded mixture library with {mixture_library.get("num_models", 0)} clusters')
        if 'policies' in mixture_library:
            if not isinstance(mixture_library['policies'], list):
                raise ValueError(f'policies must be a list, got {type(mixture_library["policies"])}')
            num_policies_in_library = sum(1 for p in mixture_library['policies'] if p is not None)
            print(f'Found {num_policies_in_library} policies in mixture library')
    except Exception as e:
        print(f'[WARNING] Error loading mixture library: {e}. Will create new mixture library.')
        mixture_library = None
else:
    print('[WARNING] env_models.pth not found. Will create new mixture library.')

# Load CRP state để có priors cho general policy
from myrllib.mixture.inference import CRP
import pickle
crp_path = os.path.join(args.model_path, 'crp_state.pkl')
crp = None
if os.path.exists(crp_path):
    try:
        with open(crp_path, 'rb') as f:
            crp_data = pickle.load(f)
        if not isinstance(crp_data, dict):
            raise ValueError(f'CRP data must be a dict, got {type(crp_data)}')
        crp = CRP(zeta=crp_data.get('zeta', 1.0))
        crp._L = int(crp_data.get('L', 1))
        crp._t = int(crp_data.get('t', 1))
        prior_data = crp_data.get('prior', [0.5, 0.5])
        if isinstance(prior_data, list):
            crp._prior = np.array(prior_data)
        else:
            crp._prior = np.array(prior_data)
        # Validate prior
        if len(crp._prior) < 1:
            raise ValueError(f'CRP prior must have at least 1 element, got {len(crp._prior)}')
        if not np.all(crp._prior >= 0):
            raise ValueError('CRP prior must be non-negative')
        print(f'Loaded CRP state: L={crp._L}, t={crp._t}, prior_len={len(crp._prior)}')
    except Exception as e:
        print(f'[WARNING] Error loading CRP state: {e}. Will use default CRP.')
        crp = None

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
        if 'rews_llirl' in globals() and len(rews_llirl) > 0 and not np.all(rews_llirl == 0):
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
# Try to initialize from DDQN if available (Transfer Learning)
if args.ddqn_init_path and os.path.exists(args.ddqn_init_path):
    print(f'\n[TRANSFER LEARNING] Initializing policy from DDQN: {args.ddqn_init_path}')
    try:
        from utils.ddqn_to_policy import convert_ddqn_to_policy
        policy_init, success = convert_ddqn_to_policy(
            args.ddqn_init_path, state_dim, action_dim,
            hidden_sizes=(args.hidden_size,) * args.num_layers,
            device=device
        )
        if success:
            print('✓ Policy initialized from DDQN (Transfer Learning)')
        else:
            print('[WARNING] DDQN conversion failed, using random initialization')
    except Exception as e:
        print(f'[WARNING] Failed to load DDQN for transfer learning: {e}')
        print('  Using random initialization instead')
policy_init = NormalMLPPolicy(state_dim, action_dim, 
        hidden_sizes=(args.hidden_size,) * args.num_layers)
else:
    policy_init = NormalMLPPolicy(state_dim, action_dim, 
            hidden_sizes=(args.hidden_size,) * args.num_layers)

learner_init = generate_learner(policy_init)

### record the performance 
# Initialize with correct shape, handling variable iteration counts due to early stopping
rews_llirl = np.zeros((args.num_periods, args.num_iter))
# Track actual iteration counts per period (for early stopping)
actual_iterations = [args.num_iter] * args.num_periods

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
# Get initial learning rate for scheduling
initial_lr = args.lr
if hasattr(learner_init, 'opt') and learner_init.opt is not None:
    if len(learner_init.opt.param_groups) > 0:
        initial_lr = learner_init.opt.param_groups[0]['lr']
rews_init, init_metrics = inner_train(policy_init, learner_init, period=0, track_metrics=True, initial_lr=initial_lr)

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
# Load policies from mixture library if available
policies = [policy_init]
learners = [learner_init]
num_policies = 1

# Try to load existing policies from mixture library
if mixture_library is not None and 'policies' in mixture_library:
    library_policies = mixture_library['policies']
    num_clusters = mixture_library.get('num_models', len(library_policies))
    
    # Load policies for existing clusters (skip first one as we already have policy_init)
    for cluster_idx in range(1, num_clusters):
        if cluster_idx < len(library_policies) and library_policies[cluster_idx] is not None:
            # Load policy from mixture library
            policy = NormalMLPPolicy(state_dim, action_dim, 
                    hidden_sizes=(args.hidden_size,) * args.num_layers)
            policy.load_state_dict(library_policies[cluster_idx])
            policies.append(policy)
            learners.append(generate_learner(policy))
            num_policies += 1
            print(f'Loaded policy for cluster {cluster_idx + 1} from mixture library')
        else:
            # No policy yet for this cluster, will create during training
            policies.append(None)
            learners.append(None)
            print(f'No policy found for cluster {cluster_idx + 1}, will create during training')
    
    # Ensure we have the right number of policies (match number of clusters)
    while len(policies) < num_clusters:
        policies.append(None)
        learners.append(None)
    
    if num_policies > 1:
        print(f'Loaded {num_policies - 1} additional policies from mixture library')

rews_llirl[0] = rews_init
_completed_periods = 1  # Period 0 is completed

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
        task = tasks[period]
        print('The task information:', task) 
        sampler.reset_task(task)

        task_id = int(task_ids[period])
        
        # Validate task_id
        if task_id < 1:
            raise ValueError(f'Invalid task_id: {task_id} (must be >= 1)')
        
        # Step 1: Cluster-based selection - Lấy policy từ mixture library
        cluster_policy = None
        cluster_reward = None
        if 1 <= task_id <= len(policies):
            # Lấy policy từ mixture library (policies list)
            if policies[task_id-1] is not None:
                cluster_policy = policies[task_id-1]
                print(f'Cluster-based selection: Policy {task_id} from mixture library')
            else:
                print(f'Cluster {task_id} exists but has no policy yet, will create new policy')
        elif task_id == len(policies) + 1:
            print('New cluster detected, will create new policy')
        
        # Step 2: Performance-based selection (nếu enabled)
        performance_policy = None
        performance_reward = None
        selected_policy = None
        selection_method = 'cluster'
        general_policy = None
        
        if args.use_general_policy and num_policies > 0:
            print('\n--- Policy Evaluation Phase ---')
            
            # Tạo general policy từ weighted average (chỉ dùng policies không None)
            valid_policies = [p for p in policies if p is not None]
            if len(valid_policies) > 0:
                if crp is not None and len(crp._prior) >= len(valid_policies):
                    priors = crp._prior[:len(valid_policies)].tolist()
                else:
                    priors = [1.0 / len(valid_policies)] * len(valid_policies)
                
                general_policy = create_general_policy(
                    valid_policies, priors, state_dim, action_dim, 
                    (args.hidden_size,) * args.num_layers,
                    device=device
                )
            else:
                print('[WARNING] No valid policies available for general policy creation')
                general_policy = None
            
            # Collect episodes và evaluate policies
            test_episodes_list = []
            valid_policies = [p for p in policies if p is not None]
            all_policies_to_test = valid_policies
            
            if general_policy is not None:
                print('Created general policy from weighted average')
                # Collect episodes với general policy để test
                print(f'Collecting {args.num_test_episodes} episodes with general policy...')
                for _ in range(args.num_test_episodes):
                    episodes = sampler.sample(general_policy, device=device)
                    test_episodes_list.append(episodes)
                all_policies_to_test = all_policies_to_test + [general_policy]
            else:
                print('[INFO] General policy is None, skipping general policy evaluation')
            
            # Evaluate tất cả policies (bao gồm general policy nếu có) - chỉ dùng policies không None
            if len(all_policies_to_test) > 0:
                best_policy, best_reward, policy_rewards = evaluate_policies(
                    all_policies_to_test, sampler, args.num_test_episodes, device
                )
            else:
                print('[WARNING] No policies to evaluate')
                best_policy = None
                best_reward = -np.inf
                policy_rewards = {}
            
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
                # Fallback: random existing policy (skip None policies)
                valid_policy_indices = [i for i, p in enumerate(policies) if p is not None]
                if len(valid_policy_indices) > 0:
                    index = np.random.choice(valid_policy_indices)
                    policy.load_state_dict(policies[index].state_dict())
                    print(f'Initialized new policy from random existing policy {index}')
                else:
                    print('No existing policies available, using random initialization')
            
            learner = generate_learner(policy)
            # Get initial learning rate for scheduling (with decay per period)
            period_lr = args.lr * (args.lr_decay ** period)
            period_lr = max(period_lr, args.lr_min)
            if hasattr(learner, 'opt') and learner.opt is not None:
                if len(learner.opt.param_groups) > 0:
                    learner.opt.param_groups[0]['lr'] = period_lr
            rews, period_metrics = inner_train(policy, learner, period=period, track_metrics=True, initial_lr=period_lr)
            policies.append(policy)
            learners.append(learner)
            num_policies += 1
            
            # Update mixture library with new policy
            if mixture_library is not None:
                # Ensure policies list is long enough
                while len(mixture_library['policies']) < len(policies):
                    mixture_library['policies'].append(None)
                # Validate task_id before accessing
                if not (1 <= task_id <= len(mixture_library['policies'])):
                    raise ValueError(f'Invalid task_id {task_id} for mixture_library access (len={len(mixture_library["policies"])})')
                mixture_library['policies'][task_id - 1] = policy.state_dict()
                mixture_library['num_models'] = len(policies)
                print(f'Updated mixture library: added policy for cluster {task_id}')
            
        elif task_id <= num_policies:
            # Validate task_id before accessing policies
            if not (1 <= task_id <= len(policies)):
                raise ValueError(f'Invalid task_id: {task_id} for policies access (len={len(policies)})')
            
            # Check if policy exists for this cluster
            if policies[task_id-1] is None:
                # Edge case: Cluster exists but policy not yet created - create new policy
                print(f'Cluster {task_id} exists but has no policy yet. Creating new policy...')
                policy = NormalMLPPolicy(state_dim, action_dim, 
                        hidden_sizes=(args.hidden_size,) * args.num_layers)
                
                # Initialize from best policy (similar to new cluster case)
                if performance_policy is not None and general_policy is not None and performance_policy != general_policy:
                    policy.load_state_dict(performance_policy.state_dict())
                    print('Initialized new policy from best performance policy')
                elif general_policy is not None:
                    policy.load_state_dict(general_policy.state_dict())
                    print('Initialized new policy from general policy')
                else:
                    # Fallback: random existing policy (skip None policies)
                    valid_policy_indices = [i for i, p in enumerate(policies) if p is not None]
                    if len(valid_policy_indices) > 0:
                        index = np.random.choice(valid_policy_indices)
                        policy.load_state_dict(policies[index].state_dict())
                        print(f'Initialized new policy from random existing policy {index}')
                    else:
                        print('No existing policies available, using random initialization')
                
                learner = generate_learner(policy)
                # Update policies list with newly created policy
                policies[task_id-1] = policy
                learners[task_id-1] = learner
            else:
                # Policy exists - use selection method
                # Chọn policy dựa trên selection method
                if selected_policy is not None and selection_method == 'performance':
                    # Sử dụng performance-based policy
                    policy = selected_policy
                    # Tìm learner tương ứng
                    policy_idx = None
                    for idx, p in enumerate(policies):
                        if p is selected_policy:
                            policy_idx = idx
                            break
                    
                    if policy_idx is not None:
                        learner = learners[policy_idx]
                        print(f'Using performance-selected policy (index {policy_idx})')
                    elif general_policy is not None and selected_policy == general_policy:
                        # Policy là general policy, tạo learner mới hoặc dùng cluster policy's learner
                        learner = generate_learner(policy)
                        print('Using general policy, created new learner')
                    else:
                        # Fallback to cluster policy
                        policy = policies[task_id-1]
                        learner = learners[task_id-1]
                        print(f'Fallback to cluster-based policy {task_id}')
                else:
                    # Sử dụng cluster-based policy (default)
                    policy = policies[task_id-1]
                    learner = learners[task_id-1]
                    print(f'Using cluster-based policy {task_id}')
            
            # Get initial learning rate for scheduling (with decay per period)
            period_lr = args.lr * (args.lr_decay ** period)
            period_lr = max(period_lr, args.lr_min)
            if hasattr(learner, 'opt') and learner.opt is not None:
                if len(learner.opt.param_groups) > 0:
                    learner.opt.param_groups[0]['lr'] = period_lr
            rews, period_metrics = inner_train(policy, learner, period=period, track_metrics=True, initial_lr=period_lr)
            
            # Update policy trong library (nếu policy đã thay đổi)
            if 1 <= task_id <= len(policies):
                policies[task_id-1] = policy
                learners[task_id-1] = learner
                
                # Update mixture library with trained policy
                if mixture_library is not None:
                    # Ensure policies list is long enough
                    while len(mixture_library['policies']) < len(policies):
                        mixture_library['policies'].append(None)
                    # Validate task_id before accessing
                    if not (1 <= task_id <= len(mixture_library['policies'])):
                        raise ValueError(f'Invalid task_id {task_id} for mixture_library access (len={len(mixture_library["policies"])})')
                    mixture_library['policies'][task_id - 1] = policy.state_dict()
                    print(f'Updated mixture library: updated policy for cluster {task_id}')

        else:
            raise ValueError(f'Invalid task_id: {task_id}, num_policies: {num_policies}. Task ID must be between 1 and {num_policies + 1}')
        
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
        policy_selection_history['final_reward'].append(float(rews.mean()) if 'rews' in locals() else None)
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
        import gc
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
                'task_ids': task_ids[:period+1] if len(task_ids) > period + 1 else task_ids,
                'completed_periods': period + 1
            }, temp_lightweight)
            if os.path.exists(lightweight_checkpoint):
                os.remove(lightweight_checkpoint)
            os.rename(temp_lightweight, lightweight_checkpoint)
        except Exception as e:
            print(f'[WARNING] Could not save lightweight checkpoint: {e}')
        
        # Save mixture library (env_models + policies) after EVERY period
        if mixture_library is not None:
            try:
                # Ensure policies list matches number of clusters
                while len(mixture_library['policies']) < len(policies):
                    mixture_library['policies'].append(None)
                # Update all policies that exist
                for idx, policy_obj in enumerate(policies):
                    if policy_obj is not None:
                        mixture_library['policies'][idx] = policy_obj.state_dict()
                mixture_library['num_models'] = len(policies)
                mixture_library['state_dim'] = state_dim
                mixture_library['action_dim'] = action_dim
                
                # Atomic save
                temp_mixture_path = env_models_path + '.tmp'
                torch.save(mixture_library, temp_mixture_path)
                if os.path.exists(env_models_path):
                    os.remove(env_models_path)
                os.rename(temp_mixture_path, env_models_path)
                print(f'[OK] Updated mixture library with {sum(1 for p in mixture_library["policies"] if p is not None)} policies')
            except Exception as e:
                print(f'[WARNING] Could not save mixture library: {e}')
        
        # Cleanup SUMO connections periodically to prevent memory leaks
        if (period + 1) % 5 == 0:
            try:
                # Close SUMO connection if exists
                if hasattr(sampler, '_env') and hasattr(sampler._env, 'sumo_running'):
                    if sampler._env.sumo_running:
                        try:
                            import traci
                            traci.close()
                            sampler._env.sumo_running = False
                        except Exception as e:
                            print(f'[WARNING] Error closing SUMO: {e}')
                
                # Also check sampler.envs for multiprocessing case
                if hasattr(sampler, 'envs') and sampler.envs is not None:
                    for env in sampler.envs:
                        if hasattr(env, 'sumo_running') and env.sumo_running:
                            try:
                                import traci
                                traci.close()
                                env.sumo_running = False
                            except:
                                pass
                
                # Force garbage collection
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f'[WARNING] Error during cleanup: {e}')

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
        if _completed_periods > 0:
            completed_periods = _completed_periods
        else:
            # Count periods with valid data (not all zeros)
            completed_periods = sum(1 for p in range(args.num_periods) 
                                  if p < len(rews_llirl) and 
                                  len(rews_llirl[p]) > 0 and 
                                  not np.all(rews_llirl[p] == 0))
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
        # Close single environment
        if 'sampler' in globals() and hasattr(sampler, '_env'):
            if hasattr(sampler._env, 'sumo_running') and sampler._env.sumo_running:
                try:
                    sampler._env.close()  # Use close() method which handles cleanup
                except Exception as e:
                    print(f'[WARNING] Error closing sampler._env: {e}')
                    try:
                        import traci
                        traci.close()
                        sampler._env.sumo_running = False
                    except:
                        pass
        
        # Close multiprocessing environments
        if 'sampler' in globals() and hasattr(sampler, 'envs') and sampler.envs is not None:
            for env in sampler.envs:
                if hasattr(env, 'sumo_running') and env.sumo_running:
                    try:
                        env.close()
                    except:
                        try:
                            import traci
                            traci.close()
                            env.sumo_running = False
                        except:
                            pass
    except Exception as e:
        print(f'[WARNING] Error during final SUMO cleanup: {e}')
    
    # Force garbage collection before saving
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save final mixture library (env_models + policies) - atomic write
    if 'mixture_library' in globals() and mixture_library is not None:
        try:
            # Ensure policies list matches number of clusters
            while len(mixture_library['policies']) < len(policies):
                mixture_library['policies'].append(None)
            # Update all policies that exist
            for idx, policy_obj in enumerate(policies):
                if policy_obj is not None:
                    mixture_library['policies'][idx] = policy_obj.state_dict()
            mixture_library['num_models'] = len(policies)
            mixture_library['state_dim'] = state_dim
            mixture_library['action_dim'] = action_dim
            
            # Atomic save
            temp_mixture_path = env_models_path + '.tmp'
            torch.save(mixture_library, temp_mixture_path)
            if os.path.exists(env_models_path):
                os.remove(env_models_path)
            os.rename(temp_mixture_path, env_models_path)
            num_policies_in_library = sum(1 for p in mixture_library['policies'] if p is not None)
            print(f'[OK] Saved mixture library with {num_policies_in_library} policies to {env_models_path}')
        except Exception as e:
            print(f'[WARNING] Error saving final mixture library: {e}')
    
    # Save final policy library (atomic write) - for backward compatibility
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
        import json
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
        # Check if we have valid reward data
        has_reward_data = (len(rews_llirl) > 0 and 
                          not np.all(rews_llirl == 0) and 
                          np.any(np.isfinite(rews_llirl)))
        
        training_summary = {
    'total_periods': args.num_periods,
    'num_clusters': num_policies,
    'num_iterations_per_period': args.num_iter,
    'algorithm': args.algorithm,
    'learning_rate': args.lr,
    'optimizer': args.opt,
    'hidden_size': args.hidden_size,
    'num_layers': args.num_layers,
    'final_average_reward': float(rews_llirl[-1].mean()) if has_reward_data and len(rews_llirl) > 0 and len(rews_llirl[-1]) > 0 else 0.0,
    'best_period_reward': float(rews_llirl.max()) if has_reward_data else 0.0,
    'best_period': int(rews_llirl.max(axis=1).argmax()) + 1 if has_reward_data else 1,
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
        # Check if we have valid reward data (more robust than sum() != 0)
        has_data = False
        try:
            has_data = (len(rews_llirl) > 0 and 
                       not np.all(rews_llirl == 0) and 
                       np.any(np.isfinite(rews_llirl)))
        except:
            has_data = False

        performance_stats = {
            'overall_mean_reward': float(rews_llirl.mean()) if has_data else 0.0,
            'overall_std_reward': float(rews_llirl.std()) if has_data else 0.0,
            'overall_max_reward': float(rews_llirl.max()) if has_data else 0.0,
            'overall_min_reward': float(rews_llirl.min()) if has_data else 0.0,
            'final_period_mean': float(rews_llirl[-1].mean()) if has_data and len(rews_llirl) > 0 and len(rews_llirl[-1]) > 0 else 0.0,
            'best_period_mean': float(rews_llirl.mean(axis=1).max()) if has_data else 0.0,
            'worst_period_mean': float(rews_llirl.mean(axis=1).min()) if has_data else 0.0,
            'learning_trajectory': {
                'early_periods_mean': float(rews_llirl[:args.num_periods//3].mean()) if has_data and args.num_periods >= 3 else 0.0,
                'middle_periods_mean': float(rews_llirl[args.num_periods//3:2*args.num_periods//3].mean()) if has_data and args.num_periods >= 3 else 0.0,
                'late_periods_mean': float(rews_llirl[2*args.num_periods//3:].mean()) if has_data and args.num_periods >= 3 else 0.0
            },
            'consistency': {
                'period_std_mean': float(rews_llirl.std(axis=1).mean()) if has_data else 0.0,
                'iteration_std_mean': float(rews_llirl.std(axis=0).mean()) if has_data else 0.0
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

