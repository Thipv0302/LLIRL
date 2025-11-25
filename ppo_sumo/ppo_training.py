"""
PPO Training Script for SUMO Environment
Proximal Policy Optimization baseline for comparison with LLIRL
"""

import sys
import os
import gym
import numpy as np
import argparse
import torch
from tqdm import tqdm
import time
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Register environment before importing
from myrllib.envs import sumo_env
import gym
gym.register(
    'SUMO-SingleIntersection-v1',
    entry_point='myrllib.envs.sumo_env:SUMOEnv',
    max_episode_steps=14400  # 4 giờ = 14400 giây
)

from myrllib.policies import NormalMLPPolicy
from myrllib.algorithms.ppo import PPO
from myrllib.samplers.sampler import BatchSampler

start_time = time.time()

######################## Arguments ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--sumo_config', type=str, 
        default='../nets/single-intersection/run_morning_6to10.sumocfg',
        help='path to SUMO configuration file')
parser.add_argument('--output', type=str, default='output/sumo_single_intersection',
        help='output folder for saving results')
parser.add_argument('--model_path', type=str, default='saves/sumo_single_intersection',
        help='folder for saving models')
parser.add_argument('--batch_size', type=int, default=8,
        help='number of rollouts per iteration')
parser.add_argument('--hidden_size', type=int, default=200,
        help='hidden size of the policy network')
parser.add_argument('--num_layers', type=int, default=2,
        help='number of hidden layers')
parser.add_argument('--num_iter', type=int, default=50,
        help='number of policy iterations per period')
parser.add_argument('--num_periods', type=int, default=30,
        help='number of environment changes')
parser.add_argument('--lr', type=float, default=3e-4,
        help='learning rate')
parser.add_argument('--algorithm', type=str, default='ppo',
        help='ppo algorithm')
parser.add_argument('--opt', type=str, default='adam',
        help='optimizer: sgd or adam')
parser.add_argument('--baseline', type=str, default='linear',
        help='baseline: linear or None')
parser.add_argument('--clip', type=float, default=0.2,
        help='PPO clip parameter')
parser.add_argument('--epochs', type=int, default=5,
        help='PPO epochs per update')
parser.add_argument('--tau', type=float, default=1.0,
        help='GAE tau parameter')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
print(args)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

######################## Main Functions #######################################
# Create sampler
env_name = 'SUMO-SingleIntersection-v1'
sumo_config_path = os.path.abspath(args.sumo_config)
# Use num_workers=0 on Windows to avoid multiprocessing issues
import platform
num_workers = 0 if platform.system() == 'Windows' else 1
sampler = BatchSampler(env_name, args.batch_size, num_workers=num_workers, seed=args.seed,
                      sumo_config_path=sumo_config_path)

# Handle num_workers=0 case (single env)
if hasattr(sampler, 'envs') and sampler.envs is not None:
    state_dim = int(np.prod(sampler.envs[0].observation_space.shape))
    action_dim = int(np.prod(sampler.envs[0].action_space.shape))
else:
    state_dim = int(np.prod(sampler._env.observation_space.shape))
    action_dim = int(np.prod(sampler._env.action_space.shape))
print(f'State dim: {state_dim}; Action dim: {action_dim}')

# Generate tasks
tasks = np.random.uniform(0.5, 2.0, size=(args.num_periods, 1))
route_vars = np.random.uniform(0.0, 0.5, size=(args.num_periods, 1))
tasks = np.concatenate([tasks, route_vars], axis=1)

# Initialize policy (single policy for all periods - no clustering)
policy = NormalMLPPolicy(state_dim, action_dim, 
        hidden_sizes=(args.hidden_size,) * args.num_layers)
learner = PPO(policy, epochs=args.epochs, clip=args.clip, opt=args.opt, 
              lr=args.lr, baseline=args.baseline, tau=args.tau, device=device)

# Storage for results
all_rewards = np.zeros((args.num_periods, args.num_iter))

if not os.path.exists(args.output):
    os.makedirs(args.output)
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

print('====== PPO Training for SUMO ======')

# Training loop
for period in range(args.num_periods):
    print(f'\n----------- Period {period+1}/{args.num_periods} -----------')
    task = tasks[period]
    print(f'Task: {task}')
    sampler.reset_task(task)
    
    period_rewards = []
    
    for iteration in tqdm(range(args.num_iter), desc=f'Period {period+1}'):
        # Sample episodes
        episodes = sampler.sample(policy, device=device)
        
        # Evaluate
        reward = episodes.evaluate()
        period_rewards.append(reward)
        
        # Update policy with PPO
        learner.step(episodes, clip=True)
    
    all_rewards[period] = period_rewards
    avg_reward = np.mean(period_rewards)
    print(f'Average reward: {avg_reward:.2f}')
    
    # Save results
    np.save(os.path.join(args.output, 'rews_ppo.npy'), all_rewards)
    
    # Save model
    if (period + 1) % 10 == 0:
        torch.save(policy.state_dict(), 
                  os.path.join(args.model_path, f'ppo_policy_period_{period+1}.pth'))

# Save final model
torch.save(policy.state_dict(), os.path.join(args.model_path, 'ppo_policy_final.pth'))

print(f'\nTraining completed!')
print(f'Running time: {(time.time() - start_time) / 60:.2f} minutes')

# Cleanup SUMO connections
try:
    if hasattr(sampler, '_env') and hasattr(sampler._env, 'sumo_running'):
        if sampler._env.sumo_running:
            import traci
            try:
                traci.close()
                sampler._env.sumo_running = False
            except Exception as e:
                print(f'[WARNING] Error closing SUMO: {e}')
    if hasattr(sampler, 'envs') and sampler.envs is not None:
        # Cleanup multiprocessing environments
        try:
            sampler.envs.close()
        except Exception as e:
            print(f'[WARNING] Error closing environments: {e}')
except Exception as e:
    print(f'[WARNING] Error during cleanup: {e}')

print('[OK] Cleanup completed')

