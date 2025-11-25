"""
Resume LLIRL training from checkpoint
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

# Register environment
from myrllib.envs import sumo_env
import gym
gym.register(
    'SUMO-SingleIntersection-v1',
    entry_point='myrllib.envs.sumo_env:SUMOEnv',
    max_episode_steps=14400  # 4 giờ = 14400 giây
)

from myrllib.episodes.episode import BatchEpisodes
from myrllib.samplers.sampler import BatchSampler
from myrllib.policies import NormalMLPPolicy
from myrllib.algorithms.reinforce import REINFORCE
from myrllib.algorithms.trpo import TRPO
from myrllib.algorithms.ppo import PPO
from load_models import load_policies, load_task_info, load_checkpoint

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--sumo_config', type=str, 
        default='../nets/single-intersection/run_morning_6to10.sumocfg',
        help='path to SUMO configuration file')
parser.add_argument('--model_path', type=str, default='saves/sumo_single_intersection',
        help='folder for loading models')
parser.add_argument('--output', type=str, default='output/sumo_single_intersection',
        help='output folder for saving results')
parser.add_argument('--resume_from_period', type=int, default=None,
        help='resume from specific period (if None, resume from last checkpoint)')
parser.add_argument('--num_iter', type=int, default=50,
        help='number of policy iterations per period')
parser.add_argument('--num_periods', type=int, default=30,
        help='total number of periods')
parser.add_argument('--algorithm', type=str, default='reinforce',
        help='reinforce, trpo, or ppo')
parser.add_argument('--opt', type=str, default='sgd',
        help='sgd or adam')
parser.add_argument('--lr', type=float, default=1e-3,
        help='learning rate')
parser.add_argument('--baseline', type=str, default=None,
        help='linear or None')
parser.add_argument('--batch_size', type=int, default=8,
        help='number of rollouts per iteration')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

# Load saved models
print('Loading saved models...')
tasks, task_ids = load_task_info(args.model_path)
if tasks is None:
    raise ValueError("Task info not found. Please run env_clustering.py first.")

if args.resume_from_period:
    policies, checkpoint = load_checkpoint(args.model_path, args.resume_from_period, device)
    start_period = args.resume_from_period
    num_policies = checkpoint['num_policies']
    state_dim = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
else:
    policies, checkpoint = load_policies(args.model_path, device)
    start_period = len(policies)  # Continue from where we left off
    num_policies = len(policies)
    state_dim = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']

print(f'Resuming from period {start_period + 1}')

# Create sampler
env_name = 'SUMO-SingleIntersection-v1'
sumo_config_path = os.path.abspath(args.sumo_config)
sampler = BatchSampler(env_name, args.batch_size, num_workers=1, seed=args.seed,
                      sumo_config_path=sumo_config_path)

# Create learners for existing policies
def generate_learner(policy):
    if args.algorithm == 'trpo':
        return TRPO(policy, baseline=args.baseline, device=device)
    elif args.algorithm == 'ppo':
        return PPO(policy, baseline=args.baseline, lr=args.lr, opt=args.opt, device=device)
    else:
        return REINFORCE(policy, baseline=args.baseline, lr=args.lr, opt=args.opt, device=device)

learners = [generate_learner(policy) for policy in policies]

# Load existing rewards if available
rews_llirl = None
if os.path.exists(os.path.join(args.output, 'rews_llirl.npy')):
    rews_llirl = np.load(os.path.join(args.output, 'rews_llirl.npy'))
    # Extend if needed
    if rews_llirl.shape[0] < args.num_periods:
        new_shape = (args.num_periods, rews_llirl.shape[1])
        extended = np.zeros(new_shape)
        extended[:rews_llirl.shape[0]] = rews_llirl
        rews_llirl = extended
else:
    rews_llirl = np.zeros((args.num_periods, args.num_iter))

def inner_train(policy, learner, num_iter):
    rews = np.zeros(num_iter)
    for idx in tqdm(range(num_iter)):
        episodes = sampler.sample(policy, device=device)
        rews[idx] = episodes.evaluate()
        learner.step(episodes, clip=True)
    return rews

# Continue training
print('====== Resuming LLIRL Training =======')
for period in range(start_period, min(args.num_periods, len(tasks))):
    print(f'\n----------- Time period {period+1}------')
    task = tasks[period]
    print(f'The task information: {task}')
    sampler.reset_task(task)

    task_id = int(task_ids[period])
    if task_id == num_policies + 1:
        print('Generate a new policy...')
        policy = NormalMLPPolicy(state_dim, action_dim, 
                hidden_sizes=(hidden_size,) * num_layers)
        index = np.random.choice(num_policies)
        policy.load_state_dict(policies[index].state_dict())
        learner = generate_learner(policy)
        rews = inner_train(policy, learner, args.num_iter)
        policies.append(policy)
        learners.append(learner)
        num_policies += 1
    elif task_id <= num_policies:
        print(f'Choosing the policy {task_id}')
        policy = policies[task_id-1]
        learner = learners[task_id-1]
        rews = inner_train(policy, learner, args.num_iter)
        policies[task_id-1] = policy
        learners[task_id-1] = learner
    else:
        raise ValueError(f'Error on task id: {task_id}')
    
    rews_llirl[period] = rews
    print(f'Average return: {rews.mean():.2f}')
    np.save(os.path.join(args.output, 'rews_llirl.npy'), rews_llirl)
    
    # Save checkpoint
    if (period + 1) % 10 == 0:
        checkpoint_path = os.path.join(args.model_path, f'llirl_checkpoint_period_{period+1}.pth')
        torch.save({
            'policies': [p.state_dict() for p in policies],
            'num_policies': num_policies,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'period': period + 1,
            'task_ids': task_ids[:period+1]
        }, checkpoint_path)

# Save final
final_policies_path = os.path.join(args.model_path, 'policies_final.pth')
torch.save({
    'policies': [p.state_dict() for p in policies],
    'num_policies': num_policies,
    'state_dim': state_dim,
    'action_dim': action_dim,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'task_ids': task_ids
}, final_policies_path)

print(f'\nTraining completed!')
print(f'Running time: {(time.time() - start_time) / 60:.2f} min')

