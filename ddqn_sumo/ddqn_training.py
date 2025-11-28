"""
DDQN Training Script for SUMO Environment
Double Deep Q-Network baseline for comparison with LLIRL
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
    max_episode_steps=3600  # 1 giờ = 3600 giây
)

from myrllib.algorithms.ddqn import DDQN
from myrllib.envs.sumo_env import SUMOEnv

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
parser.add_argument('--num_periods', type=int, default=30,
        help='number of environment changes')
parser.add_argument('--num_episodes', type=int, default=100,
        help='number of episodes per period')
parser.add_argument('--max_steps', type=int, default=3600,
        help='maximum steps per episode (1 giờ = 3600 giây)')
parser.add_argument('--lr', type=float, default=1e-3,
        help='learning rate')
parser.add_argument('--gamma', type=float, default=0.95,
        help='discount factor')
parser.add_argument('--batch_size', type=int, default=32,
        help='batch size for training')
parser.add_argument('--replay_buffer_size', type=int, default=10000,
        help='replay buffer size')
parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[200, 200],
        help='hidden layer sizes')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
print(args)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

######################## Main Functions #######################################
# Create environment
sumo_config_path = os.path.abspath(args.sumo_config)
env = SUMOEnv(sumo_config_path=sumo_config_path, max_steps=args.max_steps)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print(f'State dim: {state_dim}; Action dim: {action_dim}')

# Create DDQN agent
ddqn = DDQN(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=args.lr,
    gamma=args.gamma,
    batch_size=args.batch_size,
    replay_buffer_size=args.replay_buffer_size,
    hidden_sizes=tuple(args.hidden_sizes),
    device=device
)

# Generate tasks (different traffic conditions)
tasks = np.random.uniform(0.5, 2.0, size=(args.num_periods, 1))
route_vars = np.random.uniform(0.0, 0.5, size=(args.num_periods, 1))
tasks = np.concatenate([tasks, route_vars], axis=1)

# Storage for results
all_rewards = np.zeros((args.num_periods, args.num_episodes))
all_losses = []

if not os.path.exists(args.output):
    os.makedirs(args.output)
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

print('====== DDQN Training for SUMO ======')

# Training loop
for period in range(args.num_periods):
    print(f'\n----------- Period {period+1}/{args.num_periods} -----------')
    task = tasks[period]
    print(f'Task: {task}')
    env.reset_task(task)
    
    period_rewards = []
    period_losses = []
    
    for episode in tqdm(range(args.num_episodes), desc=f'Period {period+1}'):
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        
        for step in range(args.max_steps):
            # Select action
            action = ddqn.select_action(state, training=True)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            ddqn.store_transition(state, action, reward, next_state, done)
            
            # Train
            if len(ddqn.replay_buffer) >= args.batch_size:
                loss = ddqn.train_step()
                if loss is not None:
                    episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        period_rewards.append(episode_reward)
        if episode_losses:
            period_losses.extend(episode_losses)
        
        # Update epsilon
        ddqn.update_epsilon()
    
    all_rewards[period] = period_rewards
    if period_losses:
        all_losses.append(np.mean(period_losses))
    
    avg_reward = np.mean(period_rewards)
    print(f'Average reward: {avg_reward:.2f}')
    print(f'Average loss: {np.mean(period_losses) if period_losses else 0:.4f}')
    print(f'Epsilon: {ddqn.epsilon:.4f}')
    
    # Save results
    np.save(os.path.join(args.output, 'rews_ddqn.npy'), all_rewards)
    if all_losses:
        np.save(os.path.join(args.output, 'losses_ddqn.npy'), np.array(all_losses))
    
    # Save model
    if (period + 1) % 10 == 0:
        ddqn.save(os.path.join(args.model_path, f'ddqn_model_period_{period+1}.pth'))

# Save final model
ddqn.save(os.path.join(args.model_path, 'ddqn_model_final.pth'))

print(f'\nTraining completed!')
print(f'Running time: {(time.time() - start_time) / 60:.2f} minutes')

env.close()

