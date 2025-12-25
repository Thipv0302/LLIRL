"""
Double Deep Q-Network (DDQN) Algorithm
Based on: van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", AAAI 2016
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """Deep Q-Network"""
    def __init__(self, state_dim, action_dim, hidden_sizes=(200, 200)):
        super(DQNNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        layers = []
        input_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
    
    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DDQN:
    """Double Deep Q-Network Agent"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.95, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 batch_size=32, replay_buffer_size=10000, target_update_freq=100,
                 hidden_sizes=(200, 200), device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Q-networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        # Training step counter
        self.step_count = 0
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update_epsilon(self):
        """Decay epsilon"""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step with Double DQN"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use main network to select action, target network to evaluate
        with torch.no_grad():
            next_q_values = self.q_network(next_states)  # Main network selects action
            next_actions = next_q_values.argmax(1)
            next_q_values_target = self.target_network(next_states)  # Target network evaluates
            next_q_value = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_value = rewards + (1 - dones) * self.gamma * next_q_value
        
        # Compute loss
        loss = F.mse_loss(q_value, target_q_value)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.step_count = checkpoint.get('step_count', 0)





