"""
Utility functions for policy operations
Including general policy creation and evaluation
"""

import torch
import numpy as np
from myrllib.policies import NormalMLPPolicy
from myrllib.mixture.env_model import EnvModel


def create_general_policy(policies, priors, state_dim, action_dim, hidden_sizes, device='cpu'):
    """
    Tạo general policy từ weighted average của tất cả policies
    
    Args:
        policies: List of policy objects
        priors: Prior probabilities (from CRP)
        state_dim: State dimension
        action_dim: Action dimension
        hidden_sizes: Hidden layer sizes
        device: Device to place the policy on
    
    Returns:
        general_policy: Weighted average policy
    """
    if len(policies) == 0:
        return None
    
    # Validate priors length matches policies length
    if len(priors) != len(policies):
        raise ValueError(f'Priors length ({len(priors)}) must match policies length ({len(policies)})')
    
    # Normalize priors
    priors_array = np.array(priors)
    if priors_array.sum() > 0:
        priors_normalized = priors_array / priors_array.sum()
    else:
        priors_normalized = np.ones(len(policies)) / len(policies)
    
    # Create general policy với cùng architecture
    general_policy = NormalMLPPolicy(
        state_dim,
        action_dim,
        hidden_sizes=hidden_sizes
    )
    
    # Get device from first policy if available
    if len(policies) > 0:
        first_param = next(policies[0].parameters())
        if first_param.is_cuda:
            device = first_param.device
    
    # Weighted average của parameters
    for name, param in general_policy.named_parameters():
        weighted_param = None
        for prior, policy in zip(priors_normalized, policies):
            policy_param = dict(policy.named_parameters())[name]
            # Ensure policy_param is on the same device
            policy_param_data = policy_param.data.clone().to(device)
            if weighted_param is None:
                weighted_param = prior * policy_param_data
            else:
                weighted_param += prior * policy_param_data
        
        if weighted_param is not None:
            param.data.copy_(weighted_param.to(param.device))
    
    # Move general policy to device
    general_policy = general_policy.to(device)
    
    return general_policy


def create_general_env_model(env_models, priors, input_size, output_size, hidden_sizes):
    """
    Tạo general environment model từ weighted average
    
    Args:
        env_models: List of environment model objects
        priors: Prior probabilities
        input_size: Input dimension
        output_size: Output dimension
        hidden_sizes: Hidden layer sizes
    
    Returns:
        general_env_model: Weighted average environment model
    """
    if len(env_models) == 0:
        return None
    
    # Validate priors length matches env_models length
    if len(priors) != len(env_models):
        raise ValueError(f'Priors length ({len(priors)}) must match env_models length ({len(env_models)})')
    
    # Normalize priors
    priors_array = np.array(priors)
    if priors_array.sum() > 0:
        priors_normalized = priors_array / priors_array.sum()
    else:
        priors_normalized = np.ones(len(env_models)) / len(env_models)
    
    # Create general env model
    general_env_model = EnvModel(
        input_size,
        output_size,
        hidden_sizes=hidden_sizes
    )
    
    # Weighted average của parameters
    for name, param in general_env_model.named_parameters():
        weighted_param = None
        for prior, env_model in zip(priors_normalized, env_models):
            env_param = dict(env_model.named_parameters())[name]
            if weighted_param is None:
                weighted_param = prior * env_param.data.clone()
            else:
                weighted_param += prior * env_param.data.clone()
        
        if weighted_param is not None:
            param.data.copy_(weighted_param)
    
    return general_env_model


def evaluate_policy_performance(policy, sampler, num_episodes=5, device='cpu'):
    """
    Evaluate policy performance bằng cách collect episodes và tính average reward
    
    Args:
        policy: Policy to evaluate
        sampler: BatchSampler
        num_episodes: Number of episodes to collect
        device: Device
    
    Returns:
        average_reward: Average reward over episodes
        episodes: Collected episodes
    """
    total_reward = 0.0
    all_episodes = []
    
    for _ in range(num_episodes):
        episodes = sampler.sample(policy, device=device)
        reward = episodes.evaluate()
        total_reward += reward
        all_episodes.append(episodes)
    
    average_reward = total_reward / num_episodes
    return average_reward, all_episodes


def evaluate_policies(policies, sampler, num_test_episodes=3, device='cpu'):
    """
    Evaluate tất cả policies và trả về policy tốt nhất
    
    Args:
        policies: List of policies
        sampler: BatchSampler
        num_test_episodes: Number of test episodes per policy
        device: Device
    
    Returns:
        best_policy: Policy với performance tốt nhất
        best_reward: Reward của best policy
        policy_rewards: Dictionary mapping policy index to reward
    """
    if len(policies) == 0:
        return None, -np.inf, {}
    
    policy_rewards = {}
    best_policy = None
    best_reward = -np.inf
    
    for idx, policy in enumerate(policies):
        avg_reward, _ = evaluate_policy_performance(
            policy, sampler, num_test_episodes, device
        )
        policy_rewards[idx] = avg_reward
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_policy = policy
    
    return best_policy, best_reward, policy_rewards


