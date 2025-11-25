"""
Utility script to load saved LLIRL models
"""

import torch
import numpy as np
import pickle
import os
from myrllib.mixture.env_model import EnvModel
from myrllib.policies import NormalMLPPolicy
from myrllib.mixture.inference import CRP

def load_env_models(model_path, device='cpu'):
    """Load environment models library"""
    env_models_path = os.path.join(model_path, 'env_models.pth')
    if not os.path.exists(env_models_path):
        raise FileNotFoundError(f"Environment models not found at {env_models_path}")
    
    checkpoint = torch.load(env_models_path, map_location=device)
    env_models = []
    for state_dict in checkpoint['env_models']:
        env_model = EnvModel(
            checkpoint['input_size'],
            checkpoint['output_size'],
            hidden_sizes=checkpoint['hidden_sizes']
        ).to(device)
        env_model.load_state_dict(state_dict)
        env_models.append(env_model)
    
    print(f"Loaded {len(env_models)} environment models")
    return env_models

def load_env_model_init(model_path, device='cpu'):
    """Load universal initialization model"""
    env_model_init_path = os.path.join(model_path, 'env_model_init.pth')
    if not os.path.exists(env_model_init_path):
        return None
    
    checkpoint = torch.load(env_model_init_path, map_location=device)
    # Need to know the architecture - load from env_models if available
    env_models_path = os.path.join(model_path, 'env_models.pth')
    if os.path.exists(env_models_path):
        info = torch.load(env_models_path, map_location=device)
        env_model_init = EnvModel(
            info['input_size'],
            info['output_size'],
            hidden_sizes=info['hidden_sizes']
        ).to(device)
        env_model_init.load_state_dict(checkpoint)
        return env_model_init
    return None

def load_crp_state(model_path):
    """Load CRP state"""
    crp_path = os.path.join(model_path, 'crp_state.pkl')
    if not os.path.exists(crp_path):
        return None
    
    with open(crp_path, 'rb') as f:
        crp_data = pickle.load(f)
    
    crp = CRP(zeta=crp_data['zeta'])
    crp._L = crp_data['L']
    crp._t = crp_data['t']
    crp._prior = crp_data['prior']
    
    print(f"Loaded CRP state: L={crp._L}, t={crp._t}")
    return crp

def load_policies(model_path, device='cpu'):
    """Load policy library"""
    policies_path = os.path.join(model_path, 'policies_final.pth')
    if not os.path.exists(policies_path):
        raise FileNotFoundError(f"Policies not found at {policies_path}")
    
    checkpoint = torch.load(policies_path, map_location=device)
    policies = []
    for state_dict in checkpoint['policies']:
        policy = NormalMLPPolicy(
            checkpoint['state_dim'],
            checkpoint['action_dim'],
            hidden_sizes=(checkpoint['hidden_size'],) * checkpoint['num_layers']
        ).to(device)
        policy.load_state_dict(state_dict)
        policies.append(policy)
    
    print(f"Loaded {len(policies)} policies")
    return policies, checkpoint

def load_task_info(model_path):
    """Load task information"""
    task_info_path = os.path.join(model_path, 'task_info.npy')
    if not os.path.exists(task_info_path):
        return None
    
    task_info = np.load(task_info_path)
    tasks = task_info[:, :-1]
    task_ids = task_info[:, -1]
    
    print(f"Loaded task info: {len(tasks)} periods")
    return tasks, task_ids

def load_checkpoint(model_path, period, device='cpu'):
    """Load checkpoint at specific period"""
    checkpoint_path = os.path.join(model_path, f'llirl_checkpoint_period_{period}.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policies = []
    for state_dict in checkpoint['policies']:
        policy = NormalMLPPolicy(
            checkpoint['state_dim'],
            checkpoint['action_dim'],
            hidden_sizes=(checkpoint['hidden_size'],) * checkpoint['num_layers']
        ).to(device)
        policy.load_state_dict(state_dict)
        policies.append(policy)
    
    print(f"Loaded checkpoint at period {checkpoint['period']}: {len(policies)} policies")
    return policies, checkpoint

