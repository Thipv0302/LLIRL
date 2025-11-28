"""
Convert DDQN Q-network weights to Policy network initialization
Transfer Learning: Initialize LLIRL policy from trained DDQN model
"""

import torch
import torch.nn as nn
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myrllib.policies import NormalMLPPolicy


def convert_ddqn_to_policy(ddqn_model_path, state_dim, action_dim, 
                           hidden_sizes=(200, 200), device='cpu'):
    """
    Load DDQN model and convert to policy initialization
    
    Args:
        ddqn_model_path: Path to DDQN model (.pth file)
        state_dim: State dimension
        action_dim: Action dimension (for DDQN, this is discrete actions)
        hidden_sizes: Hidden layer sizes tuple
        device: Device to load on
    
    Returns:
        policy: Initialized policy with DDQN weights (if successful)
        success: Boolean indicating if conversion was successful
    """
    try:
        # Load DDQN checkpoint
        checkpoint = torch.load(ddqn_model_path, map_location=device)
        
        # Extract Q-network state dict
        if isinstance(checkpoint, dict):
            if 'q_network' in checkpoint:
                ddqn_state_dict = checkpoint['q_network']
            elif 'network' in checkpoint:
                # Direct network state dict
                ddqn_state_dict = checkpoint
            else:
                # Assume it's the state dict itself
                ddqn_state_dict = checkpoint
        else:
            ddqn_state_dict = checkpoint
        
        # Create policy
        policy = NormalMLPPolicy(state_dim, action_dim, hidden_sizes=hidden_sizes).to(device)
        policy_state_dict = policy.state_dict()
        
        # Map DDQN layers to policy layers
        # DDQN structure: network.0 (Linear), network.1 (ReLU), network.2 (Linear), network.3 (ReLU), network.4 (Linear)
        # Policy structure: layer1, layer2, mu, sigma
        
        copied_layers = []
        
        # Map first hidden layer
        if 'network.0.weight' in ddqn_state_dict and 'layer1.weight' in policy_state_dict:
            if ddqn_state_dict['network.0.weight'].shape == policy_state_dict['layer1.weight'].shape:
                policy_state_dict['layer1.weight'] = ddqn_state_dict['network.0.weight'].clone()
                policy_state_dict['layer1.bias'] = ddqn_state_dict['network.0.bias'].clone()
                copied_layers.append('layer1')
        
        # Map second hidden layer
        if 'network.2.weight' in ddqn_state_dict and 'layer2.weight' in policy_state_dict:
            if ddqn_state_dict['network.2.weight'].shape == policy_state_dict['layer2.weight'].shape:
                policy_state_dict['layer2.weight'] = ddqn_state_dict['network.2.weight'].clone()
                policy_state_dict['layer2.bias'] = ddqn_state_dict['network.2.bias'].clone()
                copied_layers.append('layer2')
        
        # For mu layer (output), use Q-network output layer
        # Since DDQN outputs Q-values for each action, we average them for policy mean
        if 'network.4.weight' in ddqn_state_dict and 'mu.weight' in policy_state_dict:
            q_output_weight = ddqn_state_dict['network.4.weight']  # [action_dim, hidden_size]
            q_output_bias = ddqn_state_dict.get('network.4.bias', None)  # [action_dim]
            
            # Average Q-values across actions for policy mean initialization
            # This gives us a "general" action value
            if len(hidden_sizes) > 0:
                hidden_size = hidden_sizes[-1]
                if q_output_weight.shape[0] == action_dim and q_output_weight.shape[1] == hidden_size:
                    # Average the Q-value weights
                    avg_weight = q_output_weight.mean(dim=0, keepdim=True)  # [1, hidden_size]
                    # Repeat for action_dim outputs (since policy outputs action_dim values)
                    policy_state_dict['mu.weight'] = avg_weight.repeat(action_dim, 1)
                    
                    if q_output_bias is not None:
                        avg_bias = q_output_bias.mean()
                        policy_state_dict['mu.bias'] = torch.full((action_dim,), avg_bias, device=device)
                    else:
                        policy_state_dict['mu.bias'] = torch.zeros(action_dim, device=device)
                    
                    copied_layers.append('mu')
        
        # Load into policy
        policy.load_state_dict(policy_state_dict)
        
        print(f"✓ Successfully converted DDQN to Policy")
        print(f"  Copied layers: {', '.join(copied_layers)}")
        print(f"  Policy initialized from: {ddqn_model_path}")
        
        return policy, True
        
    except Exception as e:
        print(f"[WARNING] Failed to convert DDQN to Policy: {e}")
        print(f"  Will use random initialization instead")
        # Return randomly initialized policy
        policy = NormalMLPPolicy(state_dim, action_dim, hidden_sizes=hidden_sizes).to(device)
        return policy, False


def load_ddqn_for_transfer(ddqn_model_path, device='cpu'):
    """
    Load DDQN model to extract architecture info
    
    Returns:
        state_dim, action_dim, hidden_sizes if successful, None otherwise
    """
    try:
        checkpoint = torch.load(ddqn_model_path, map_location=device)
        
        # Try to infer from state dict
        if isinstance(checkpoint, dict) and 'q_network' in checkpoint:
            state_dict = checkpoint['q_network']
        else:
            state_dict = checkpoint
        
        # Infer hidden sizes from layer names
        hidden_sizes = []
        layer_idx = 0
        while f'network.{layer_idx * 2}.weight' in state_dict:
            weight = state_dict[f'network.{layer_idx * 2}.weight']
            if layer_idx == 0:
                state_dim = weight.shape[1]
            hidden_size = weight.shape[0]
            if layer_idx < 2:  # Only count hidden layers (not output)
                hidden_sizes.append(hidden_size)
            layer_idx += 1
        
        # Get action_dim from output layer
        if f'network.{len(hidden_sizes) * 2}.weight' in state_dict:
            action_dim = state_dict[f'network.{len(hidden_sizes) * 2}.weight'].shape[0]
        else:
            action_dim = None
        
        return state_dim, action_dim, tuple(hidden_sizes)
    except:
        return None, None, None


