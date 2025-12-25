"""
Export LLIRL results to various formats for analysis
"""

import numpy as np
import json
import torch
import os
import argparse
from load_models import load_policies, load_env_models, load_crp_state, load_task_info

def export_to_csv(model_path, output_path, export_dir='exported_results'):
    """Export results to CSV format"""
    import pandas as pd
    
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    # Export rewards
    rewards_path = os.path.join(output_path, 'rews_llirl.npy')
    if os.path.exists(rewards_path):
        rewards = np.load(rewards_path)
        df_rewards = pd.DataFrame(rewards, 
                                 columns=[f'iter_{i+1}' for i in range(rewards.shape[1])])
        df_rewards.insert(0, 'period', range(1, len(rewards) + 1))
        df_rewards['avg_reward'] = rewards.mean(axis=1)
        df_rewards['std_reward'] = rewards.std(axis=1)
        df_rewards['max_reward'] = rewards.max(axis=1)
        df_rewards['min_reward'] = rewards.min(axis=1)
        csv_path = os.path.join(export_dir, 'rewards.csv')
        df_rewards.to_csv(csv_path, index=False)
        print(f"Exported rewards to {csv_path}")
    
    # Export task info
    tasks, task_ids = load_task_info(model_path)
    if tasks is not None:
        df_tasks = pd.DataFrame(tasks, columns=[f'task_param_{i+1}' for i in range(tasks.shape[1])])
        df_tasks.insert(0, 'period', range(1, len(tasks) + 1))
        df_tasks['cluster_id'] = task_ids.flatten()
        csv_path = os.path.join(export_dir, 'task_clustering.csv')
        df_tasks.to_csv(csv_path, index=False)
        print(f"Exported task clustering to {csv_path}")
    
    # Export optimal data
    optimal_path = os.path.join(model_path, 'optimal_period_data.json')
    if os.path.exists(optimal_path):
        with open(optimal_path, 'r') as f:
            opt_data = json.load(f)
        df_optimal = pd.DataFrame(opt_data)
        csv_path = os.path.join(export_dir, 'optimal_period_data.csv')
        df_optimal.to_csv(csv_path, index=False)
        print(f"Exported optimal period data to {csv_path}")

def export_model_info(model_path, export_dir='exported_results'):
    """Export model information"""
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    info = {}
    
    # Load and count models
    try:
        env_models = load_env_models(model_path)
        info['num_env_models'] = len(env_models)
        info['env_model_architecture'] = {
            'input_size': env_models[0].input_size if len(env_models) > 0 else None,
            'output_size': env_models[0].output_size if len(env_models) > 0 else None,
            'hidden_sizes': env_models[0].hidden_sizes if len(env_models) > 0 else None
        }
    except:
        info['num_env_models'] = 0
    
    try:
        policies, policy_info = load_policies(model_path)
        info['num_policies'] = len(policies)
        info['policy_architecture'] = {
            'state_dim': policy_info.get('state_dim'),
            'action_dim': policy_info.get('action_dim'),
            'hidden_size': policy_info.get('hidden_size'),
            'num_layers': policy_info.get('num_layers')
        }
    except:
        info['num_policies'] = 0
    
    # Save model info
    info_path = os.path.join(export_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Exported model info to {info_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='saves/sumo_single_intersection',
            help='path to saved models')
    parser.add_argument('--output_path', type=str, default='output/sumo_single_intersection',
            help='path to output folder')
    parser.add_argument('--export_dir', type=str, default='exported_results',
            help='directory to save exported files')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'json', 'all'],
            help='export format')
    args = parser.parse_args()
    
    print("Exporting LLIRL results...")
    
    if args.format in ['csv', 'all']:
        try:
            import pandas as pd
            export_to_csv(args.model_path, args.output_path, args.export_dir)
        except ImportError:
            print("Warning: pandas not installed. Skipping CSV export.")
            print("Install with: pip install pandas")
    
    if args.format in ['json', 'all']:
        export_model_info(args.model_path, args.export_dir)
    
    print("\nExport completed!")

if __name__ == '__main__':
    main()





