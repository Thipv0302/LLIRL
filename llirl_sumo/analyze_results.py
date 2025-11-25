"""
Analyze LLIRL training results
Load and visualize all saved outputs
"""

import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import os
import argparse
from load_models import load_policies, load_env_models, load_crp_state, load_task_info

def load_all_results(model_path, output_path):
    """Load all LLIRL results"""
    results = {}
    
    # Load rewards
    rewards_path = os.path.join(output_path, 'rews_llirl.npy')
    if os.path.exists(rewards_path):
        results['rewards'] = np.load(rewards_path)
        print(f"Loaded rewards: shape {results['rewards'].shape}")
    
    # Load task info
    tasks, task_ids = load_task_info(model_path)
    if tasks is not None:
        results['tasks'] = tasks
        results['task_ids'] = task_ids
        print(f"Loaded task info: {len(tasks)} periods")
    
    # Load optimal period data
    optimal_path = os.path.join(model_path, 'optimal_period_data.json')
    if os.path.exists(optimal_path):
        with open(optimal_path, 'r') as f:
            results['optimal_data'] = json.load(f)
        print(f"Loaded optimal period data")
    
    # Load period-cluster mapping
    mapping_path = os.path.join(model_path, 'period_cluster_mapping.json')
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            results['mapping'] = json.load(f)
        print(f"Loaded period-cluster mapping")
    
    # Load training summary
    summary_path = os.path.join(model_path, 'training_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            results['summary'] = json.load(f)
        print(f"Loaded training summary")
    
    # Load clustering summary
    clustering_path = os.path.join(model_path, 'clustering_summary.json')
    if os.path.exists(clustering_path):
        with open(clustering_path, 'r') as f:
            results['clustering'] = json.load(f)
        print(f"Loaded clustering summary")
    
    # Load cluster statistics
    cluster_stats_path = os.path.join(model_path, 'cluster_statistics.json')
    if os.path.exists(cluster_stats_path):
        with open(cluster_stats_path, 'r') as f:
            results['cluster_stats'] = json.load(f)
        print(f"Loaded cluster statistics")
    
    return results

def plot_analysis(results, save_dir='analysis_results'):
    """Plot comprehensive analysis"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Reward progression
    ax1 = plt.subplot(3, 2, 1)
    if 'rewards' in results:
        rewards = results['rewards']
        periods = range(1, rewards.shape[0] + 1)
        avg_rewards = rewards.mean(axis=1)
        std_rewards = rewards.std(axis=1)
        ax1.plot(periods, avg_rewards, 'b-o', linewidth=2, markersize=6, label='Average')
        ax1.fill_between(periods, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.3)
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Reward Progression Across Periods')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Plot 2: Learning curves (first and last periods)
    ax2 = plt.subplot(3, 2, 2)
    if 'rewards' in results:
        rewards = results['rewards']
        iterations = range(1, rewards.shape[1] + 1)
        ax2.plot(iterations, rewards[0], 'b-', label='Period 1', alpha=0.7)
        if rewards.shape[0] > 1:
            ax2.plot(iterations, rewards[-1], 'r-', label=f'Period {rewards.shape[0]}', alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Reward')
        ax2.set_title('Learning Curves: First vs Last Period')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Plot 3: Cluster assignments
    ax3 = plt.subplot(3, 2, 3)
    if 'task_ids' in results:
        task_ids = results['task_ids']
        periods = range(1, len(task_ids) + 1)
        ax3.scatter(periods, task_ids, c=task_ids, cmap='tab10', s=50, alpha=0.6)
        ax3.set_xlabel('Period')
        ax3.set_ylabel('Cluster ID')
        ax3.set_title('Cluster Assignments Over Time')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Optimal rewards per period
    ax4 = plt.subplot(3, 2, 4)
    if 'optimal_data' in results:
        opt_data = results['optimal_data']
        ax4.plot(opt_data['periods'], opt_data['optimal_rewards'], 'g-s', linewidth=2, markersize=6)
        ax4.set_xlabel('Period')
        ax4.set_ylabel('Optimal Reward')
        ax4.set_title('Best Reward Per Period')
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Cluster size distribution
    ax5 = plt.subplot(3, 2, 5)
    if 'cluster_stats' in results:
        cluster_stats = results['cluster_stats']
        cluster_ids = []
        cluster_sizes = []
        for key, value in cluster_stats.items():
            cluster_id = int(key.split('_')[1])
            cluster_ids.append(cluster_id)
            cluster_sizes.append(value['num_periods'])
        ax5.bar(cluster_ids, cluster_sizes, color='skyblue', edgecolor='black')
        ax5.set_xlabel('Cluster ID')
        ax5.set_ylabel('Number of Periods')
        ax5.set_title('Cluster Size Distribution')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Performance comparison by cluster
    ax6 = plt.subplot(3, 2, 6)
    if 'rewards' in results and 'task_ids' in results:
        rewards = results['rewards']
        task_ids = results['task_ids'].flatten()
        unique_clusters = np.unique(task_ids)
        cluster_avg_rewards = []
        for cluster_id in unique_clusters:
            periods_in_cluster = np.where(task_ids == cluster_id)[0]
            cluster_rewards = rewards[periods_in_cluster].mean()
            cluster_avg_rewards.append(cluster_rewards)
        ax6.bar(unique_clusters, cluster_avg_rewards, color='coral', edgecolor='black')
        ax6.set_xlabel('Cluster ID')
        ax6.set_ylabel('Average Reward')
        ax6.set_title('Average Performance by Cluster')
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'llirl_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plots saved to {save_dir}/llirl_analysis.png")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("LLIRL ANALYSIS SUMMARY")
    print("="*60)
    
    if 'summary' in results:
        summary = results['summary']
        print(f"\nTraining Configuration:")
        print(f"  Total periods: {summary['total_periods']}")
        print(f"  Number of clusters: {summary['num_clusters']}")
        print(f"  Algorithm: {summary['algorithm']}")
        print(f"  Learning rate: {summary['learning_rate']}")
        print(f"  Training time: {summary['training_time_minutes']:.2f} minutes")
    
    if 'rewards' in results:
        rewards = results['rewards']
        print(f"\nPerformance Metrics:")
        print(f"  Final average reward: {rewards[-1].mean():.2f}")
        print(f"  Best period average: {rewards.mean(axis=1).max():.2f}")
        print(f"  Overall average: {rewards.mean():.2f}")
        print(f"  Overall std: {rewards.std():.2f}")
    
    if 'clustering' in results:
        clustering = results['clustering']
        print(f"\nClustering Statistics:")
        print(f"  Number of clusters: {clustering['num_clusters']}")
        print(f"  Zeta (concentration): {clustering['zeta']}")
    
    if 'cluster_stats' in results:
        cluster_stats = results['cluster_stats']
        print(f"\nCluster Distribution:")
        for key, value in sorted(cluster_stats.items(), key=lambda x: int(x[0].split('_')[1])):
            print(f"  {key}: {value['num_periods']} periods")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='saves/sumo_single_intersection',
            help='path to saved models')
    parser.add_argument('--output_path', type=str, default='output/sumo_single_intersection',
            help='path to output folder')
    parser.add_argument('--save_dir', type=str, default='analysis_results',
            help='directory to save analysis plots')
    args = parser.parse_args()
    
    print("Loading LLIRL results...")
    results = load_all_results(args.model_path, args.output_path)
    
    print("\nGenerating analysis plots...")
    plot_analysis(results, args.save_dir)
    
    print("\nAnalysis completed!")

if __name__ == '__main__':
    main()





