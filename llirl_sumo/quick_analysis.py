"""
Quick analysis script - Load and print summary of LLIRL results
"""

import numpy as np
import json
import os
import argparse

def quick_summary(model_path, output_path):
    """Print quick summary of results"""
    print("="*60)
    print("LLIRL QUICK SUMMARY")
    print("="*60)
    
    # Load rewards
    rewards_path = os.path.join(output_path, 'rews_llirl.npy')
    if os.path.exists(rewards_path):
        rewards = np.load(rewards_path)
        print(f"\n📊 Performance:")
        print(f"   Total periods: {rewards.shape[0]}")
        print(f"   Iterations per period: {rewards.shape[1]}")
        print(f"   Final average reward: {rewards[-1].mean():.2f}")
        print(f"   Best period average: {rewards.mean(axis=1).max():.2f}")
        print(f"   Overall average: {rewards.mean():.2f}")
    
    # Load training summary
    summary_path = os.path.join(model_path, 'training_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        print(f"\n🎯 Training Info:")
        print(f"   Number of clusters: {summary['num_clusters']}")
        print(f"   Algorithm: {summary['algorithm']}")
        print(f"   Training time: {summary['training_time_minutes']:.2f} minutes")
        print(f"   Best period: {summary['best_period']}")
    
    # Load clustering summary
    clustering_path = os.path.join(model_path, 'clustering_summary.json')
    if os.path.exists(clustering_path):
        with open(clustering_path, 'r') as f:
            clustering = json.load(f)
        print(f"\n🔍 Clustering Info:")
        print(f"   Number of clusters: {clustering['num_clusters']}")
        print(f"   Zeta: {clustering['zeta']}")
    
    # Load cluster stats
    cluster_stats_path = os.path.join(model_path, 'cluster_statistics.json')
    if os.path.exists(cluster_stats_path):
        with open(cluster_stats_path, 'r') as f:
            cluster_stats = json.load(f)
        print(f"\n📈 Cluster Distribution:")
        for key in sorted(cluster_stats.keys(), key=lambda x: int(x.split('_')[1])):
            num_periods = cluster_stats[key]['num_periods']
            print(f"   {key}: {num_periods} periods")
    
    # Load optimal data
    optimal_path = os.path.join(model_path, 'optimal_period_data.json')
    if os.path.exists(optimal_path):
        with open(optimal_path, 'r') as f:
            optimal = json.load(f)
        if optimal['optimal_rewards']:
            print(f"\n⭐ Optimal Performance:")
            print(f"   Best reward: {max(optimal['optimal_rewards']):.2f}")
            print(f"   Average optimal: {np.mean(optimal['optimal_rewards']):.2f}")
    
    print("\n" + "="*60)
    print("✅ All output files are available!")
    print("="*60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='saves/sumo_single_intersection',
            help='path to saved models')
    parser.add_argument('--output_path', type=str, default='output/sumo_single_intersection',
            help='path to output folder')
    args = parser.parse_args()
    
    quick_summary(args.model_path, args.output_path)

if __name__ == '__main__':
    main()





