"""
Comprehensive Analysis Tool for LLIRL
Analyzes all saved outputs and generates detailed reports
"""

import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import os
import argparse
from load_models import load_policies, load_env_models, load_crp_state, load_task_info

def load_all_data(model_path, output_path):
    """Load all LLIRL data"""
    data = {}
    
    # Core components
    try:
        data['rewards'] = np.load(os.path.join(output_path, 'rews_llirl.npy'))
    except:
        pass
    
    try:
        data['tasks'], data['task_ids'] = load_task_info(model_path)
    except:
        pass
    
    # JSON files
    json_files = [
        'optimal_period_data.json',
        'period_cluster_mapping.json',
        'training_summary.json',
        'clustering_summary.json',
        'cluster_statistics.json',
        'clustering_history.json',
        'convergence_metrics.json',
        'training_metrics.json',
        'experiment_config.json',
        'performance_statistics.json'
    ]
    
    for json_file in json_files:
        filepath = os.path.join(model_path, json_file)
        if os.path.exists(filepath):
            key = json_file.replace('.json', '')
            with open(filepath, 'r') as f:
                data[key] = json.load(f)
    
    return data

def analyze_clustering_quality(data):
    """Analyze clustering quality"""
    if 'clustering_history' not in data:
        return None
    
    history = data['clustering_history']
    
    analysis = {
        'cluster_creation_rate': [],
        'likelihood_confidence': [],
        'posterior_confidence': [],
        'em_convergence': []
    }
    
    for period_idx in range(len(history['periods'])):
        # Cluster creation
        analysis['cluster_creation_rate'].append(
            1.0 if history['new_cluster_created'][period_idx] else 0.0
        )
        
        # Likelihood confidence (entropy)
        if period_idx < len(history['final_likelihoods']):
            llls = np.array(history['final_likelihoods'][period_idx])
            entropy = -np.sum(llls * np.log(llls + 1e-10))
            analysis['likelihood_confidence'].append(float(entropy))
        
        # Posterior confidence
        if period_idx < len(history['final_posteriors']):
            post = np.array(history['final_posteriors'][period_idx])
            entropy = -np.sum(post * np.log(post + 1e-10))
            analysis['posterior_confidence'].append(float(entropy))
        
        # EM convergence
        if period_idx < len(history['em_likelihoods_history']):
            em_llls = history['em_likelihoods_history'][period_idx]
            if len(em_llls) > 1:
                initial = np.array(em_llls[0])
                final = np.array(em_llls[-1])
                change = np.abs(final - initial).max()
                analysis['em_convergence'].append(float(change))
    
    return analysis

def generate_comprehensive_report(data, save_dir='comprehensive_analysis'):
    """Generate comprehensive analysis report"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    report = []
    report.append("="*80)
    report.append("LLIRL COMPREHENSIVE ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    # Experiment Configuration
    if 'experiment_config' in data:
        report.append("EXPERIMENT CONFIGURATION")
        report.append("-"*80)
        config = data['experiment_config']
        for key, value in config.items():
            report.append(f"  {key:30s}: {value}")
        report.append("")
    
    # Performance Summary
    if 'performance_statistics' in data:
        report.append("PERFORMANCE STATISTICS")
        report.append("-"*80)
        stats = data['performance_statistics']
        report.append(f"  Overall Mean Reward:     {stats['overall_mean_reward']:.4f}")
        report.append(f"  Overall Std Reward:       {stats['overall_std_reward']:.4f}")
        report.append(f"  Best Period Mean:         {stats['best_period_mean']:.4f}")
        report.append(f"  Final Period Mean:        {stats['final_period_mean']:.4f}")
        report.append(f"  Learning Trajectory:")
        report.append(f"    Early periods:          {stats['learning_trajectory']['early_periods_mean']:.4f}")
        report.append(f"    Middle periods:         {stats['learning_trajectory']['middle_periods_mean']:.4f}")
        report.append(f"    Late periods:           {stats['learning_trajectory']['late_periods_mean']:.4f}")
        report.append("")
    
    # Clustering Analysis
    if 'clustering_summary' in data:
        report.append("CLUSTERING SUMMARY")
        report.append("-"*80)
        clustering = data['clustering_summary']
        report.append(f"  Total Periods:            {clustering['total_periods']}")
        report.append(f"  Number of Clusters:       {clustering['num_clusters']}")
        report.append(f"  Zeta (concentration):     {clustering['zeta']:.4f}")
        report.append(f"  Clustering Time:          {clustering['clustering_time_minutes']:.2f} minutes")
        report.append("")
    
    # Clustering Quality
    clustering_quality = analyze_clustering_quality(data)
    if clustering_quality:
        report.append("CLUSTERING QUALITY")
        report.append("-"*80)
        report.append(f"  Cluster Creation Rate:    {np.mean(clustering_quality['cluster_creation_rate']):.4f}")
        if clustering_quality['likelihood_confidence']:
            report.append(f"  Avg Likelihood Confidence: {np.mean(clustering_quality['likelihood_confidence']):.4f}")
        if clustering_quality['em_convergence']:
            report.append(f"  Avg EM Convergence:       {np.mean(clustering_quality['em_convergence']):.4f}")
        report.append("")
    
    # Training Summary
    if 'training_summary' in data:
        report.append("TRAINING SUMMARY")
        report.append("-"*80)
        training = data['training_summary']
        report.append(f"  Training Time:             {training['training_time_minutes']:.2f} minutes")
        report.append(f"  Best Period:              {training['best_period']}")
        report.append(f"  Final Average Reward:     {training['final_average_reward']:.4f}")
        report.append("")
    
    # Save report
    report_text = "\n".join(report)
    report_path = os.path.join(save_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"Saved comprehensive report to {report_path}")
    
    # Save clustering quality analysis
    if clustering_quality:
        quality_path = os.path.join(save_dir, 'clustering_quality.json')
        with open(quality_path, 'w') as f:
            json.dump(clustering_quality, f, indent=2)
        print(f"Saved clustering quality analysis to {quality_path}")
    
    return report_text

def plot_comprehensive_analysis(data, save_dir='comprehensive_analysis'):
    """Generate comprehensive plots"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Reward progression
    ax1 = plt.subplot(4, 3, 1)
    if 'rewards' in data:
        rewards = data['rewards']
        periods = range(1, rewards.shape[0] + 1)
        avg_rewards = rewards.mean(axis=1)
        std_rewards = rewards.std(axis=1)
        ax1.plot(periods, avg_rewards, 'b-o', linewidth=2, markersize=4, label='Average')
        ax1.fill_between(periods, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.3)
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Reward Progression')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Plot 2: Clustering evolution
    ax2 = plt.subplot(4, 3, 2)
    if 'clustering_history' in data:
        history = data['clustering_history']
        periods = history['periods']
        num_clusters = history['num_clusters']
        ax2.plot(periods, num_clusters, 'g-s', linewidth=2, markersize=4)
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Number of Clusters')
        ax2.set_title('Cluster Evolution')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Likelihood confidence
    ax3 = plt.subplot(4, 3, 3)
    if 'clustering_history' in data:
        history = data['clustering_history']
        periods = history['periods']
        confidences = []
        for llls in history['final_likelihoods']:
            llls_arr = np.array(llls)
            entropy = -np.sum(llls_arr * np.log(llls_arr + 1e-10))
            confidences.append(entropy)
        if confidences:
            ax3.plot(periods, confidences, 'r-o', linewidth=2, markersize=4)
            ax3.set_xlabel('Period')
            ax3.set_ylabel('Likelihood Entropy')
            ax3.set_title('Clustering Confidence')
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: EM Convergence
    ax4 = plt.subplot(4, 3, 4)
    if 'convergence_metrics' in data:
        conv = data['convergence_metrics']
        if conv['likelihood_convergence']:
            periods = range(1, len(conv['likelihood_convergence']) + 1)
            ax4.plot(periods, conv['likelihood_convergence'], 'm-o', linewidth=2, markersize=4)
            ax4.set_xlabel('Period')
            ax4.set_ylabel('Likelihood Change')
            ax4.set_title('EM Convergence')
            ax4.grid(True, alpha=0.3)
    
    # Plot 5: Gradient norms
    ax5 = plt.subplot(4, 3, 5)
    if 'training_metrics' in data:
        metrics = data['training_metrics']
        if metrics['policy_gradients_norm']:
            # Plot average gradient norm per period
            avg_grads = []
            for grads in metrics['policy_gradients_norm']:
                if grads:
                    avg_grads.append(np.mean(grads))
            if avg_grads:
                periods = range(len(avg_grads))
                ax5.plot(periods, avg_grads, 'c-o', linewidth=2, markersize=4)
                ax5.set_xlabel('Period')
                ax5.set_ylabel('Avg Gradient Norm')
                ax5.set_title('Gradient Magnitude')
                ax5.grid(True, alpha=0.3)
    
    # Plot 6: Learning trajectory
    ax6 = plt.subplot(4, 3, 6)
    if 'performance_statistics' in data:
        stats = data['performance_statistics']
        trajectory = stats['learning_trajectory']
        phases = ['Early', 'Middle', 'Late']
        values = [
            trajectory['early_periods_mean'],
            trajectory['middle_periods_mean'],
            trajectory['late_periods_mean']
        ]
        ax6.bar(phases, values, color=['red', 'orange', 'green'], alpha=0.7)
        ax6.set_ylabel('Mean Reward')
        ax6.set_title('Learning Trajectory')
        ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7-12: Additional plots can be added here
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive plots to {save_dir}/comprehensive_analysis.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='saves/sumo_single_intersection',
            help='path to saved models')
    parser.add_argument('--output_path', type=str, default='output/sumo_single_intersection',
            help='path to output folder')
    parser.add_argument('--save_dir', type=str, default='comprehensive_analysis',
            help='directory to save analysis')
    args = parser.parse_args()
    
    print("Loading all LLIRL data...")
    data = load_all_data(args.model_path, args.output_path)
    
    print("Generating comprehensive report...")
    report = generate_comprehensive_report(data, args.save_dir)
    print("\n" + report)
    
    print("\nGenerating comprehensive plots...")
    plot_comprehensive_analysis(data, args.save_dir)
    
    print("\nComprehensive analysis completed!")

if __name__ == '__main__':
    main()





