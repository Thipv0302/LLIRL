"""
This the code for the paper:
[1] Zhi Wang, Chunlin Chen, and Daoyi Dong, "Lifelong Incremental Reinforcement Learning with 
Online Bayesian Inference", IEEE Transactions on Neural Networks and Learning Systems, 2021.
https://github.com/HeyuanMingong/llinrl.git

This file is for clustering the environment models in a latent space with online
Bayesian inference. The prior distribution on the mixture of environment models
is instantiated as the Chinese Restaurant Process.
"""

### common lib
import sys
import os
import gym
import numpy as np
import argparse 
import torch
from tqdm import tqdm
import time 
from torch.optim import Adam, SGD 
import copy 
from collections import OrderedDict
import torch.nn.functional as F
import pickle
import shutil
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Register environment before importing
from myrllib.envs import sumo_env
import gym
gym.register(
    'SUMO-SingleIntersection-v1',
    entry_point='myrllib.envs.sumo_env:SUMOEnv',
    max_episode_steps=3600  # Default, can be overridden by max_steps parameter
)

### personal lib
from myrllib.episodes.episode import BatchEpisodes 
from myrllib.samplers.sampler import BatchSampler 
from myrllib.policies import NormalMLPPolicy, UniformPolicy  
from myrllib.baselines.baseline import LinearFeatureBaseline
from myrllib.algorithms.reinforce import REINFORCE 
from myrllib.algorithms.trpo import TRPO 
from myrllib.mixture.env_train import env_update, env_nominal_train
from myrllib.mixture.env_model import EnvModel, construct_env_io
from myrllib.mixture.inference import CRP, compute_likelihood


start_time = time.time()
######################## Arguments ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, 
        help='number of rollouts/learning episodes in one policy iteration')
parser.add_argument('--model_path', type=str, default='saves/sumo_single_intersection',
        help='the folder for saving and loading the pretrained model')
parser.add_argument('--sumo_config', type=str, 
        default='../nets/single-intersection/run_morning_6to10.sumocfg',
        help='path to SUMO configuration file')
parser.add_argument('--env_num_layers', type=int, default=2, 
        help='the number of hidden layers of the environment model')
parser.add_argument('--env_hidden_size', type=int, default=200,
        help='the size of the hidden layer of the environment model')
parser.add_argument('--H', type=int, default=4,
        help='using the consecutive H transitions to construct the data')
parser.add_argument('--num_periods', type=int, default=30, 
        help='number of the environment changes')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--et_length', type=int, default=1, 
        help='length of episodic transitions for collecting samples')
parser.add_argument('--max_steps', type=int, default=3600,
        help='maximum steps per episode')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--zeta', type=float, default=0.5,
        help='CRP concentration parameter (lower = easier to create new clusters)')
parser.add_argument('--sigma', type=float, default=0.1,
        help='Likelihood computation sigma (lower = more sensitive)')
parser.add_argument('--tau1', type=float, default=0.5,
        help='Temperature for likelihood normalization')
parser.add_argument('--tau2', type=float, default=0.5,
        help='Temperature for prior normalization')
parser.add_argument('--em_steps', type=int, default=5,
        help='Number of EM algorithm iterations')
args = parser.parse_args()
print(args)

# Override hyperparameters with command line arguments if provided
SIGMA = args.sigma
TAU1 = args.tau1
TAU2 = args.tau2
EM_STEPS = args.em_steps
ZETA = args.zeta
print(f'\nClustering Hyperparameters:')
print(f'  ZETA (CRP concentration): {ZETA}')
print(f'  SIGMA (likelihood): {SIGMA}')
print(f'  TAU1 (likelihood temp): {TAU1}')
print(f'  TAU2 (prior temp): {TAU2}')
print(f'  EM_STEPS: {EM_STEPS}')
print()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Print device information
print(f"\n{'='*60}")
print(f"Device Configuration:")
print(f"  Requested device: {args.device}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Actual device: {device}")
if torch.cuda.is_available():
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  [OK] Training will use GPU")
else:
    print(f"  [WARNING] CUDA not available - Training will use CPU")
print(f"{'='*60}\n")

np.set_printoptions(precision=3)
np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)

######################## Small functions ######################################
def softmax_normalize(array, temperature=1.0):
    array = np.array(array).reshape(-1)
    array -= array.mean()
    array_exp = np.exp(array * temperature)
    array_exp /= array_exp.sum()
    return array_exp

######################## Hyperparameters #####################################
### tune these hyperparameters to get different clustering results
### Optimized values for better clustering (allows more clusters to be created)
SIGMA = 0.1; TAU1 = 0.5; TAU2 = 0.5; EM_STEPS = 5; ZETA = 0.5

######################## Main Functions #######################################
""" ENV_TYPE: the type of parameterizing the environment
Select a model to parameterize the environment
reward: using reward function, default
state-transition: using state transition function
both: using the concatenation of both functions
"""
ENV_TYPE = 'reward'  # For SUMO, we use reward function

# Generate task parameters (traffic intensity variations)
# Each task represents different traffic conditions
tasks = np.random.uniform(0.5, 2.0, size=(args.num_periods, 1))  # traffic intensity
route_vars = np.random.uniform(0.0, 0.5, size=(args.num_periods, 1))
tasks = np.concatenate([tasks, route_vars], axis=1)

### build a sampler given an environment
env_name = 'SUMO-SingleIntersection-v1'
sumo_config_path = os.path.abspath(args.sumo_config)
# Use num_workers=0 on Windows to avoid multiprocessing issues
import platform
num_workers = 0 if platform.system() == 'Windows' else 1
sampler = BatchSampler(env_name, args.batch_size, num_workers=num_workers, seed=args.seed, 
                      sumo_config_path=sumo_config_path, max_steps=args.max_steps) 
env = gym.make(env_name, sumo_config_path=sumo_config_path, max_steps=args.max_steps)
# Handle num_workers=0 case (single env)
if hasattr(sampler, 'envs') and sampler.envs is not None:
    state_dim = int(np.prod(sampler.envs.observation_space.shape))
    action_dim = int(np.prod(sampler.envs.action_space.shape))
else:
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
print('state dim: %d; action dim: %d'%(state_dim, action_dim))

### generate a uniform policy to collect samples for environment parameterization
# Handle num_workers=0 case (single env)
if hasattr(sampler, 'envs') and sampler.envs is not None:
    action_l, action_h = sampler.envs.action_space.low, sampler.envs.action_space.high
else:
    action_l, action_h = env.action_space.low, env.action_space.high
policy_uni = UniformPolicy(state_dim, action_dim, low=action_l, high=action_h)


################ in the initial time period, nominal model ####################
if not os.path.exists(args.model_path): os.makedirs(args.model_path)
task = tasks[0]
print('The nominal task: ', task) 
sampler.reset_task(task)

### generate a batch of episodes to collect samples for environment parameterization
def collect_episodes():
    episodes = []
    for _ in range(args.et_length): 
        episodes.append(sampler.sample(policy_uni, device=device))
    return episodes

### constrict the input-output pairs w.r.t. the environment model accroding to the type
episodes = collect_episodes()
inputs, outputs = construct_env_io(episodes, env_type=ENV_TYPE, H=args.H)

### construct the MLP of the environment model
env_model = EnvModel(inputs.shape[1], outputs.shape[1], 
        hidden_sizes=(args.env_hidden_size,) * args.env_num_layers).to(device=device)

### train the environment model for the nominal task
print('Training the nominal environment model...')
env_model, tloss = env_nominal_train(env_model, inputs, outputs, device=device)

### initialize the Dirichlet  mixture of environment models
env_models = [env_model]

### initilize the CRP prior distribution
crp = CRP(zeta=ZETA)


'''
We train a universal model for the initialization of new environment models.
In principle, the new environment models can be initialized in any way, 
e.g., randomly initialization.

We use a fixed number of periods (max 10) to avoid overfitting when num_periods is small,
and to ensure the universal model is representative but not too specific.
'''
# Use fixed number of periods for universal model (max 10) to avoid overfitting
# If num_periods is small, use fewer periods to avoid using all data
universal_model_periods = min(10, max(3, args.num_periods // 2))
print(f'Training universal model on {universal_model_periods} periods (out of {args.num_periods} total)')

epi_list = []
for idx in range(universal_model_periods):
    task = tasks[idx]; sampler.reset_task(task)
    episodes = sampler.sample(policy_uni, device=device)
    epi_list.append(episodes)
inputs, outputs = construct_env_io(epi_list, env_type=ENV_TYPE, H=args.H)
env_model_init = EnvModel(inputs.shape[1], outputs.shape[1], 
        hidden_sizes=(args.env_hidden_size,) * args.env_num_layers).to(device=device)
print('Training the universal model for initialization of new environment models...')
env_model_init, tloss = env_nominal_train(env_model_init, inputs, outputs, device=device)


########## in the following time periods, dynamic environments ################ 
tasks = tasks[:args.num_periods]

### the id of the sequential tasks, the id of the first task is 1
task_ids = np.zeros((tasks.shape[0], 1)); task_ids[0] = 1

# Track detailed clustering history for analysis
clustering_history = {
    'periods': [],
    'tasks': [],
    'initial_likelihoods': [],      # llls before normalization
    'normalized_likelihoods': [],   # llls after softmax
    'priors': [],                   # prior distributions
    'posteriors': [],               # posterior distributions
    'prior_selections': [],         # l_post choices
    'new_cluster_created': [],      # boolean: new cluster created?
    'em_likelihoods_history': [],   # EM step likelihoods
    'em_posteriors_history': [],    # EM step posteriors
    'final_likelihoods': [],        # updated_llls
    'final_posteriors': [],         # final posterior
    'cluster_assignments': [],      # l_star
    'prior_evolution': [],         # prior at each period
    'num_clusters': [],             # L at each period
    'num_samples': []               # number of samples collected
}

for period in range(1, args.num_periods):
    print('\n----------- Time period %d--------------'%period)
    
    # ===== DYNAMIC ZETA: Giảm dần theo thời gian =====
    # Bắt đầu cao (0.7) để tạo clusters sớm, giảm dần (0.3) để ổn định
    if period == 1:
        # Period 1: ZETA cao để dễ tạo cluster mới
        dynamic_zeta = 0.7
    else:
        # Giảm dần: 0.7 → 0.3
        dynamic_zeta = 0.7 - 0.1 * (period - 1) / max(1, args.num_periods - 2)
        dynamic_zeta = max(0.3, min(0.7, dynamic_zeta))
    
    # Update CRP zeta (cần update prior tương ứng)
    if period > 1:
        crp._zeta = dynamic_zeta
        print(f'Using dynamic ZETA = {dynamic_zeta:.2f} for period {period}')
    
    L = crp._L; prior = crp._prior
    
    task = tasks[period]
    print('Task information', task) 
    sampler.reset_task(task)

    episodes = collect_episodes()
    inputs, outputs = construct_env_io(episodes, env_type=ENV_TYPE, H=args.H)

    # Track number of samples
    num_samples = inputs.shape[0]
    
    # Track current state
    clustering_history['periods'].append(period)
    clustering_history['tasks'].append(task.tolist())
    clustering_history['num_samples'].append(num_samples)
    clustering_history['num_clusters'].append(int(L))
    clustering_history['prior_evolution'].append(prior.copy().tolist())

    ### create a potentially new environment model
    env_model_new = EnvModel(inputs.shape[1], outputs.shape[1], 
            hidden_sizes=(args.env_hidden_size,) * args.env_num_layers).to(device=device)
    env_model_new.load_state_dict(env_model_init.state_dict())

    ### predictive likelihood of the collected samples, including the empty new model
    llls = np.zeros(L+1)
    for idx in range(L):
        llls[idx] = compute_likelihood(env_models[idx], inputs, outputs, sigma=SIGMA)
    llls[-1] = compute_likelihood(env_model_new, inputs, outputs, sigma=SIGMA)
    
    # Store initial likelihoods (before normalization)
    clustering_history['initial_likelihoods'].append(llls.copy().tolist())
    
    llls_normalized = softmax_normalize(llls, temperature=TAU1)
    clustering_history['normalized_likelihoods'].append(llls_normalized.copy().tolist())
    print('Predictive likelihood: ', llls_normalized)
    
    prior_normalized = softmax_normalize(prior, temperature=TAU2)
    clustering_history['priors'].append(prior_normalized.copy().tolist())
    print('Prior distribution: ', prior_normalized)

    posterior = llls_normalized * prior_normalized[:llls_normalized.shape[0]]
    posterior = softmax_normalize(posterior)
    clustering_history['posteriors'].append(posterior.copy().tolist())
    print('Posterior over environment models: ', posterior)
    l_post = np.argmax(posterior) + 1
    clustering_history['prior_selections'].append(int(l_post))
    clustering_history['new_cluster_created'].append(bool(l_post == L+1))
    print('Posterior selection: %d'%l_post)
    if l_post == L+1:
        print('Add a new cluster...')
        env_models.append(env_model_new)

    ### update the CRP prior distribution using the posterior selection
    crp.update(l_post)

    def Estep(env_models, inputs, outputs):
        ### the Expectation-step, compute the posterior of environment-to-cluster assignment
        llls = np.zeros(len(env_models))
        for idx in range(len(env_models)):
            llls[idx] = compute_likelihood(env_models[idx], inputs, outputs, sigma=SIGMA)
        llls = softmax_normalize(llls, temperature=100)
        
        # Safe prior slicing with validation
        prior_len = min(len(prior), llls.shape[0])
        if prior_len < llls.shape[0]:
            # Pad prior with last value if needed (shouldn't happen in normal operation)
            prior_slice = np.concatenate([prior[:prior_len], np.full(llls.shape[0] - prior_len, prior[-1] if len(prior) > 0 else 0.5)])
        else:
            prior_slice = prior[:llls.shape[0]]
        
        posterior = llls * prior_slice
        posterior = softmax_normalize(posterior, temperature=100)
        return llls, posterior
    
    def Mstep(env_models, inputs, outputs, posterior):
        ### the Maximization-step, update parameters of the mixture model
        for idx in range(len(env_models)):
            env_model = env_models[idx]
            env_model, _ = env_update(env_model, inputs, outputs, 
                    posterior=posterior[idx], device=device)
            env_models[idx] = env_model
        return env_models

    print('********** EM Algorithm **********')
    em_likelihoods_steps = []
    em_posteriors_steps = []
    for em_step in range(EM_STEPS):
        llls, posterior = Estep(env_models, inputs, outputs)
        em_likelihoods_steps.append(llls.copy().tolist())
        em_posteriors_steps.append(posterior.copy().tolist())
        print('Predictive likelihood: ', llls)
        print('Posterior: ', posterior)
        env_models = Mstep(env_models, inputs, outputs, posterior)
    clustering_history['em_likelihoods_history'].append(em_likelihoods_steps)
    clustering_history['em_posteriors_history'].append(em_posteriors_steps)
    print('********** EM Algorithm **********')

    ### compute the final assignment of the current environment
    updated_llls, final_posterior = Estep(env_models, inputs, outputs)
    clustering_history['final_likelihoods'].append(updated_llls.copy().tolist())
    clustering_history['final_posteriors'].append(final_posterior.copy().tolist())
    l_star = np.argmax(updated_llls) + 1
    task_ids[period] = l_star
    clustering_history['cluster_assignments'].append(int(l_star))
    print('Updated likelihood: ', updated_llls)
    print('Choosing the cluster %d'%l_star)

task_info = np.concatenate((tasks, task_ids), axis=1)
np.save(os.path.join(args.model_path, 'task_info.npy'), task_info)

# Also save tasks separately for reproducibility
import json
tasks_info = {
    'num_periods': args.num_periods,
    'tasks': tasks.tolist(),
    'task_ids': task_ids.flatten().tolist(),
    'seed': args.seed
}
tasks_path = os.path.join(args.model_path, 'tasks_info.json')
with open(tasks_path, 'w') as f:
    json.dump(tasks_info, f, indent=2)
print(f'Saved tasks info to {tasks_path}')

# Save environment models library (with policy placeholders)
print('\nSaving environment models library...')
env_models_path = os.path.join(args.model_path, 'env_models.pth')
torch.save({
    'env_models': [env_model.state_dict() for env_model in env_models],
    'policies': [None] * len(env_models),  # Placeholder for policies (will be filled in policy_training.py)
    'num_models': len(env_models),
    'input_size': inputs.shape[1],
    'output_size': outputs.shape[1],
    'hidden_sizes': (args.env_hidden_size,) * args.env_num_layers,
    'state_dim': state_dim,  # Save for policy creation later
    'action_dim': action_dim  # Save for policy creation later
}, env_models_path)
print(f'Saved {len(env_models)} environment models to {env_models_path}')
print(f'Note: Policies will be added during policy training phase')

# Save universal initialization model
env_model_init_path = os.path.join(args.model_path, 'env_model_init.pth')
torch.save(env_model_init.state_dict(), env_model_init_path)
print(f'Saved universal initialization model to {env_model_init_path}')

# Save CRP state
import pickle
crp_path = os.path.join(args.model_path, 'crp_state.pkl')
with open(crp_path, 'wb') as f:
    pickle.dump({
        'zeta': crp._zeta,
        'L': crp._L,
        't': crp._t,
        'prior': crp._prior.tolist()  # Convert to list for JSON compatibility
    }, f)
print(f'Saved CRP state to {crp_path}')

# Save clustering summary
import json
clustering_summary = {
    'total_periods': args.num_periods,
    'num_clusters': int(crp._L),
    'zeta': float(crp._zeta),
    'final_prior': crp._prior.tolist(),
    'cluster_assignments': task_ids.flatten().tolist(),
    'task_params': tasks.tolist(),
    'env_type': ENV_TYPE,
    'H': args.H,
    'sigma': SIGMA,
    'tau1': TAU1,
    'tau2': TAU2,
    'em_steps': EM_STEPS,
    'clustering_time_minutes': float((time.time() - start_time) / 60.0)
}
summary_path = os.path.join(args.model_path, 'clustering_summary.json')
with open(summary_path, 'w') as f:
    json.dump(clustering_summary, f, indent=2)
print(f'Saved clustering summary to {summary_path}')

# Save cluster statistics
cluster_stats = {}
for cluster_id in range(1, int(crp._L) + 1):
    periods_in_cluster = np.where(task_ids.flatten() == cluster_id)[0]
    cluster_stats[f'cluster_{cluster_id}'] = {
        'num_periods': len(periods_in_cluster),
        'periods': (periods_in_cluster + 1).tolist(),  # 1-indexed
        'task_params': tasks[periods_in_cluster].tolist()
    }
stats_path = os.path.join(args.model_path, 'cluster_statistics.json')
with open(stats_path, 'w') as f:
    json.dump(cluster_stats, f, indent=2)
print(f'Saved cluster statistics to {stats_path}')

# Save detailed clustering history
# Convert numpy types to Python native types for JSON serialization
def convert_to_native(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    else:
        return obj

history_path = os.path.join(args.model_path, 'clustering_history.json')
with open(history_path, 'w') as f:
    json.dump(convert_to_native(clustering_history), f, indent=2)
print(f'Saved detailed clustering history to {history_path}')

# Save convergence metrics
convergence_metrics = {
    'likelihood_convergence': [],
    'posterior_stability': [],
    'cluster_stability': []
}
for period_idx in range(len(clustering_history['em_likelihoods_history'])):
    em_llls = clustering_history['em_likelihoods_history'][period_idx]
    if len(em_llls) > 1:
        # Measure convergence: change in likelihoods between last two EM steps
        llls_diff = np.abs(np.array(em_llls[-1]) - np.array(em_llls[-2]))
        convergence_metrics['likelihood_convergence'].append(float(llls_diff.max()))
        
        # Posterior stability
        em_posts = clustering_history['em_posteriors_history'][period_idx]
        if len(em_posts) > 1:
            post_diff = np.abs(np.array(em_posts[-1]) - np.array(em_posts[-2]))
            convergence_metrics['posterior_stability'].append(float(post_diff.max()))
    
    # Cluster stability: check if assignment changed during EM
    if len(em_llls) > 0:
        initial_assignment = np.argmax(clustering_history['normalized_likelihoods'][period_idx])
        final_assignment = clustering_history['cluster_assignments'][period_idx] - 1
        convergence_metrics['cluster_stability'].append(bool(initial_assignment == final_assignment))

convergence_path = os.path.join(args.model_path, 'convergence_metrics.json')
with open(convergence_path, 'w') as f:
    json.dump(convert_to_native(convergence_metrics), f, indent=2)
print(f'Saved convergence metrics to {convergence_path}')

print('Running time: %.2f minutes.'%((time.time()-start_time)/60.0))

