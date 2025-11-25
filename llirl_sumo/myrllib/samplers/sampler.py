import gym
import torch
import multiprocessing as mp
import numpy as np
from torch.distributions import Uniform 
from myrllib.envs.subproc_vec_env import SubprocVecEnv
from myrllib.episodes.episode import BatchEpisodes

def make_env(env_name, sumo_config_path=None, seed=0, use_gui=False):
    def _make_env():
        env = gym.make(env_name, sumo_config_path=sumo_config_path, use_gui=use_gui)
        env.seed(seed)
        return env
    return _make_env

class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=1, seed=0, sumo_config_path=None, use_gui=False):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sumo_config_path = sumo_config_path
        self.use_gui = use_gui
        
        self._env = gym.make(env_name, sumo_config_path=sumo_config_path, use_gui=use_gui)
        
        # Handle num_workers=0 (no multiprocessing, e.g., on Windows)
        if num_workers == 0:
            self.envs = None  # Will use single env directly
            self.use_multiprocessing = False
        else:
            self.queue = mp.Queue()
            self.envs = SubprocVecEnv([make_env(env_name, sumo_config_path, seed=seed, use_gui=use_gui) for _ in range(num_workers)],
                queue=self.queue)
            self.use_multiprocessing = True

    def sample(self, policy, params=None, gamma=0.95, device='cpu', recurrent=False, seq_len=5):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        
        if not self.use_multiprocessing:
            # Single environment mode (num_workers=0)
            s_traj, a_traj, r_traj, ns_traj, id_traj = [], [], [], [], []
            for batch_id in range(self.batch_size):
                obs = self._env.reset()
                done = False
                episode_obs, episode_acts, episode_rews, episode_next_obs = [], [], [], []
                
                while not done:
                    with torch.no_grad():
                        obs_tensor = torch.from_numpy(obs).float().to(device)
                        if len(obs_tensor.shape) == 1:
                            obs_tensor = obs_tensor.unsqueeze(0)
                        pi = policy(obs_tensor)
                        action_tensor = pi.sample()
                        action = action_tensor.cpu().numpy()[0] if action_tensor.shape[0] == 1 else action_tensor.cpu().numpy()
                    
                    next_obs, reward, terminated, truncated, info = self._env.step(action)
                    done = terminated or truncated
                    episode_obs.append(obs)
                    episode_acts.append(action)
                    episode_rews.append(reward)
                    episode_next_obs.append(next_obs)
                    obs = next_obs
                
                # Append episode to batch
                for i in range(len(episode_obs)):
                    episodes.append(
                        np.array([episode_obs[i]]), 
                        np.array([episode_acts[i]]), 
                        np.array([episode_rews[i]]), 
                        np.array([episode_next_obs[i]]), 
                        [batch_id]
                    )
                    s_traj.append(np.array([[episode_obs[i]]]))
                    a_traj.append(np.array([[episode_acts[i]]]))
                    r_traj.append(np.array([[episode_rews[i]]]))
                    ns_traj.append(np.array([[episode_next_obs[i]]]))
                    id_traj.append(np.array([[batch_id]]))
            
            if s_traj:
                s_traj = np.concatenate(s_traj, axis=1) if len(s_traj) > 0 else np.array([])
                a_traj = np.concatenate(a_traj, axis=1) if len(a_traj) > 0 else np.array([])
                r_traj = np.concatenate(r_traj, axis=1) if len(r_traj) > 0 else np.array([])
                ns_traj = np.concatenate(ns_traj, axis=1) if len(ns_traj) > 0 else np.array([])
                id_traj = np.concatenate(id_traj, axis=1) if len(id_traj) > 0 else np.array([])
                episodes.append_traj(s_traj, a_traj, r_traj, ns_traj, id_traj)
            return episodes
        
        # Multiprocessing mode (num_workers > 0)
        for i in range(self.batch_size): self.queue.put(i)
        for _ in range(self.num_workers): self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        obs_hist = [observations]
        s_traj, a_traj, r_traj, ns_traj, id_traj = [], [], [], [], []
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                if recurrent:
                    obs_seq = np.stack(obs_hist)[-seq_len:]
                    obs_seq = torch.from_numpy(obs_seq).float().to(device)
                    pi = policy(obs_seq)
                else:
                    obs_tensor = torch.from_numpy(observations).float().to(device)
                    pi = policy(obs_tensor)

                actions_tensor = pi.sample()
                actions = actions_tensor.cpu().numpy()
            new_obs, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, new_obs, batch_ids)

            s_traj.append(np.expand_dims(observations, axis=1))
            a_traj.append(np.expand_dims(actions, axis=1))
            r_traj.append(np.expand_dims(rewards, axis=1))
            ns_traj.append(np.expand_dims(new_obs, axis=1))

            traj_ids = list(new_batch_ids)
            for idx, id_ in enumerate(traj_ids):
                if id_ is None:
                    traj_ids[idx] = np.inf
            traj_ids = np.array(traj_ids).reshape(-1,1)
            id_traj.append(traj_ids)

            observations, batch_ids = new_obs, new_batch_ids
            obs_hist.append(observations)
    
        s_traj = np.concatenate(s_traj, axis=1)
        a_traj = np.concatenate(a_traj, axis=1)
        r_traj = np.concatenate(r_traj, axis=1)
        ns_traj = np.concatenate(ns_traj, axis=1)
        id_traj = np.concatenate(id_traj, axis=1)
        episodes.append_traj(s_traj, a_traj, r_traj, ns_traj, id_traj)
        return episodes
    
    def reset_task(self, task):
        if not self.use_multiprocessing:
            self._env.unwrapped.reset_task(task)
            return True
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def domain_randomization(self):
        tasks = self._env.unwrapped.sample_task(num_tasks=self.num_workers)
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_task(self, num_tasks=1):
        tasks = self._env.unwrapped.sample_task(num_tasks=num_tasks)
        return tasks

