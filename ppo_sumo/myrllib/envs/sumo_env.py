"""
SUMO Environment Wrapper for LLIRL
Implements a gym-compatible interface for SUMO traffic control
"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import os
import sys
try:
    import traci
    import sumolib
except ImportError:
    print("Warning: SUMO libraries not found. Please install SUMO and set SUMO_HOME.")
    traci = None
    sumolib = None
from collections import defaultdict

class SUMOEnv(gym.Env):
    """
    SUMO Environment for traffic light control at single intersection
    """
    def __init__(self, sumo_config_path, max_steps=3600, yellow_time=3, green_time_min=10, green_time_max=60):
        super(SUMOEnv, self).__init__()
        
        self.sumo_config_path = sumo_config_path
        self.max_steps = max_steps
        self.yellow_time = yellow_time
        self.green_time_min = green_time_min
        self.green_time_max = green_time_max
        
        # SUMO connection
        self.sumo_running = False
        self.step_count = 0
        self.current_phase = 0
        
        # Traffic light ID (assuming single intersection)
        self.tl_id = None
        
        # Environment parameters (task parameters for LLIRL)
        # These can change over time to represent different traffic patterns
        self._traffic_intensity = 1.0  # Multiplier for traffic flow
        self._route_variation = 0.0    # Variation in route probabilities
        
        # Observation space: queue lengths, waiting times, vehicle counts per lane
        # Assuming 4 approaches with 3 lanes each = 12 lanes
        num_lanes = 12
        self.observation_space = spaces.Box(
            low=0.0, 
            high=np.inf, 
            shape=(num_lanes * 3,),  # queue_length, waiting_time, vehicle_count per lane
            dtype=np.float32
        )
        
        # Action space: phase duration (green time) for each phase
        # Assuming 4 phases (N-S straight, E-W straight, N-S left, E-W left)
        self.action_space = spaces.Box(
            low=green_time_min, 
            high=green_time_max, 
            shape=(4,), 
            dtype=np.float32
        )
        
        self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset_task(self, task):
        """
        Reset environment task parameters
        task: array of [traffic_intensity, route_variation, ...]
        """
        if len(task) >= 1:
            self._traffic_intensity = float(task[0])
        if len(task) >= 2:
            self._route_variation = float(task[1])
    
    def sample_task(self, num_tasks=1):
        """
        Sample random task parameters
        Returns: array of tasks, each with [traffic_intensity, route_variation]
        """
        tasks = np.random.uniform(0.5, 2.0, size=(num_tasks, 1))  # traffic intensity
        if num_tasks > 1:
            route_vars = np.random.uniform(0.0, 0.5, size=(num_tasks, 1))
            tasks = np.concatenate([tasks, route_vars], axis=1)
        else:
            tasks = np.concatenate([tasks, np.array([[0.0]])], axis=1)
        return tasks
    
    def _start_sumo(self):
        """Start SUMO simulation"""
        if self.sumo_running:
            traci.close()
        
        try:
            sumo_binary = sumolib.checkBinary('sumo')
        except:
            sumo_binary = sumolib.checkBinary('sumo-gui')
        sumo_cmd = [sumo_binary, "-c", self.sumo_config_path, 
                   "--no-step-log", "true", 
                   "--no-warnings", "true",
                   "--quit-on-end", "true"]
        
        traci.start(sumo_cmd)
        self.sumo_running = True
        
        # Get traffic light ID
        tl_ids = traci.trafficlight.getIDList()
        if len(tl_ids) > 0:
            self.tl_id = tl_ids[0]
        else:
            raise ValueError("No traffic light found in SUMO network")
    
    def _get_observation(self):
        """Extract observation from SUMO"""
        obs = []
        
        if self.tl_id is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Get all controlled lanes
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        
        for lane_id in controlled_lanes:
            # Queue length (vehicles with speed < 0.1 m/s)
            queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
            
            # Waiting time (sum of waiting times of all vehicles)
            waiting_time = traci.lane.getWaitingTime(lane_id)
            
            # Vehicle count
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
            
            obs.extend([queue_length, waiting_time, vehicle_count])
        
        # Pad if necessary
        while len(obs) < self.observation_space.shape[0]:
            obs.extend([0.0, 0.0, 0.0])
        
        obs = np.array(obs[:self.observation_space.shape[0]], dtype=np.float32)
        
        # Normalize
        obs = obs / (obs.max() + 1e-8) if obs.max() > 0 else obs
        
        return obs
    
    def _get_reward(self):
        """Calculate reward based on traffic performance"""
        if self.tl_id is None:
            return 0.0
        
        reward = 0.0
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        
        for lane_id in controlled_lanes:
            # Negative reward for waiting time
            waiting_time = traci.lane.getWaitingTime(lane_id)
            reward -= waiting_time * 0.01
            
            # Negative reward for queue length
            queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
            reward -= queue_length * 0.1
            
            # Positive reward for vehicles passing
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
            reward += vehicle_count * 0.05
        
        return float(reward)
    
    def reset(self, env=True):
        """Reset environment"""
        if not self.sumo_running:
            self._start_sumo()
        else:
            traci.load(["-c", self.sumo_config_path])
        
        self.step_count = 0
        self.current_phase = 0
        
        # Set initial phase
        if self.tl_id:
            traci.trafficlight.setPhase(self.tl_id, 0)
        
        # Advance simulation a few steps
        for _ in range(5):
            traci.simulationStep()
        
        observation = self._get_observation()
        return observation
    
    def step(self, action):
        """
        Execute action in environment
        action: array of phase durations for each phase
        """
        if not self.sumo_running:
            raise RuntimeError("SUMO not running. Call reset() first.")
        
        # Clip action to valid range
        action = np.clip(action, self.green_time_min, self.green_time_max)
        
        # Get current phase
        current_phase = traci.trafficlight.getPhase(self.tl_id)
        # Get number of phases from program definition
        try:
            program = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)
            if len(program) > 0:
                num_phases = len(program[0].phases)
            else:
                num_phases = 4  # Default to 4 phases
        except:
            num_phases = 4  # Default to 4 phases
        
        # Determine which phase duration to use
        phase_idx = current_phase % len(action)
        phase_duration = int(action[phase_idx])
        
        # Execute phase
        phase_steps = 0
        total_reward = 0.0
        
        while phase_steps < phase_duration and self.step_count < self.max_steps:
            traci.simulationStep()
            self.step_count += 1
            phase_steps += 1
            
            # Collect reward
            total_reward += self._get_reward()
            
            # Check if done
            if self.step_count >= self.max_steps:
                break
        
        # Get observation
        observation = self._get_observation()
        
        # Done if max steps reached
        done = self.step_count >= self.max_steps
        
        info = {
            'step_count': self.step_count,
            'phase': current_phase,
            'phase_duration': phase_duration
        }
        
        # Gym new API: return (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False
        return observation, total_reward, terminated, truncated, info
    
    def close(self):
        """Close SUMO connection"""
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

