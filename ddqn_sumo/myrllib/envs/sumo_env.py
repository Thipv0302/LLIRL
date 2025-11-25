"""
SUMO Environment Wrapper for DDQN
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
    Discrete action space for DDQN (select phase)
    """
    def __init__(self, sumo_config_path, max_steps=14400, yellow_time=3, green_time_min=10, green_time_max=60):
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
        
        # Environment parameters (task parameters)
        self._traffic_intensity = 1.0
        self._route_variation = 0.0
        
        # Observation space: queue lengths, waiting times, vehicle counts per lane
        num_lanes = 12
        self.observation_space = spaces.Box(
            low=0.0, 
            high=np.inf, 
            shape=(num_lanes * 3,),
            dtype=np.float32
        )
        
        # Discrete action space: select which phase to activate (4 phases)
        self.action_space = spaces.Discrete(4)
        
        self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset_task(self, task):
        """Reset environment task parameters"""
        if len(task) >= 1:
            self._traffic_intensity = float(task[0])
        if len(task) >= 2:
            self._route_variation = float(task[1])
    
    def sample_task(self, num_tasks=1):
        """Sample random task parameters"""
        tasks = np.random.uniform(0.5, 2.0, size=(num_tasks, 1))
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
        
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        
        for lane_id in controlled_lanes:
            queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
            waiting_time = traci.lane.getWaitingTime(lane_id)
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
            obs.extend([queue_length, waiting_time, vehicle_count])
        
        while len(obs) < self.observation_space.shape[0]:
            obs.extend([0.0, 0.0, 0.0])
        
        obs = np.array(obs[:self.observation_space.shape[0]], dtype=np.float32)
        obs = obs / (obs.max() + 1e-8) if obs.max() > 0 else obs
        
        return obs
    
    def _get_reward(self):
        """Calculate reward based on traffic performance"""
        if self.tl_id is None:
            return 0.0
        
        reward = 0.0
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        
        for lane_id in controlled_lanes:
            waiting_time = traci.lane.getWaitingTime(lane_id)
            reward -= waiting_time * 0.01
            queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
            reward -= queue_length * 0.1
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
        
        if self.tl_id:
            traci.trafficlight.setPhase(self.tl_id, 0)
        
        for _ in range(5):
            traci.simulationStep()
        
        observation = self._get_observation()
        return observation
    
    def step(self, action):
        """
        Execute action in environment
        action: discrete action (0-3) selecting which phase to activate
        """
        if not self.sumo_running:
            raise RuntimeError("SUMO not running. Call reset() first.")
        
        # Convert discrete action to phase
        phase = int(action) % 4
        
        # Get actual number of phases from SUMO
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
            num_phases = len(logic.phases)
            # Each phase in SUMO is typically 2 steps (green + yellow), so divide by 2
            actual_num_phases = num_phases // 2
            if actual_num_phases > 0:
                phase = phase % actual_num_phases
        except:
            # Fallback: use 4 phases
            phase = phase % 4
        
        # Set phase duration (fixed green time)
        phase_duration = self.green_time_min + (self.green_time_max - self.green_time_min) // 2
        
        # Execute phase - each phase is 2 steps (green + yellow)
        phase_index = phase * 2
        # Ensure phase index is within valid range
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
            max_phase_index = len(logic.phases) - 1
            phase_index = min(phase_index, max_phase_index)
        except:
            pass
        
        traci.trafficlight.setPhase(self.tl_id, phase_index)
        
        phase_steps = 0
        total_reward = 0.0
        
        while phase_steps < phase_duration and self.step_count < self.max_steps:
            traci.simulationStep()
            self.step_count += 1
            phase_steps += 1
            total_reward += self._get_reward()
            
            if self.step_count >= self.max_steps:
                break
        
        observation = self._get_observation()
        done = self.step_count >= self.max_steps
        
        info = {
            'step_count': self.step_count,
            'phase': phase,
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

