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
import xml.etree.ElementTree as ET
import tempfile
import shutil
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
    def __init__(self, sumo_config_path, max_steps=3600, yellow_time=3, green_time_min=10, green_time_max=60, use_gui=False):
        super(SUMOEnv, self).__init__()
        
        self._base_config_path = sumo_config_path
        self.sumo_config_path = sumo_config_path
        self.max_steps = max_steps
        self.yellow_time = yellow_time
        self.green_time_min = green_time_min
        self.green_time_max = green_time_max
        self.use_gui = use_gui  # Flag to use GUI
        
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
        
        # For reward shaping: track previous values
        self._prev_queue = 0.0
        self._prev_speed = 0.0
        
        # Early termination threshold
        self._early_termination_threshold = 15.0  # vehicles per lane
        
        # Route file management for dynamic traffic intensity
        self._original_route_file = None
        self._temp_route_file = None
        self._temp_config_file = None
        self._route_file_modified = False
        
        # Parse config to get route file path
        self._parse_config_for_route_file()
        
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
    
    def _parse_config_for_route_file(self):
        """Parse SUMO config to get route file path"""
        try:
            tree = ET.parse(self._base_config_path)
            root = tree.getroot()
            
            # Find route-files in config
            route_files_elem = root.find('.//route-files')
            if route_files_elem is not None:
                route_file_name = route_files_elem.get('value')
                # Create absolute path
                config_dir = os.path.dirname(os.path.abspath(self._base_config_path))
                self._original_route_file = os.path.join(config_dir, route_file_name)
                if not os.path.exists(self._original_route_file):
                    print(f"[WARNING] Route file not found: {self._original_route_file}")
                    self._original_route_file = None
            else:
                print("[WARNING] No route-files found in SUMO config")
        except Exception as e:
            print(f"[WARNING] Could not parse config: {e}")
            self._original_route_file = None
    
    def reset_task(self, task):
        """
        Reset environment task parameters and generate new route file
        task: array of [traffic_intensity, route_variation, ...]
        """
        old_intensity = self._traffic_intensity
        old_variation = self._route_variation
        
        if len(task) >= 1:
            self._traffic_intensity = float(task[0])
        if len(task) >= 2:
            self._route_variation = float(task[1])
        
        # Always generate route file when reset_task is called to ensure fresh file
        # This avoids issues with file caching or reuse
        if self._original_route_file and os.path.exists(self._original_route_file):
            self._generate_route_file()
            self._route_file_modified = True
            print(f"[INFO] Generated route file: traffic_intensity={self._traffic_intensity:.3f}, "
                  f"route_variation={self._route_variation:.3f}")
        else:
            print("[WARNING] Cannot generate route file: original route file not found")
            self._route_file_modified = False
    
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
    
    def _generate_route_file(self):
        """Generate route file with traffic intensity and route variation applied"""
        try:
            # Parse original route file
            tree = ET.parse(self._original_route_file)
            root = tree.getroot()
            
            # Get all routes for route variation
            routes = root.findall('route')
            route_ids = [r.get('id') for r in routes] if routes else []
            
            # Scale all flows according to traffic_intensity and route_variation
            flows_modified = 0
            for flow in root.findall('flow'):
                # Scale vehsPerHour by traffic_intensity
                if 'vehsPerHour' in flow.attrib:
                    old_rate = float(flow.get('vehsPerHour'))
                    new_rate = old_rate * self._traffic_intensity
                    flow.set('vehsPerHour', str(new_rate))
                    flows_modified += 1
                
                # Scale probability by traffic_intensity (if exists)
                if 'probability' in flow.attrib:
                    old_prob = float(flow.get('probability'))
                    new_prob = old_prob * self._traffic_intensity
                    # Clamp probability to [0, 1]
                    new_prob = min(1.0, max(0.0, new_prob))
                    flow.set('probability', str(new_prob))
                    flows_modified += 1
                
                # Apply route variation: randomly change route assignment
                if self._route_variation > 0 and 'route' in flow.attrib and len(route_ids) > 0:
                    current_route = flow.get('route')
                    # With probability = route_variation, change to random route
                    if np.random.random() < self._route_variation:
                        new_route = np.random.choice(route_ids)
                        flow.set('route', new_route)
            
            # Save temporary route file
            if self._temp_route_file and os.path.exists(self._temp_route_file):
                try:
                    os.remove(self._temp_route_file)
                except:
                    pass
            
            config_dir = os.path.dirname(self._original_route_file)
            # Include traffic_intensity and route_variation in filename to ensure uniqueness
            # Use hash to avoid filesystem issues with special characters
            task_hash = hash((self._traffic_intensity, self._route_variation)) % (10**8)
            temp_route_name = f"route_temp_{os.getpid()}_{id(self)}_{task_hash}.rou.xml"
            self._temp_route_file = os.path.join(config_dir, temp_route_name)
            
            # Write XML with declaration
            tree.write(self._temp_route_file, encoding='UTF-8', xml_declaration=True)
            
            # Create temporary config file with new route file
            self._create_temp_config_file()
            
        except Exception as e:
            print(f"[ERROR] Could not generate route file: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: use original route file
            self._temp_route_file = None
            self.sumo_config_path = self._base_config_path
            self._route_file_modified = False
    
    def _create_temp_config_file(self):
        """Create temporary SUMO config file with modified route file"""
        try:
            # Parse original config
            tree = ET.parse(self._base_config_path)
            root = tree.getroot()
            
            # Update route-files path
            route_files_elem = root.find('.//route-files')
            if route_files_elem is not None:
                # Get filename from temp route file
                route_file_name = os.path.basename(self._temp_route_file)
                route_files_elem.set('value', route_file_name)
            else:
                print("[WARNING] No route-files element found in config")
            
            # Save temporary config file
            if self._temp_config_file and os.path.exists(self._temp_config_file):
                try:
                    os.remove(self._temp_config_file)
                except:
                    pass
            
            config_dir = os.path.dirname(self._base_config_path)
            temp_config_name = f"config_temp_{os.getpid()}_{id(self)}.sumocfg"
            self._temp_config_file = os.path.join(config_dir, temp_config_name)
            
            tree.write(self._temp_config_file, encoding='UTF-8', xml_declaration=True)
            
            # Update sumo_config_path to use temp config
            self.sumo_config_path = self._temp_config_file
            
        except Exception as e:
            print(f"[ERROR] Could not create temp config: {e}")
            self.sumo_config_path = self._base_config_path
    
    def _start_sumo(self):
        """Start SUMO simulation"""
        if self.sumo_running:
            traci.close()
        
        # Use GUI if requested
        if self.use_gui:
            try:
                sumo_binary = sumolib.checkBinary('sumo-gui')
            except:
                print("[WARNING] sumo-gui not found, falling back to sumo")
                sumo_binary = sumolib.checkBinary('sumo')
        else:
            try:
                sumo_binary = sumolib.checkBinary('sumo')
            except:
                sumo_binary = sumolib.checkBinary('sumo-gui')
        
        sumo_cmd = [sumo_binary, "-c", self.sumo_config_path, 
                   "--no-step-log", "true", 
                   "--no-warnings", "true"]
        
        # Only add --quit-on-end if not using GUI (GUI should stay open)
        if not self.use_gui:
            sumo_cmd.append("--quit-on-end")
            sumo_cmd.append("true")
        
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
        
        try:
            # Get all controlled lanes
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException, Exception) as e:
            # Connection lost, return zero observation
            print(f"[WARNING] Error getting controlled lanes: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        for lane_id in controlled_lanes:
            try:
                # Queue length (vehicles with speed < 0.1 m/s)
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                
                # Waiting time (sum of waiting times of all vehicles)
                waiting_time = traci.lane.getWaitingTime(lane_id)
                
                # Vehicle count
                vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                
                obs.extend([queue_length, waiting_time, vehicle_count])
            except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException, Exception) as e:
                # Connection lost for this lane, use zeros
                obs.extend([0.0, 0.0, 0.0])
        
        # Pad if necessary
        while len(obs) < self.observation_space.shape[0]:
            obs.extend([0.0, 0.0, 0.0])
        
        obs = np.array(obs[:self.observation_space.shape[0]], dtype=np.float32)
        
        # Improved normalization: Min-Max normalization with reasonable bounds for SUMO
        # Typical ranges: queue_length (0-50), waiting_time (0-3600s), vehicle_count (0-50)
        # We normalize each feature type separately for better stability
        obs_normalized = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Normalize each feature type (queue_length, waiting_time, vehicle_count) separately
        # Process in groups of 3 (one per lane)
        for i in range(0, min(len(obs), len(obs_normalized)), 3):
            if i + 2 < len(obs):
                # Queue length: normalize to [0, 1] with max=50
                queue_max = 50.0
                obs_normalized[i] = np.clip(obs[i] / (queue_max + 1e-8), 0.0, 1.0)
                
                # Waiting time: normalize to [0, 1] with max=3600s (1 hour)
                waiting_max = 3600.0
                obs_normalized[i + 1] = np.clip(obs[i + 1] / (waiting_max + 1e-8), 0.0, 1.0)
                
                # Vehicle count: normalize to [0, 1] with max=50
                vehicle_max = 50.0
                obs_normalized[i + 2] = np.clip(obs[i + 2] / (vehicle_max + 1e-8), 0.0, 1.0)
            elif i < len(obs):
                # Handle incomplete triplets (shouldn't happen, but safe)
                obs_normalized[i] = np.clip(obs[i] / (50.0 + 1e-8), 0.0, 1.0)
        
        # Remaining elements are already zeros (from initialization)
        return obs_normalized
    
    def _get_reward(self):
        """Calculate reward based on traffic performance (optimized for training)"""
        if self.tl_id is None:
            return 0.0
        
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException, Exception) as e:
            # Connection lost, return zero reward
            return 0.0
        
        reward = 0.0
        
        total_waiting = 0.0
        total_queue = 0.0
        total_vehicles = 0.0
        total_speed = 0.0
        lane_count = 0
        
        for lane_id in controlled_lanes:
            try:
                # Waiting time (penalty for vehicles waiting)
                waiting_time = traci.lane.getWaitingTime(lane_id)
                total_waiting += waiting_time
                
                # Queue length (penalty for stopped vehicles)
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                total_queue += queue_length
                
                # Vehicle count (positive for throughput)
                vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                total_vehicles += vehicle_count
                
                # Average speed (positive for flow efficiency)
                mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                if mean_speed > 0:
                    total_speed += mean_speed
                
                lane_count += 1
            except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException, Exception) as e:
                # Connection lost for this lane, skip it
                continue
        
        if lane_count > 0:
            # Normalize by number of lanes for stability
            avg_waiting = total_waiting / lane_count
            avg_queue = total_queue / lane_count
            avg_vehicles = total_vehicles / lane_count
            avg_speed = total_speed / lane_count if total_speed > 0 else 0.0
            
            # ===== IMPROVED REWARD SHAPING =====
            # 1. Penalty for waiting time (tăng weight)
            reward -= avg_waiting * 0.1  # Tăng từ 0.05 → 0.1
            
            # 2. Penalty for queue (tăng weight, thêm exponential penalty)
            reward -= avg_queue * 0.8  # Tăng từ 0.5 → 0.8
            if avg_queue > 5.0:
                # Exponential penalty cho queue dài
                reward -= (avg_queue - 5.0) ** 1.5 * 2.0
            
            # 3. Reward for throughput
            reward += avg_vehicles * 0.2
            
            # 4. Reward for speed (tăng weight)
            if avg_speed > 0:
                speed_reward = (avg_speed / 20.0) * 0.5  # Tăng từ 0.3 → 0.5
                reward += speed_reward
            
            # 5. BONUS lớn hơn cho low congestion
            if avg_queue < 2.0 and avg_speed > 5.0:
                reward += 5.0  # Tăng từ 1.0 → 5.0
            
            # 6. BONUS mới: Reward cho việc giảm queue
            if hasattr(self, '_prev_queue'):
                queue_reduction = self._prev_queue - avg_queue
                if queue_reduction > 0:
                    reward += queue_reduction * 2.0  # Bonus cho giảm queue
            self._prev_queue = avg_queue
            
            # 7. BONUS mới: Reward cho việc tăng speed
            if hasattr(self, '_prev_speed'):
                speed_increase = avg_speed - self._prev_speed
                if speed_increase > 0:
                    reward += speed_increase * 0.5  # Bonus cho tăng speed
            self._prev_speed = avg_speed
        
        # Clip reward to prevent extreme values (mở rộng range)
        reward = np.clip(float(reward), -300.0, 300.0)
        
        return reward
    
    def reset(self, env=True):
        """Reset environment"""
        # Ensure route file is generated if traffic intensity changed
        if self._route_file_modified and self._original_route_file:
            if not self._temp_route_file or not os.path.exists(self._temp_route_file):
                self._generate_route_file()
        
        if not self.sumo_running:
            self._start_sumo()
        else:
            # Reload with new config (which may have new route file)
            try:
                traci.load(["-c", self.sumo_config_path])
            except Exception as e:
                print(f"[WARNING] Could not reload SUMO config: {e}")
                # Fallback: close and restart
                traci.close()
                self.sumo_running = False
                self._start_sumo()
        
        self.step_count = 0
        self.current_phase = 0
        
        # Set initial phase
        if self.tl_id:
            traci.trafficlight.setPhase(self.tl_id, 0)
        
        # Advance simulation a few steps
        try:
            for _ in range(5):
                traci.simulationStep()
        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException, Exception) as e:
            # Connection lost during reset, try to recover
            print(f"[WARNING] SUMO connection lost during reset: {e}. Attempting to recover...")
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False
            # Restart SUMO
            self._start_sumo()
            # Retry advancing simulation
            for _ in range(5):
                traci.simulationStep()
        
        # Reset tracking variables for reward shaping
        self._prev_queue = 0.0
        self._prev_speed = 0.0
        
        observation = self._get_observation()
        return observation
    
    def step(self, action):
        """
        Execute action in environment
        action: array of phase durations for each phase
        """
        if not self.sumo_running:
            raise RuntimeError("SUMO not running. Call reset() first.")
        
        # Check if SUMO connection is still alive
        try:
            traci.simulation.getTime()
        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException, Exception) as e:
            # Connection lost, try to recover
            print(f"[WARNING] SUMO connection lost during step: {e}. Attempting to recover...")
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False
            # Try to restart
            try:
                self._start_sumo()
                print("[INFO] SUMO connection recovered")
            except Exception as e2:
                raise RuntimeError(f"Failed to recover SUMO connection: {e2}")
        
        # Clip action to valid range
        action = np.clip(action, self.green_time_min, self.green_time_max)
        
        # Get current phase with error handling
        try:
            current_phase = traci.trafficlight.getPhase(self.tl_id)
        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException, Exception) as e:
            print(f"[WARNING] Error getting phase: {e}. Resetting SUMO...")
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False
            self._start_sumo()
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
            try:
                traci.simulationStep()
                self.step_count += 1
                phase_steps += 1
                
                # Collect reward
                try:
                    total_reward += self._get_reward()
                except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException, Exception) as e:
                    # Connection lost during reward collection, try to recover
                    print(f"[WARNING] SUMO connection lost during reward collection: {e}")
                    try:
                        traci.close()
                    except:
                        pass
                    self.sumo_running = False
                    # Restart SUMO
                    self._start_sumo()
                    # Continue with zero reward for this step
                    total_reward += 0.0
                
                # Check if done
                if self.step_count >= self.max_steps:
                    break
            except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException, Exception) as e:
                # Connection lost during simulation step, try to recover
                print(f"[WARNING] SUMO connection lost during simulation step: {e}. Attempting to recover...")
                try:
                    traci.close()
                except:
                    pass
                self.sumo_running = False
                # Restart SUMO
                try:
                    self._start_sumo()
                    print("[INFO] SUMO connection recovered, continuing simulation")
                    # Continue from where we left off
                    continue
                except Exception as e2:
                    print(f"[ERROR] Failed to recover SUMO connection: {e2}")
                    # Return current state with zero reward
                    observation = self._get_observation() if self.sumo_running else np.zeros(self.observation_space.shape, dtype=np.float32)
                    return observation, total_reward, True, False, {'error': str(e2), 'step_count': self.step_count}
        
        # Get observation with error handling
        try:
            observation = self._get_observation()
        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException, Exception) as e:
            print(f"[WARNING] Error getting observation: {e}")
            # Return zero observation if connection lost
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            # Mark as done since we can't continue
            done = True
        else:
            # Done if max steps reached
            done = self.step_count >= self.max_steps
        
        # ===== EARLY TERMINATION khi tắc đường nặng =====
        # Kiểm tra nếu queue quá dài → terminate sớm (giống DDQN)
        early_terminated = False
        if not done and self.tl_id:
            try:
                controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
                total_queue = 0.0
                lane_count = 0
                
                for lane_id in controlled_lanes:
                    try:
                        queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                        total_queue += queue_length
                        lane_count += 1
                    except:
                        continue
                
                if lane_count > 0:
                    avg_queue = total_queue / lane_count
                    # Early termination nếu queue quá dài (tắc đường nặng)
                    if avg_queue > self._early_termination_threshold:
                        done = True
                        early_terminated = True
                        # Thêm penalty cho early termination do tắc đường
                        total_reward -= 50.0
            except:
                pass  # Nếu có lỗi, tiếp tục bình thường
        
        info = {
            'step_count': self.step_count,
            'phase': current_phase,
            'phase_duration': phase_duration,
            'early_termination': early_terminated
        }
        if early_terminated:
            info['termination_reason'] = 'congestion'
        
        # Gym new API: return (obs, reward, terminated, truncated, info)
        # Map done to terminated and truncated
        terminated = done
        truncated = False  # Not truncated by time limit (handled by wrapper)
        return observation, total_reward, terminated, truncated, info
    
    def close(self):
        """Close SUMO connection and cleanup temp files"""
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False
        
        # Cleanup temporary files
        if self._temp_route_file and os.path.exists(self._temp_route_file):
            try:
                os.remove(self._temp_route_file)
            except:
                pass
        
        if self._temp_config_file and os.path.exists(self._temp_config_file):
            try:
                os.remove(self._temp_config_file)
            except:
                pass

