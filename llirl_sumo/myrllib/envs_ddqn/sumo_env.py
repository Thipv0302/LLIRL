"""
SUMO Environment Wrapper for DDQN (đã chỉnh sửa theo yêu cầu)
- Phase đi đúng thứ tự (0 -> 1 -> 2 -> 3 -> ...)
- DDQN CHỈ chọn thời gian xanh cho phase hiện tại (discrete durations)
- Dùng normalization & reward shaping tương tự LLIRL
- Bổ sung xử lý lỗi SUMO (connection lost)
- KHÔNG dùng early stopping (không terminate khi tắc đường nặng)
"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import os
import xml.etree.ElementTree as ET

try:
    import traci
    import sumolib
except ImportError:
    print("Warning: SUMO libraries not found. Please install SUMO and set SUMO_HOME.")
    traci = None
    sumolib = None


class SUMOEnv(gym.Env):
    """
    SUMO Environment for traffic light control at single intersection
    DDQN chọn thời lượng phase, thứ tự phase cố định
    """

    def __init__(
        self,
        sumo_config_path,
        max_steps=4090,
        yellow_time=3,
        green_time_min=10,
        green_time_max=60,
        use_gui=False,
    ):
        super(SUMOEnv, self).__init__()

        self._base_config_path = sumo_config_path
        self.sumo_config_path = sumo_config_path
        self.max_steps = max_steps
        self.yellow_time = yellow_time
        self.green_time_min = green_time_min
        self.green_time_max = green_time_max
        self.use_gui = use_gui  # hỗ trợ GUI

        # SUMO connection
        self.sumo_running = False
        self.step_count = 0
        self.current_phase_idx = 0  # phase hiện tại (index logic trong TLS)
        self.current_phase = 0      # giữ cho thông tin / debug

        # Traffic light ID (assuming single intersection)
        self.tl_id = None

        # Environment parameters (task parameters)
        self._traffic_intensity = 1.0
        self._route_variation = 0.0

        # Route file management for dynamic traffic intensity
        self._original_route_file = None
        self._temp_route_file = None
        self._temp_config_file = None
        self._route_file_modified = False

        # Parse config to get route file path
        self._parse_config_for_route_file()

        # Observation space: queue lengths, waiting times, vehicle counts per lane
        self.num_lanes = 8
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.num_lanes * 3,),
            dtype=np.float32,
        )

        # --------- ACTION SPACE: CHỌN THỜI GIAN CHO PHASE HIỆN TẠI ----------
        # Số phase logic (bạn có thể chỉnh nếu TLS có nhiều hơn 4 phase "xanh")
        self.num_phases = 4

        # Số mức thời lượng rời rạc mà agent có thể chọn
        self.num_durations = 6  # ví dụ: 6 mức trong khoảng [green_time_min, green_time_max]

        # Danh sách thời lượng (theo step) để chọn
        # VD: min=10, max=60, num_durations=6 -> [10,20,30,40,50,60]
        self.durations = np.linspace(
            self.green_time_min,
            self.green_time_max,
            self.num_durations,
            dtype=int
        )

        # Action: chọn 1 trong các thời lượng cho phase hiện tại
        self.action_space = spaces.Discrete(self.num_durations)
        # ---------------------------------------------------------------------

        self.seed()

        # For reward shaping: track previous values (giống LLIRL)
        self._prev_queue = 0.0
        self._prev_speed = 0.0

    # ---------------------------------------------------
    # Utils
    # ---------------------------------------------------
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _parse_config_for_route_file(self):
        """Parse SUMO config file to get route file path"""
        try:
            tree = ET.parse(self._base_config_path)
            root = tree.getroot()

            # Find route-files element
            route_files_elem = root.find(".//route-files")
            if route_files_elem is not None:
                route_file_name = route_files_elem.get("value")
                if route_file_name:
                    config_dir = os.path.dirname(os.path.abspath(self._base_config_path))
                    self._original_route_file = os.path.join(config_dir, route_file_name)
                    if not os.path.exists(self._original_route_file):
                        print(f"[WARNING] Route file not found: {self._original_route_file}")
                        self._original_route_file = None
            else:
                print("[WARNING] No route-files element found in config")
                self._original_route_file = None
        except Exception as e:
            print(f"[WARNING] Could not parse config for route file: {e}")
            self._original_route_file = None

    # ---------------------------------------------------
    # Task handling
    # ---------------------------------------------------
    def reset_task(self, task):
        """
        Reset environment task parameters and generate new route file
        task: array of [traffic_intensity, route_variation, ...]
        """
        if len(task) >= 1:
            self._traffic_intensity = float(task[0])
        if len(task) >= 2:
            self._route_variation = float(task[1])

        # Generate route file with new traffic intensity and route variation
        if self._original_route_file and os.path.exists(self._original_route_file):
            self._generate_route_file()
            self._route_file_modified = True
            print(
                f"[INFO] Generated route file: traffic_intensity={self._traffic_intensity:.3f}, "
                f"route_variation={self._route_variation:.3f}"
            )
        else:
            if self._original_route_file:
                print(f"[WARNING] Original route file not found: {self._original_route_file}")
            self._route_file_modified = False

    def sample_task(self, num_tasks=1):
        """Sample random task parameters"""
        tasks = np.random.uniform(0.5, 2.0, size=(num_tasks, 1))
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
            routes = root.findall("route")
            route_ids = [r.get("id") for r in routes] if routes else []

            # Scale all flows according to traffic_intensity and route_variation
            for flow in root.findall("flow"):
                # Scale vehsPerHour by traffic_intensity
                if "vehsPerHour" in flow.attrib:
                    old_rate = float(flow.get("vehsPerHour"))
                    new_rate = old_rate * self._traffic_intensity
                    flow.set("vehsPerHour", str(new_rate))

                # Scale probability by traffic_intensity (if exists)
                if "probability" in flow.attrib:
                    old_prob = float(flow.get("probability"))
                    new_prob = old_prob * self._traffic_intensity
                    new_prob = min(1.0, max(0.0, new_prob))
                    flow.set("probability", str(new_prob))

                # Apply route variation
                if (
                    self._route_variation > 0
                    and "route" in flow.attrib
                    and len(route_ids) > 0
                ):
                    if np.random.random() < self._route_variation:
                        new_route = np.random.choice(route_ids)
                        flow.set("route", new_route)

            # Cleanup old temp route
            if self._temp_route_file and os.path.exists(self._temp_route_file):
                try:
                    os.remove(self._temp_route_file)
                except Exception:
                    pass

            config_dir = os.path.dirname(self._original_route_file)
            task_hash = hash((self._traffic_intensity, self._route_variation)) % (10**8)
            temp_route_name = f"route_temp_{os.getpid()}_{id(self)}_{task_hash}.rou.xml"
            self._temp_route_file = os.path.join(config_dir, temp_route_name)

            tree.write(self._temp_route_file, encoding="UTF-8", xml_declaration=True)

            # Create temp config
            self._create_temp_config_file()

        except Exception as e:
            print(f"[ERROR] Could not generate route file: {e}")
            import traceback

            traceback.print_exc()
            self._temp_route_file = None
            self.sumo_config_path = self._base_config_path
            self._route_file_modified = False

    def _create_temp_config_file(self):
        """Create temporary SUMO config file with modified route file"""
        try:
            tree = ET.parse(self._base_config_path)
            root = tree.getroot()

            route_files_elem = root.find(".//route-files")
            if route_files_elem is not None:
                route_file_name = os.path.basename(self._temp_route_file)
                route_files_elem.set("value", route_file_name)
            else:
                print("[WARNING] No route-files element found in config")

            if self._temp_config_file and os.path.exists(self._temp_config_file):
                try:
                    os.remove(self._temp_config_file)
                except Exception:
                    pass

            config_dir = os.path.dirname(self._base_config_path)
            temp_config_name = f"config_temp_{os.getpid()}_{id(self)}.sumocfg"
            self._temp_config_file = os.path.join(config_dir, temp_config_name)

            tree.write(self._temp_config_file, encoding="UTF-8", xml_declaration=True)

            self.sumo_config_path = self._temp_config_file
        except Exception as e:
            print(f"[ERROR] Could not create temp config: {e}")
            self.sumo_config_path = self._base_config_path

    # ---------------------------------------------------
    # SUMO start / observation / reward
    # ---------------------------------------------------
    def _start_sumo(self):
        """Start SUMO simulation (có hỗ trợ GUI)"""
        if self.sumo_running:
            try:
                traci.close()
            except Exception:
                pass

        if self.use_gui:
            try:
                sumo_binary = sumolib.checkBinary("sumo-gui")
            except Exception:
                print("[WARNING] sumo-gui not found, falling back to sumo")
                sumo_binary = sumolib.checkBinary("sumo")
        else:
            try:
                sumo_binary = sumolib.checkBinary("sumo")
            except Exception:
                sumo_binary = sumolib.checkBinary("sumo-gui")

        sumo_cmd = [
            sumo_binary,
            "-c",
            self.sumo_config_path,
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
        ]
        sumo_cmd.append("--quit-on-end")
        sumo_cmd.append("true")

        traci.start(sumo_cmd)
        self.sumo_running = True

        tl_ids = traci.trafficlight.getIDList()
        if len(tl_ids) > 0:
            self.tl_id = tl_ids[0]
        else:
            raise ValueError("No traffic light found in SUMO network")

    def _get_observation(self):
        """Extract observation from SUMO + normalization kiểu LLIRL"""
        obs = []

        if self.tl_id is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            controlled_lanes = [l for l in controlled_lanes if not l.startswith(':')]
            controlled_lanes = list(dict.fromkeys(controlled_lanes))
            controlled_lanes = controlled_lanes[:self.num_lanes]
        except Exception as e:
            print(f"[WARNING] Error getting controlled lanes: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        for lane_id in controlled_lanes:
            try:
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                waiting_time = traci.lane.getWaitingTime(lane_id)
                vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                obs.extend([queue_length, waiting_time, vehicle_count])
            except Exception:
                obs.extend([0.0, 0.0, 0.0])

        # Pad nếu thiếu
        while len(obs) < self.observation_space.shape[0]:
            obs.extend([0.0, 0.0, 0.0])

        obs = np.array(obs[: self.observation_space.shape[0]], dtype=np.float32)

        # normalize theo từng feature
        obs_normalized = np.zeros_like(obs, dtype=np.float32)

        for i in range(0, len(obs_normalized), 3):
            if i + 2 < len(obs):
                # queue_length: 0–50
                obs_normalized[i] = np.clip(obs[i] / (50.0 + 1e-8), 0.0, 1.0)
                # waiting_time: 0–4090
                obs_normalized[i + 1] = np.clip(
                    obs[i + 1] / (4090.0 + 1e-8), 0.0, 1.0
                )
                # vehicle_count: 0–50
                obs_normalized[i + 2] = np.clip(obs[i + 2] / (50.0 + 1e-8), 0.0, 1.0)

        return obs_normalized

    def _get_reward(self):
        """
        Reward shaping:
        - Phạt waiting time & queue
        - Thưởng throughput & speed
        - Thưởng giảm queue, tăng speed
        """
        if self.tl_id is None:
            return 0.0

        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            controlled_lanes = [l for l in controlled_lanes if not l.startswith(':')]
            controlled_lanes = list(dict.fromkeys(controlled_lanes))
            controlled_lanes = controlled_lanes[:self.num_lanes]
        except Exception:
            return 0.0

        reward = 0.0

        total_waiting = 0.0
        total_queue = 0.0
        total_vehicles = 0.0
        total_speed = 0.0
        lane_count = 0

        for lane_id in controlled_lanes:
            try:
                waiting_time = traci.lane.getWaitingTime(lane_id)
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)

                total_waiting += waiting_time
                total_queue += queue_length
                total_vehicles += vehicle_count
                if mean_speed > 0:
                    total_speed += mean_speed
                lane_count += 1
            except Exception:
                continue

        if lane_count > 0:
            avg_waiting = total_waiting / lane_count
            avg_queue = total_queue / lane_count
            avg_vehicles = total_vehicles / lane_count
            avg_speed = total_speed / lane_count if total_speed > 0 else 0.0

            # 1. Penalty waiting time
            reward -= avg_waiting * 0.1

            # 2. Penalty queue + exponential penalty
            reward -= avg_queue * 0.8
            if avg_queue > 5.0:
                reward -= (avg_queue - 5.0) ** 1.5 * 2.0

            # 3. Reward throughput
            reward += avg_vehicles * 0.2

            # 4. Reward speed
            if avg_speed > 0:
                speed_reward = (avg_speed / 20.0) * 0.5
                reward += speed_reward

            # 5. Bonus low congestion
            if avg_queue < 2.0 and avg_speed > 5.0:
                reward += 5.0

            # 6. Bonus giảm queue
            queue_reduction = self._prev_queue - avg_queue
            if queue_reduction > 0:
                reward += queue_reduction * 2.0
            self._prev_queue = avg_queue

            # 7. Bonus tăng speed
            speed_increase = avg_speed - self._prev_speed
            if speed_increase > 0:
                reward += speed_increase * 0.5
            self._prev_speed = avg_speed

        reward = np.clip(float(reward), -300.0, 300.0)
        return reward

    # ---------------------------------------------------
    # Gym API: reset / step / close
    # ---------------------------------------------------
    def reset(self, env=True):
        """Reset environment"""
        # Regenerate route file if needed
        self.episode_total_reward = 0.0
        if self._route_file_modified and self._original_route_file:
            try:
                self._generate_route_file()
            except Exception as e:
                print(f"[WARNING] Could not regenerate route file on reset: {e}")

        if not self.sumo_running:
            self._start_sumo()
        else:
            try:
                traci.load(["-c", self.sumo_config_path])
            except Exception as e:
                print(f"[WARNING] Could not reload SUMO config: {e}")
                try:
                    traci.close()
                except Exception:
                    pass
                self.sumo_running = False
                self._start_sumo()

        self.step_count = 0
        self.current_phase_idx = 0
        self.current_phase = 0

        if self.tl_id:
            try:
                traci.trafficlight.setPhase(self.tl_id, self.current_phase_idx)
            except Exception:
                pass

        # Warm-up vài step
        try:
            for _ in range(5):
                traci.simulationStep()
        except Exception as e:
            print(f"[WARNING] SUMO connection lost during reset: {e}")
            try:
                traci.close()
            except Exception:
                pass
            self.sumo_running = False
            self._start_sumo()
            for _ in range(5):
                traci.simulationStep()

        # Reset tracking reward
        self._prev_queue = 0.0
        self._prev_speed = 0.0

        observation = self._get_observation()
        return observation

    def step(self, action):
        """
        DDQN chỉ chọn duration cho GREEN_PHASES (0 và 2)
        Môi trường tự động chèn YELLOW (1 và 3) với duration cố định.
        In reward từng time-step.
        """

        GREEN_PHASES = [0, 2]
        YELLOW_MAP = {0: 1, 2: 3}

        if not self.sumo_running:
            raise RuntimeError("SUMO not running. Call reset() first.")

        # Kiểm tra connection
        try:
            traci.simulation.getTime()
        except:
            self._start_sumo()

        # Phase hiện tại
        current_phase = traci.trafficlight.getPhase(self.tl_id)

        #============================
        # 1️⃣ Nếu là GREEN → agent chọn duration
        #============================
        if current_phase in GREEN_PHASES:

            # decode action → duration
            action = int(action) % self.num_durations
            green_duration = int(self.durations[action])
            
            #check phase
            # print(f"\n===== GREEN PHASE {current_phase} | duration = {green_duration} =====")

            # giữ GREEN trong green_duration step
            for _ in range(green_duration):
                traci.simulationStep()
                self.step_count += 1

                # Reward mỗi time-step
                step_reward = self._get_reward()
                #check phase
                # print(f"[STEP {self.step_count}] reward = {step_reward:.3f}")

                if self.step_count >= self.max_steps:
                    obs = self._get_observation()
                    total_reward = self._get_reward()
                    return obs, total_reward, True, False, {}

            #============================
            # 2️⃣ Chuyển sang YELLOW phase
            #============================
            yellow_phase = YELLOW_MAP[current_phase]
            traci.trafficlight.setPhase(self.tl_id, yellow_phase)
            #check phase
            # print(f"----- YELLOW PHASE {yellow_phase} | duration = {self.yellow_time} -----")

            for _ in range(self.yellow_time):
                traci.simulationStep()
                self.step_count += 1

                step_reward = self._get_reward()
                #check phase
                # print(f"[STEP {self.step_count}] (YELLOW) reward = {step_reward:.3f}")

                if self.step_count >= self.max_steps:
                    obs = self._get_observation()
                    total_reward = self._get_reward()
                    return obs, total_reward, True, False, {}

            #============================
            # 3️⃣ Next GREEN (0 → 2 → 0)
            #============================
            next_green = (current_phase + 2) % 4
            traci.trafficlight.setPhase(self.tl_id, next_green)
            #check phase
            # print(f"===== NEXT GREEN → phase {next_green} =====")

        #============================
        # 4️⃣ Trường hợp đang ở YELLOW (không mong muốn)
        #============================
        else:
            print(f"[WARNING] Unexpected YELLOW phase {current_phase}, correcting...")

            next_green = (current_phase + 1) % 4
            traci.trafficlight.setPhase(self.tl_id, next_green)
            traci.simulationStep()
            self.step_count += 1

            step_reward = self._get_reward()
            print(f"[STEP {self.step_count}] (AUTO-CORRECT) reward = {step_reward:.3f}")

        #============================
        # 5️⃣ Output cuối
        #============================
        obs = self._get_observation()
        reward = self._get_reward()
        self.episode_total_reward += reward
        done = self.step_count >= self.max_steps
        if done:
            print(f"\n===== EPISODE FINISHED | Total Reward = {self.episode_total_reward:.3f} =====\n")
        info = {
            "phase": current_phase,
            "duration": green_duration if current_phase in GREEN_PHASES else None,
            "step_count": self.step_count,
        }

        return obs, reward, done, False, info


    def close(self):
        """Close SUMO connection and cleanup temporary files"""
        if self.sumo_running:
            try:
                traci.close()
            except Exception:
                pass
            self.sumo_running = False

        if self._temp_route_file and os.path.exists(self._temp_route_file):
            try:
                os.remove(self._temp_route_file)
            except Exception:
                pass

        if self._temp_config_file and os.path.exists(self._temp_config_file):
            try:
                os.remove(self._temp_config_file)
            except Exception:
                pass
