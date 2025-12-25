import os
import sys
import time
import json
import random
import argparse
import pickle
import gc

import numpy as np
import torch
from tqdm import tqdm

import sumolib
import traci

# Thêm thư mục hiện tại vào sys.path để import local module
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from myrllib.algorithms import DDQN
from myrllib.envs_ddqn import SUMOEnv


# ============================================================
#                    ARGUMENT PARSER
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument('--sumo_config', type=str,
                    default='../nets/60p2k/run_60p2k.sumocfg',
                    help='path to SUMO configuration file')

parser.add_argument('--model_path', type=str,
                    default='saves/sumo_llirl',
                    help='folder chứa task_info, crp_state, policies_final,...')

parser.add_argument('--output', type=str,
                    default='output/sumo_llirl_ddqn_apply',
                    help='folder lưu summary .npy và xml')

parser.add_argument('--num_episodes', type=int, default=30,
                    help='số episode online learning cho mỗi period apply')

parser.add_argument('--max_steps', type=int, default=4090,
                    help='max steps mỗi episode (nên trùng SUMOEnv)')

# Chọn 1 hoặc nhiều period để apply (1-based). -1 nghĩa là period cuối cùng
parser.add_argument('--apply_periods', type=int, nargs='+', default=None,
                    help='Danh sách period (1-based) cần apply. '
                         '-1 nghĩa là chỉ period cuối cùng.')

# Hyperparameters DDQN (giữ như logic train ban đầu)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--replay_buffer_size', type=int, default=10000)
parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[200, 200])
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=0)

# Epsilon & target update
parser.add_argument('--epsilon_start', type=float, default=1.0)
parser.add_argument('--epsilon_end', type=float, default=0.05)
parser.add_argument('--epsilon_decay', type=float, default=0.995)
parser.add_argument('--target_update_freq', type=int, default=100)

# Transfer learning
parser.add_argument('--ddqn_init_path', type=str, default=None,
                    help='path đến ddqn_model_final.pth dùng để init agent đầu tiên '
                         'nếu chưa có policies_final_ddqn_final.pth')

parser.add_argument('--use_general_policy', action='store_true', default=True,
                    help='cluster mới init từ agent cũ (CRP / uniform), không random')

parser.add_argument('--eval_use_gui', action='store_true', default=False,
                    help='dùng SUMO GUI khi chạy apply (xml)')

parser.add_argument("--gui_delay_ms", type=int, default=120,
                    help="SUMO --delay in milliseconds (slows down GUI for viewing)")
parser.add_argument("--no_save_models", action="store_true",
                    help="Do not overwrite policies/mapping json in model_path")

args = parser.parse_args()
print(args)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device} | CUDA available: {torch.cuda.is_available()}\n")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)


# ============================================================
#          LOAD TASKS & CLUSTER IDs FROM CLUSTERING
# ============================================================

model_path = os.path.abspath(args.model_path)
os.makedirs(model_path, exist_ok=True)
os.makedirs(args.output, exist_ok=True)

tasks = None
task_ids = None

tasks_info_json = os.path.join(model_path, "tasks_info.json")
task_info_npy = os.path.join(model_path, "task_info.npy")

if os.path.exists(tasks_info_json):
    with open(tasks_info_json, "r") as f:
        info = json.load(f)
    tasks = np.array(info["tasks"], dtype=np.float32)
    task_ids = np.array(info["task_ids"], dtype=np.int32)
    print(f"[OK] Loaded {len(tasks)} tasks from tasks_info.json")
elif os.path.exists(task_info_npy):
    arr = np.load(task_info_npy)
    tasks = arr[:, :-1].astype(np.float32)
    task_ids = arr[:, -1].astype(np.int32)
    print(f"[OK] Loaded {len(tasks)} tasks from task_info.npy")
else:
    raise FileNotFoundError(
        f"Không tìm thấy tasks_info.json hoặc task_info.npy trong {model_path}.\n"
        f"Hãy chạy env_clustering_runner.py / env_clustering_apply.py trước."
    )

num_total_periods = len(tasks)
print(f"[INFO] Tổng số period = {num_total_periods}")


# ============================================================
#      XÁC ĐỊNH DANH SÁCH PERIOD SẼ APPLY (0-based index)
# ============================================================

if args.apply_periods is None:
    # Mặc định: chỉ period cuối cùng
    apply_indices = [num_total_periods - 1]
else:
    apply_indices = []
    for p in args.apply_periods:
        if p == -1:
            idx = num_total_periods - 1
        else:
            if p < 1 or p > num_total_periods:
                raise ValueError(f"apply_period {p} nằm ngoài [1, {num_total_periods}]")
            idx = p - 1
        if idx not in apply_indices:
            apply_indices.append(idx)

apply_indices = sorted(apply_indices)
print(f"[INFO] Apply các period (0-based indices): {apply_indices}")
print("       (1-based):", [i + 1 for i in apply_indices])

num_apply_periods = len(apply_indices)


# ============================================================
#                 LOAD CRP PRIOR (NẾU CÓ)
# ============================================================

crp_prior = None
crp_path = os.path.join(model_path, "crp_state.pkl")
if os.path.exists(crp_path):
    try:
        with open(crp_path, "rb") as f:
            crp_data = pickle.load(f)
        crp_prior = np.array(crp_data.get("prior", []), dtype=np.float32)
        print(f"[OK] Loaded CRP prior, shape = {crp_prior.shape}")
    except Exception as e:
        print(f"[WARNING] Failed to load CRP state: {e}")
else:
    print(f"[INFO] No CRP state found at {crp_path}. Sẽ dùng uniform/random khi cần.")


# ============================================================
#                      CREATE ENVIRONMENT
# ============================================================

sumo_config_path = os.path.abspath(args.sumo_config)
env = SUMOEnv(sumo_config_path=sumo_config_path,
              max_steps=args.max_steps,
              use_gui=False)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"State dim: {state_dim}; Action dim: {action_dim}")


# ============================================================
#            LOAD EXISTING CLUSTER POLICIES (If ANY)
# ============================================================

def load_trained_ddqn_policies(model_path, state_dim, action_dim, args, device):
    """
    Load file policies_final_ddqn_final.pth để lấy cluster_agents.
    Trả về dict {cluster_id: DDQN instance}.
    """
    ckpt_path = os.path.join(model_path, "policies_final_ddqn_final.pth")
    if not os.path.exists(ckpt_path):
        print(f"[APPLY] Không tìm thấy {ckpt_path} → sẽ tạo agent khi cần.")
        return {}

    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    clusters = data.get("clusters", {})
    print(f"[APPLY] Loaded {len(clusters)} cluster policies from policies_final_ddqn_final.pth")

    cluster_agents = {}
    for cid_key, agent_data in clusters.items():
        try:
            cid = int(cid_key)
        except Exception:
            cid = int(cid_key) if not isinstance(cid_key, int) else cid_key

        agent = DDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            replay_buffer_size=args.replay_buffer_size,
            hidden_sizes=tuple(args.hidden_sizes),
            device=device,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            target_update_freq=args.target_update_freq
        )
        agent.q_network.load_state_dict(agent_data["q_network"])
        agent.target_network.load_state_dict(agent_data["target_network"])
        agent.epsilon = float(agent_data.get("epsilon", args.epsilon_end))
        cluster_agents[cid] = agent

    return cluster_agents


cluster_agents = load_trained_ddqn_policies(
    model_path, state_dim, action_dim, args, device
)

# Rewards chỉ cho các period apply (dimension: [num_apply_periods, num_episodes])
all_rewards = np.zeros((num_apply_periods, args.num_episodes), dtype=np.float32)

_completed_periods = 0  # đếm period apply đã xử lý

# ============================================================
#               PATCH SUMO START FUNCTION
# ============================================================

def patched_start_sumo(self, ep):
    sumo_binary = sumolib.checkBinary("sumo-gui" if args.eval_use_gui else "sumo")

    summary_xml = os.path.join(args.output, f"summary_ep_{ep}.xml")
    tripinfo_xml = os.path.join(args.output, f"tripinfo_ep_{ep}.xml")
    vehroute_xml = os.path.join(args.output, f"vehroute_ep_{ep}.xml")

    cmd = [
        sumo_binary, "-c", self.sumo_config_path,
        "--summary-output", summary_xml,
        "--tripinfo-output", tripinfo_xml,
        "--vehroute-output", vehroute_xml,
        "--no-step-log", "true",
        "--quit-on-end", "true"
    ]

    if args.eval_use_gui:
        cmd += ["--start", "--delay", str(args.gui_delay_ms)]  

    traci.start(cmd)
    self.sumo_running = True
    self.tl_id = traci.trafficlight.getIDList()[0]




# ============================================================
#           GENERAL POLICY CHOICE (GIỮ LOGIC CŨ)
# ============================================================

def get_general_policy_cluster_id(cluster_agents, crp_prior=None):
    """
    Chọn 1 cluster làm 'general policy' cho cluster mới.

    - Nếu có CRP prior: chọn cluster có prior lớn nhất (nhiều cluster = random trong số đó).
    - Nếu không: chọn random 1 cluster đang tồn tại.
    """
    if len(cluster_agents) == 0:
        return None

    existing_clusters = list(cluster_agents.keys())

    if crp_prior is not None and len(crp_prior) >= max(existing_clusters):
        priors = np.array([crp_prior[cid - 1] for cid in existing_clusters], dtype=float)
        max_prior = priors.max()
        candidate_clusters = [cid for cid, p in zip(existing_clusters, priors) if p == max_prior]
        chosen = random.choice(candidate_clusters)
        print(f"[GENERAL-POLICY] PRIOR-based, priors={priors}, max_prior={max_prior}, "
              f"candidates={candidate_clusters}, chosen={chosen}")
        return chosen

    chosen = random.choice(existing_clusters)
    print(f"[GENERAL-POLICY] No CRP prior → random cluster {chosen} as general policy.")
    return chosen



# ============================================================
#               TRAINING FUNCTION PER PERIOD
def collect_step_metrics(env, info, action):
    """Thu các metrics giống apply_ddqn.py."""
    tl = env.tl_id
    lanes = traci.trafficlight.getControlledLanes(tl)

    q = np.mean([traci.lane.getLastStepHaltingNumber(l) for l in lanes])   # queue
    w = np.mean([traci.lane.getWaitingTime(l) for l in lanes])             # waiting time
    v = np.mean([traci.lane.getLastStepVehicleNumber(l) for l in lanes])   # vehicle count
    s = np.mean([traci.lane.getLastStepMeanSpeed(l) for l in lanes])       # mean speed
    phase = traci.trafficlight.getPhase(tl)
    phase_duration = info.get("phase_duration", info.get("duration", None))

    return [q, w, v, s, action, phase, phase_duration]


def train_one_period_episode_summaries(agent: DDQN,
                                      global_period_idx: int,
                                      task_vec: np.ndarray):

    print(f"\n[APPLY] Online learning for period {global_period_idx+1}")

    env.reset_task(task_vec)


    for ep in range(args.num_episodes):
        print(f"[APPLY] Episode {ep+1}/{args.num_episodes}")

            # Force SUMO restart
        try:
            traci.close(False)
        except:
            pass
        env.sumo_running = False

        current_ep = ep + 1
        env._start_sumo = lambda self=env, ep=current_ep: patched_start_sumo(self, ep)

        state = env.reset()

       
        done = False
        ep_reward = 0.0
        step_records = []

        step_idx = 0
        while not done and step_idx < args.max_steps:

            action = agent.select_action(state, training=True)
            out = env.step(action)

            if len(out) == 5:
                next_state, reward, terminated, truncated, info = out
                done = terminated or truncated
            else:
                next_state, reward, done, info = out

            agent.store_transition(state, action, reward, next_state, done)
            if len(agent.replay_buffer) >= args.batch_size:
                agent.train_step()

            # ======= SUMMARY FORMAT GIỐNG apply_ddqn.py =======
            tl = env.tl_id
            lanes = traci.trafficlight.getControlledLanes(tl)

            q = np.mean([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
            w = np.mean([traci.lane.getWaitingTime(l) for l in lanes])
            v = np.mean([traci.lane.getLastStepVehicleNumber(l) for l in lanes])
            s = np.mean([traci.lane.getLastStepMeanSpeed(l) for l in lanes])
            phase = traci.trafficlight.getPhase(tl)
            phase_duration = info.get("phase_duration", info.get("duration", None))

            step_records.append([
                reward, q, w, v, s,
                action, phase, phase_duration
            ])
            # ====================================================

            ep_reward += reward
            state = next_state
            step_idx += 1

        agent.update_epsilon()
        print(f"[EP {ep+1}] reward={ep_reward:.2f}, eps={agent.epsilon:.4f}")

        # ===== SAVE THIS EPISODE =====
        summary_arr = np.array(step_records, dtype=np.float32)


        npy_path = os.path.join(args.output, f"summary_ep_{ep+1}.npy")
        np.save(npy_path, summary_arr)
        print(f"[SAVE] Episode {ep+1} saved → {npy_path}")




    all_records = []
    for ep in range(args.num_episodes):
        arr = np.load(os.path.join(args.output, f"summary_ep_{ep+1}.npy"))
        all_records.append(arr)

    max_len = max(len(r) for r in all_records)
    padded = [np.vstack([r, np.zeros((max_len - len(r), r.shape[1]))]) for r in all_records]

    avg = np.mean(np.stack(padded, axis=0), axis=0)
    np.save(os.path.join(args.output, "summary_ddqn_avg.npy"), avg)

# ============================================================
#               SAVE POLICIES & PERIOD MAPPING
# ============================================================

def save_policies_and_mapping(apply_indices, tag="final"):
    """
    Lưu:
      - policies_final_ddqn_{tag}.pth
      - period_cluster_mapping_{tag}.json (merge với file cũ nếu có)
    KHÔNG lưu:
      - policy_selection_history_{tag}.json
      - training_metrics_ddqn_{tag}.json
    """
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    if getattr(args, "no_save_models", False):
        print("[SKIP] no_save_models=True → skip saving policies AND mapping")
        return
    # 1) Save policies_final
    policies_final = {
        "clusters": {
            int(cid): {
                "q_network": agent.q_network.state_dict(),
                "target_network": agent.target_network.state_dict(),
                "epsilon": agent.epsilon,
            }
            for cid, agent in cluster_agents.items()
        }
    }
    
    pol_path = os.path.join(model_path, f"policies_final_ddqn_{tag}.pth")
    torch.save(policies_final, pol_path)
    print(f"[OK] Saved policies to {pol_path}")


    # 2) period_cluster_mapping (merge nếu đã tồn tại)
    mapping_path = os.path.join(model_path, f"period_cluster_mapping_{tag}.json")
    mapping_dict = {}

    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r") as f:
                old_list = json.load(f)
            for item in old_list:
                mapping_dict[int(item["period"])] = item
        except Exception as e:
            print(f"[WARNING] Failed to load old mapping: {e}")

    # Cập nhật / thêm mapping cho các period apply
    for idx in apply_indices:
        period_1b = idx + 1
        cid = int(task_ids[idx])
        mapping_dict[period_1b] = {
            "period": period_1b,
            "cluster_id": cid,
            "task": tasks[idx].tolist(),
            "assigned_policy": cid
        }

    # Ghi lại list sắp xếp theo period
    new_list = [mapping_dict[k] for k in sorted(mapping_dict.keys())]
    with open(mapping_path, "w") as f:
        json.dump(new_list, f, indent=2)

    print(f"[OK] Saved period-cluster mapping to {mapping_path}")


# ============================================================
#                       MAIN LOOP
# ============================================================

print("\n========== DDQN + LLIRL APPLY (ONLINE LEARNING) ==========\n")
start_time = time.time()

try:
    for apply_idx, period in enumerate(apply_indices):
        task = tasks[period]
        cid = int(task_ids[period])

        print(f"\n----------- APPLY Period (global) {period} (1-based={period+1}) -----------")
        print(f"Task params: {task}")
        print(f"Cluster ID from env_clustering: {cid}")

        if cid < 1:
            raise ValueError(f"Invalid cluster id {cid} at period {period}")

        num_clusters_now = len(cluster_agents)

        # ----------- Lấy / tạo agent cho cluster này -----------
        if cid in cluster_agents:
            print(f"[CLUSTER-OLD] Using existing agent for cluster {cid}")
            agent = cluster_agents[cid]
        else:
            print(f"[CLUSTER-NEW] Creating new agent for cluster {cid}")
            agent = DDQN(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=args.lr,
                gamma=args.gamma,
                batch_size=args.batch_size,
                replay_buffer_size=args.replay_buffer_size,
                hidden_sizes=tuple(args.hidden_sizes),
                device=device,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay=args.epsilon_decay,
                target_update_freq=args.target_update_freq
            )

            # Nếu chưa có cluster nào → init từ ddqn_init_path (nếu có)
            if num_clusters_now == 0:
                init_ckpt = args.ddqn_init_path
                if init_ckpt is None:
                    init_ckpt = os.path.join(args.model_path, "ddqn_model_final.pth")

                if os.path.exists(init_ckpt):
                    try:
                        agent.load(init_ckpt)
                        print(f"[INIT] Loaded initial DDQN weights from {init_ckpt}")
                    except Exception as e:
                        print(f"[WARNING] Failed to load {init_ckpt}: {e}")
                        print("         Using random initialization for first agent.")
                else:
                    print(f"[WARNING] ddqn_init_path {init_ckpt} not found. Using random init.")
            # Nếu đã có cluster trước → dùng general policy
            elif args.use_general_policy and num_clusters_now > 0:
                base_cid = get_general_policy_cluster_id(cluster_agents, crp_prior)
                if base_cid is None or base_cid not in cluster_agents:
                    base_cid = random.choice(list(cluster_agents.keys()))
                    print(f"[INIT-GENERAL] No explicit general policy, "
                          f"using random cluster {base_cid} as base.")

                base_agent = cluster_agents[base_cid]
                agent.q_network.load_state_dict(base_agent.q_network.state_dict())
                agent.target_network.load_state_dict(base_agent.target_network.state_dict())
                agent.epsilon = base_agent.epsilon
                print(f"[INIT-TRANSFER] Init new cluster {cid} from cluster {base_cid}")
            else:
                print("[INIT] New cluster with random initialization.")

            cluster_agents[cid] = agent
        
        
        # ----------- Train online + ghi summary -----------
        train_one_period_episode_summaries(agent, period, task)



        _completed_periods = apply_idx + 1

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

except Exception as e:
    print(f"[ERROR] Exception during apply/online training: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Lưu policies + mapping (merge với cũ nếu có)
    save_policies_and_mapping(apply_indices, tag="final")

    env.close()
    try:
        traci.close(False)
    except:
        pass

    total_min = (time.time() - start_time) / 60.0
    print(f"\n[FINISHED] Total apply-online time: {total_min:.2f} minutes")
