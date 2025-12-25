"""
DDQN + LLIRL-style Training for SUMO (cluster-based transfer)

Logic LLIRL:
- Xác định xem task thuộc cluster nào (task_ids từ env_clustering).
- Nếu cluster cũ  -> lấy agent DDQN cũ ra và train tiếp.
- Nếu cluster mới -> tạo agent mới, init từ:
- Transferlearning từ ddqn_model_final.pth (nếu có)

Lưu các thông tin:
- policies_final_ddqn_{tag}.pth
- period_cluster_mapping_{tag}.json
- policy_selection_history_{tag}.json
- training_metrics_ddqn_{tag}.json   <-- quan trọng
"""

import os
import sys
import time
import json
import random
import argparse
import pickle
import signal
import atexit
import gc

import numpy as np
import torch
from tqdm import tqdm

# Thêm thư mục hiện tại vào sys.path để import local module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from myrllib.algorithms import DDQN                  
from myrllib.envs_ddqn import SUMOEnv      

# ======================= Args =======================

parser = argparse.ArgumentParser()

parser.add_argument('--sumo_config', type=str,
                    default='../nets/60p2k/run_60p2k.sumocfg',
                    help='path to SUMO configuration file')

parser.add_argument('--model_path', type=str,
                    default='saves/sumo_llirl',
                    help='folder chứa task_info, crp_state, ...')

parser.add_argument('--output', type=str,
                    default='output/sumo_llirl_ddqn',
                    help='folder lưu reward & log')

parser.add_argument('--num_periods', type=int, default=10,
                    help='số period muốn train (<= số task thực tế)')

parser.add_argument('--num_episodes', type=int, default=30,
                    help='số episode train mỗi period')

parser.add_argument('--max_steps', type=int, default=4090,
                    help='max steps mỗi episode (nên trùng SUMOEnv)')

# Hyperparameters DDQN (giống ddqn_training.py)
parser.add_argument('--lr', type=float, default=1e-3)
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
                    help='path đến ddqn_model_final.pth dùng để init agent đầu tiên')


args = parser.parse_args()
print(args)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device} | CUDA available: {torch.cuda.is_available()}\n")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)


# ======================= Load tasks & cluster IDs =======================

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
        f"Hãy chạy env_clustering.py trước."
    )

if len(tasks) < args.num_periods:
    print(f"[WARNING] num_periods={args.num_periods} > số task={len(tasks)} → giảm num_periods.")
    args.num_periods = len(tasks)

print("Tasks shape:", tasks.shape)
print("Task IDs (sample):", task_ids[:args.num_periods])


# ======================= Load CRP priors (nếu có) =======================

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
    print(f"[INFO] No CRP state found at {crp_path}. Sẽ dùng uniform khi cần.")


# ======================= Create Environment =======================

sumo_config_path = os.path.abspath(args.sumo_config)
env = SUMOEnv(sumo_config_path=sumo_config_path,
              max_steps=args.max_steps,
              use_gui=False)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"State dim: {state_dim}; Action dim: {action_dim}")


# ======================= Containers & Metrics =======================

# all_rewards[period, episode]
all_rewards = np.zeros((args.num_periods, args.num_episodes), dtype=np.float32)

# Một agent DDQN cho mỗi cluster (cluster_id → DDQN instance)
cluster_agents = {}
cluster_mean_rewards = {}    

# training_metrics: theo period
training_metrics = {
    "periods": [],
    "cluster_ids": [],
    "mean_rewards": [],
    "std_rewards": [],
    "best_rewards": [],
    "best_episodes": [],
    "epsilon_end": [],
    "episodes_trained": [],
}

# policy_selection_history: giống LLIRL, mỗi period chọn policy nào
policy_selection_history = {
    "period": [],
    "cluster_id": [],
    "selected_policy": [],
    "reward_mean": [],
    "reward_best": [],
}

_save_on_exit = True
_completed_periods = 0


# ======================= SAVE RESULTS =======================

def save_results(final=False):
    """Lưu reward cơ bản + một số file tóm tắt."""
    global _completed_periods
    tag = "final" if final else "intermediate"

    try:
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        # 1) Rewards
        rew_path = os.path.join(args.output, f"rews_ddqn_llirl_{tag}.npy")
        np.save(rew_path, all_rewards)
        print(f"[OK] Saved rewards to {rew_path}")

        # 2) policies_final (cluster → q_network + target_network + epsilon)
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
        torch.save(
            policies_final,
            os.path.join(model_path, f"policies_final_ddqn_{tag}.pth")
        )

        # 3) period_cluster_mapping
        period_cluster_mapping = [
            {
                "period": i + 1,
                "cluster_id": int(task_ids[i]),
                "task": tasks[i].tolist(),
                "assigned_policy": int(task_ids[i]),  # 1 cluster = 1 policy
            }
            for i in range(_completed_periods)
        ]
        with open(os.path.join(model_path, f"period_cluster_mapping_{tag}.json"), "w") as f:
            json.dump(period_cluster_mapping, f, indent=2)

        # 4) policy_selection_history
        with open(os.path.join(model_path, f"policy_selection_history_{tag}.json"), "w") as f:
            json.dump(policy_selection_history, f, indent=2)
        
        #5 ) training_metrics
        with open(os.path.join(model_path, f"training_metrics_ddqn_{tag}.json"), "w") as f:
            json.dump(training_metrics, f, indent=2)
        print(f"[OK] Saved full LLIRL-DDQN logs ({tag}).")
    except Exception as e:
        print(f"[WARNING] Error while saving ({tag}): {e}")


def save_on_exit():
    global _save_on_exit
    if not _save_on_exit:
        return
    print("\n[EXIT] Saving intermediate DDQN+LLIRL results...")
    save_results(final=False)


def signal_handler(signum, frame):
    print("\n[INTERRUPT] Training interrupted.")
    save_on_exit()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(save_on_exit)
#  ======================= Helper: init agent =======================

def init_agent_for_new_cluster(agent: DDQN):
    """
    Init mỗi cluster mới từ baseline ddqn_model_final.pth (transfer learning).
    Không dùng general_policy và không random, trừ khi file mất.
    """
    # 1) Lấy đường dẫn baseline
    init_ckpt = args.ddqn_init_path
    if init_ckpt is None:
        init_ckpt = os.path.join(args.model_path, "ddqn_model_final.pth")

    # 2) Load baseline
    if os.path.exists(init_ckpt):
        try:
            agent.load(init_ckpt)
            # reset exploration cho transfer (cluster mới)
            agent.epsilon = args.epsilon_start
            agent.step_count = 0
            print(f"[INIT-TRANSFER] Loaded baseline DDQN weights from {init_ckpt}")
        except Exception as e:
            print(f"[WARNING] Failed to load baseline {init_ckpt}: {e}")
            print("         → Using random initialization for this cluster.")
    else:
        print(f"[WARNING] Baseline {init_ckpt} not found → Random initialization.")

    return agent




# ======================= Train one period =======================

def train_one_period(agent: DDQN, period_idx: int, task_vec: np.ndarray):
    """
    Train DDQN agent cho 1 period, style giống ddqn_training.py:
    select_action → env.step → store_transition → train_step → update_epsilon.
    """
    env.reset_task(task_vec)
    rewards = np.zeros(args.num_episodes, dtype=np.float32)

    for ep in tqdm(range(args.num_episodes), desc=f"Period {period_idx+1}"):
        state = env.reset()
        ep_reward = 0.0

        for step in range(args.max_steps):
            action = agent.select_action(state, training=True)
            out = env.step(action)

            # Env hiện đại: (obs, reward, terminated, truncated, info)
            if len(out) == 5:
                next_state, reward, terminated, truncated, info = out
                done = terminated or truncated
            else:
                next_state, reward, done, info = out
                truncated = False

            agent.store_transition(state, action, reward, next_state, done)

            if len(agent.replay_buffer) >= args.batch_size:
                agent.train_step()

           


            state = next_state
            ep_reward += reward

            if done:
                break

        agent.update_epsilon()
        rewards[ep] = ep_reward

        print(f"[TRAIN] Period {period_idx+1} | Episode {ep+1}/{args.num_episodes} "
              f"| Reward={ep_reward:.2f} | Epsilon={agent.epsilon:.4f}")

    return rewards


# ======================= Main LLIRL+DDQN Loop =======================

print("\n========== DDQN + LLIRL (cluster-based) ==========\n")
start_time = time.time()

try:
    for period in range(args.num_periods):
        # period_idx = 0-based dùng cho mảng all_rewards
        period_idx = period

        task = tasks[period]
        cid = int(task_ids[period])

        print(f"\n----------- Period {period}/{args.num_periods} -----------")
        print(f"Task params: {task}")
        print(f"Cluster ID from env_clustering: {cid}")

        if cid < 1:
            raise ValueError(f"Invalid cluster id {cid} at period {period}")

        num_clusters_now = len(cluster_agents)

        # --- Lấy / tạo agent cho cluster này ---
        if cid in cluster_agents:
            print(f"[CLUSTER-OLD] Using existing agent for cluster {cid}")
            agent = cluster_agents[cid]
        else :
            print(f"[CLUSTER-NEW] Creating new agent for new cluster {cid}")
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

            # INIT NEW CLUSTER FROM BASELINE 
            agent = init_agent_for_new_cluster(agent)
            cluster_agents[cid] = agent

        

        # --- Train agent cho period này ---
        period_rewards = train_one_period(agent, period_idx, task)
        all_rewards[period_idx] = period_rewards

        best_r = float(period_rewards.max())
        best_ep = int(period_rewards.argmax())
        mean_r = float(period_rewards.mean())
        std_r = float(period_rewards.std())

        print(f"[PERIOD {period}] mean_reward={mean_r:.2f}, "
              f"best_reward={best_r:.2f} at episode {best_ep+1}")


        # training_metrics (nhấn mạnh)
        training_metrics["periods"].append(period+1)
        training_metrics["cluster_ids"].append(cid)
        training_metrics["mean_rewards"].append(mean_r)
        training_metrics["std_rewards"].append(std_r)
        training_metrics["best_rewards"].append(best_r)
        training_metrics["best_episodes"].append(best_ep + 1)
        training_metrics["epsilon_end"].append(agent.epsilon)
        training_metrics["episodes_trained"].append(args.num_episodes)

        # policy_selection_history
        policy_selection_history["period"].append(period)
        policy_selection_history["cluster_id"].append(cid)
        policy_selection_history["selected_policy"].append(cid)  # 1 cluster = 1 policy
        policy_selection_history["reward_mean"].append(mean_r)
        policy_selection_history["reward_best"].append(best_r)

        _completed_periods = period

        # Lưu intermediate sau mỗi period (bao gồm cả policies)
        # save_results(final=False)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


except Exception as e:
    print(f"[ERROR] Exception during training: {e}")
    import traceback
    traceback.print_exc()

finally:
    _save_on_exit = False  # tránh atexit gọi thêm lần nữa
    save_results(final=True)
    env.close()
    total_min = (time.time() - start_time) / 60.0
    print(f"\n[FINISHED] Total training time: {total_min:.2f} minutes")
