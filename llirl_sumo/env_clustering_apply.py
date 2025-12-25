"""
ENV_CLUSTERING_RUNNER.PY
=========================
Chức năng:
- Nếu tồn tại env_models → load và tiếp tục clustering cho task tiếp theo
- Nếu chưa có → khởi tạo mới và chạy clustering từ đầu
- Giữ nguyên cơ chế đánh giá môi trường (likelihood + CRP + EM-step)
- Lưu lại tất cả kết quả: env_models, CRP, task history
"""

import os
import json
import pickle
import numpy as np
import torch
import random
import gym

from myrllib.samplers.sampler import BatchSampler
from myrllib.policies import UniformPolicy
from myrllib.mixture.env_model import EnvModel, construct_env_io
from myrllib.mixture.env_train import env_update, env_nominal_train
from myrllib.mixture.inference import CRP, compute_likelihood


# ============================================================
#                 UTILITY: LOAD CHECKPOINT
# ============================================================

def load_existing_clustering(model_path, device):
    """Load env_models, CRP and task info if exists."""
    env_models_path = os.path.join(model_path, "env_models.pth")
    crp_path = os.path.join(model_path, "crp_state.pkl")
    init_path = os.path.join(model_path, "env_model_init.pth")
    task_info_path = os.path.join(model_path, "tasks_info.json")

    if not os.path.exists(env_models_path):
        return None

    print("[CLUSTER] Loading existing clustering state...")

    ckpt = torch.load(env_models_path, map_location=device)

    input_size = ckpt["input_size"]
    output_size = ckpt["output_size"]
    hidden_sizes = ckpt["hidden_sizes"]
    sd_list = ckpt["env_models"]

    env_models = []
    for sd in sd_list:
        m = EnvModel(input_size, output_size, hidden_sizes).to(device)
        m.load_state_dict(sd)
        env_models.append(m)

    # Load universal init model
    env_model_init = EnvModel(input_size, output_size, hidden_sizes).to(device)
    env_model_init.load_state_dict(torch.load(init_path, map_location=device))

    # Load CRP
    with open(crp_path, "rb") as f:
        crp_state = pickle.load(f)
    crp = CRP(zeta=crp_state["zeta"])
    crp._L = crp_state["L"]
    crp._t = crp_state.get("t", crp._t)
    crp._prior = np.array(crp_state["prior"], dtype=float)

    # Load tasks history
    with open(task_info_path, "r") as f:
        task_info = json.load(f)

    return {
        "env_models": env_models,
        "env_model_init": env_model_init,
        "crp": crp,
        "tasks": np.array(task_info["tasks"]),
        "task_ids": np.array(task_info["task_ids"]),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_sizes": hidden_sizes
    }


# ============================================================
#                 SAVE CHECKPOINT
# ============================================================

def save_clustering(model_path, env_models, env_model_init, crp, tasks, task_ids,
                    input_size, output_size, hidden_sizes, save_state=True):

    if not save_state:
        print("[CLUSTER] Skip saving state (save_state=False)")
        return

    os.makedirs(model_path, exist_ok=True)

    torch.save({
        "env_models": [m.state_dict() for m in env_models],
        "input_size": input_size,
        "output_size": output_size,
        "hidden_sizes": hidden_sizes
    }, os.path.join(model_path, "env_models.pth"))

    torch.save(env_model_init.state_dict(),
               os.path.join(model_path, "env_model_init.pth"))

    with open(os.path.join(model_path, "crp_state.pkl"), "wb") as f:
        pickle.dump({
            "zeta": crp._zeta,
            "L": crp._L,
            "t": crp._t,
            "prior": crp._prior.tolist()
        }, f)

    with open(os.path.join(model_path, "tasks_info.json"), "w") as f:
        json.dump({
            "tasks": tasks.tolist(),
            "task_ids": task_ids.tolist()
        }, f, indent=2)

    print("[CLUSTER] State saved to", model_path)


# ============================================================
#                  CORE CLUSTERING STEP
# ============================================================

def cluster_single_task(env_models, env_model_init, crp,
                        sampler, policy_uni, task, args, device):
    """Run clustering for ONE task and return updated state."""

    sampler.reset_task(task)
    # collect episodes
    episodes = []
    for _ in range(args.et_length):
        episodes.append(sampler.sample(policy_uni, device=device))

    inputs, outputs = construct_env_io(episodes, env_type="reward", H=args.H)

    L = crp._L
    prior = crp._prior

    # ---- new model candidate ----
    new_model = EnvModel(inputs.shape[1], outputs.shape[1],
                         (args.env_hidden_size,) * args.env_num_layers).to(device)
    new_model.load_state_dict(env_model_init.state_dict())

    # ---- compute likelihood ----
    llls = np.zeros(L + 1)
    for i in range(L):
        llls[i] = compute_likelihood(env_models[i], inputs, outputs, sigma=args.sigma)
    llls[-1] = compute_likelihood(new_model, inputs, outputs, sigma=args.sigma)

    log_prior = np.log(np.clip(prior[:L + 1], 1e-12, 1.0))
    log_post = llls + log_prior
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    post /= post.sum()

    cluster_id = np.argmax(post) + 1
    print(f"[CLUSTER] Selected cluster {cluster_id}")

    if cluster_id == L + 1:     # create new cluster
        env_models.append(new_model)

    # update CRP
    crp.update(cluster_id)

    # ---- EM refinement ----
    def Estep():
        logll = np.zeros(len(env_models))
        for i in range(len(env_models)):
            logll[i] = compute_likelihood(env_models[i], inputs, outputs, sigma=args.sigma)
        prior_e = crp._prior[:len(env_models)]
        log_post = logll + np.log(np.clip(prior_e, 1e-12, 1.0))
        log_post -= np.max(log_post)
        p = np.exp(log_post)
        p /= p.sum()
        return logll, p

    def Mstep(post_e):
        for i in range(len(env_models)):
            env_models[i], _ = env_update(env_models[i], inputs, outputs,
                                          posterior=post_e[i], device=device)

    for _ in range(args.em_steps):
        _, p_e = Estep()
        Mstep(p_e)

    return env_models, env_model_init, crp, cluster_id


# ============================================================
#                 MAIN RUNNER FUNCTION
# ============================================================

def run_env_clustering(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Try load existing models first
    state = load_existing_clustering(args.model_path, device)

    # ===========================================
    # IF NO MODEL → INITIALIZE NEW CLUSTERING
    # ===========================================
    if state is None:
        print("[CLUSTER] No existing model. Creating new clustering state.")

        # create sampler
        env_name = "SUMO-SingleIntersection-v1"
        num_workers = 0
        sampler = BatchSampler(env_name, args.batch_size, num_workers=num_workers,
                               seed=args.seed, sumo_config_path=args.sumo_config,
                               max_steps=args.max_steps)

        # init state dim and action dim
        env = gym.make(env_name, sumo_config_path=args.sumo_config, max_steps=args.max_steps)
        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))

        policy_uni = UniformPolicy(state_dim, action_dim,
                                   low=env.action_space.low,
                                   high=env.action_space.high)

        # ---- prepare first task ----
        task = np.array([np.random.uniform(0.5, 1.5), np.random.uniform(0.0, 0.4)])
        tasks = [task]
        task_ids = [1]

        # ---- collect episodes ----
        sampler.reset_task(task)
        episodes = [sampler.sample(policy_uni, device=device)]
        inputs, outputs = construct_env_io(episodes, "reward", args.H)

        # ---- train nominal model ----
        env_model = EnvModel(inputs.shape[1], outputs.shape[1],
                             (args.env_hidden_size,) * args.env_num_layers).to(device)
        env_model, _ = env_nominal_train(env_model, inputs, outputs, device=device)

        env_models = [env_model]

        # ---- train universal init →
        env_model_init = EnvModel(inputs.shape[1], outputs.shape[1],
                                  (args.env_hidden_size,) * args.env_num_layers).to(device)
        env_model_init.load_state_dict(env_model.state_dict())

        crp = CRP(zeta=args.zeta)

        save_clustering(args.model_path, env_models, env_model_init, crp,
                        np.array(tasks), np.array(task_ids),
                        inputs.shape[1], outputs.shape[1],
                        (args.env_hidden_size,) * args.env_num_layers,
                        save_state=(not args.no_save_state))

        print("[CLUSTER] Initial model created.")
        return

    # ============================================================
    #      EXISTING MODEL FOUND → CONTINUE CLUSTERING
    # ============================================================

    print("[CLUSTER] Existing clustering found, continue processing new task...")

    # unpack state
    env_models = state["env_models"]
    env_model_init = state["env_model_init"]
    crp = state["crp"]
    tasks = state["tasks"]
    task_ids = state["task_ids"]
    input_size = state["input_size"]
    output_size = state["output_size"]
    hidden_sizes = state["hidden_sizes"]

    # np.random.seed(5360)
    new_task = np.array([1.0, 0])
    # new_task = np.array([1.5,  0.1])   # FOR TESTING CONSISTENCY
    tasks = np.vstack([tasks, new_task])

    # create sampler and policy
    env_name = "SUMO-SingleIntersection-v1"
    sampler = BatchSampler(env_name, args.batch_size, num_workers=0,
                           seed=args.seed, sumo_config_path=args.sumo_config,
                           max_steps=args.max_steps)
    env = gym.make(env_name, sumo_config_path=args.sumo_config, max_steps=args.max_steps)

    policy_uni = UniformPolicy(int(np.prod(env.observation_space.shape)),
                               int(np.prod(env.action_space.shape)),
                               low=env.action_space.low, high=env.action_space.high)

    # ---- run clustering ----
    env_models, env_model_init, crp, cid = cluster_single_task(
        env_models, env_model_init, crp,
        sampler, policy_uni, new_task, args, device
    )

    task_ids = np.append(task_ids, cid)

    # ---- save state ----
    save_clustering(args.model_path, env_models, env_model_init, crp,
                    tasks, task_ids,
                    input_size, output_size, hidden_sizes,
                    save_state=(not args.no_save_state))

    print(f"[CLUSTER] Task added → cluster {cid}")
    print(f"[CLUSTER] Total clusters = {crp._L}")


# ============================================================
#                        MAIN ENTRY
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="results/ddqn_apply_models")
    parser.add_argument("--sumo_config", type=str,
                        default="../nets/single-intersection/run_morning_6to10.sumocfg")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=4090)
    parser.add_argument("--et_length", type=int, default=1)
    parser.add_argument("--env_num_layers", type=int, default=2)
    parser.add_argument("--env_hidden_size", type=int, default=200)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--zeta", type=float, default=0.5)
    parser.add_argument("--em_steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_save_state", action="store_true",
                    help="Do not overwrite clustering state in model_path")


    args = parser.parse_args()
    run_env_clustering(args)
