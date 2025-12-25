import os
import sys
import numpy as np
import torch
import sumolib
import traci

# ==============================
#   ADD PATH TO MYRLLIB
# ==============================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "ddqn_sumo"))

from myrllib.envs.sumo_env import SUMOEnv
from myrllib.algorithms.ddqn import DDQN


# ========================================================================
#               APPLY DDQN — RUN MULTI EPISODE WITH XML EXPORT
# ========================================================================
def run_ddqn_multi_episode(
    sumo_config,
    model_path,
    output_folder="result_ddqn_30ep",
    max_steps=4090,
    num_episodes=30,
    use_gui=False
):

    os.makedirs(output_folder, exist_ok=True)

    print("\n==============================")
    print(" APPLY DDQN — EPISODES ")
    print("==============================\n")

    # Create base env (we override startup each EP)
    env = SUMOEnv(
        sumo_config_path=sumo_config,
        max_steps=max_steps,
        use_gui=use_gui
    )

    # =======================================================
    # 1) LOAD DDQN MODEL
    # =======================================================
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ddqn = DDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=(200, 200),
        device="cpu"
    )
    ddqn.load(model_path)
    ddqn.epsilon = 0.0  # Greedy inference

    print("Model loaded successfully.\n")

    all_episode_records = []

    # =======================================================
    # 2) LOOP OVER EPISODES
    # =======================================================
    np.random.seed(5360)
    new_task = np.array([1.0,0.0])
    for ep in range(1, num_episodes + 1):

        print(f"\n===== EPISODE {ep} / {num_episodes} =====")

        # --- FIXED TASK PARAMS ---

        env.reset_task(new_task)

        # --- OVERRIDE SUMO START FOR THIS EPISODE ---
        def patched_start_sumo(self, ep=ep):
            """Start SUMO fresh each episode + unique XML output files."""
            if self.sumo_running:
                try:
                    traci.close()
                except:
                    pass

            sumo_binary = sumolib.checkBinary("sumo-gui" if self.use_gui else "sumo")

            summary_xml = os.path.join(output_folder, f"summary_ep_{ep}.xml")
            tripinfo_xml = os.path.join(output_folder, f"tripinfo_ep_{ep}.xml")
            vehroute_xml = os.path.join(output_folder, f"vehroute_ep_{ep}.xml")

            sumo_cmd = [
                sumo_binary,
                "-c", self.sumo_config_path,
                "--no-step-log", "true",
                "--no-warnings", "true",
                "--summary-output", summary_xml,
                "--tripinfo-output", tripinfo_xml,
                "--vehroute-output", vehroute_xml,
                "--quit-on-end", "true"
            ]
            if self.use_gui:
                sumo_cmd += ["--start", "--delay", "150"]  # 150ms/step; tăng 300-600 nếu muốn chậm hơn

            print(f"[SUMO] Starting episode {ep} with XML logging...")
            traci.start(sumo_cmd)

            self.sumo_running = True

            tl_ids = traci.trafficlight.getIDList()
            if len(tl_ids) == 0:
                raise RuntimeError("No TL found in network!")
            self.tl_id = tl_ids[0]

        # Inject patch
        env._start_sumo = patched_start_sumo.__get__(env, SUMOEnv)

        # =======================================================
        # 3) RESET ENV (START SUMO PROCESS)
        # =======================================================
        state = env.reset()
        done = False
        step = 0

        ep_records = []

        # =======================================================
        # 4) RUN ONE EPISODE
        # =======================================================
        while not done and step < max_steps:

            action = ddqn.select_action(state, training=False)

            next_state, reward, terminated, truncated, info = env.step(action)

            lanes = traci.trafficlight.getControlledLanes(env.tl_id)
            lanes = list(dict.fromkeys(lanes))                 
            lanes = [l for l in lanes if not l.startswith(':')]  
            lanes = lanes[:env.num_lanes]

            q = np.mean([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
            w = np.mean([traci.lane.getWaitingTime(l) for l in lanes])
            v = np.mean([traci.lane.getLastStepVehicleNumber(l) for l in lanes])
            s = np.mean([traci.lane.getLastStepMeanSpeed(l) for l in lanes])

            current_phase = traci.trafficlight.getPhase(env.tl_id)
            duration = info.get("duration", None)

            ep_records.append([
                reward, q, w, v, s,
                action, current_phase, duration
            ])

            state = next_state
            step += 1
            done = terminated or truncated

        # Save this episode’s npy
        ep_records = np.array(ep_records, dtype=np.float32)
        np.save(os.path.join(output_folder, f"summary_ep_{ep}.npy"), ep_records)

        print(f"Saved summary_ep_{ep}.npy")
        all_episode_records.append(ep_records)

        # Close SUMO before next EP
        env.close()
        try:
            traci.close(False)
        except:
            pass

    # =======================================================
    # 5) Compute Per-Step Average Summary Across 30 Episodes
    # =======================================================
    print("\nComputing average summary across 30 episodes...")

    max_len = max(len(x) for x in all_episode_records)
    padded = []

    for rec in all_episode_records:
        pad = np.zeros((max_len - len(rec), rec.shape[1]), dtype=np.float32)
        padded.append(np.vstack([rec, pad]))

    padded = np.stack(padded, axis=0)
    avg_summary = np.mean(padded, axis=0)

    np.save(os.path.join(output_folder, "summary_ddqn_avg.npy"), avg_summary)
    print("Saved summary_ddqn_avg.npy")

    print("\n========== DONE EP DDQN EXPORT ==========\n")

    return avg_summary



# ========================================================================
#                               MAIN
# ========================================================================
if __name__ == "__main__":

    run_ddqn_multi_episode(
        sumo_config="nets/demo/run_60p2k.sumocfg",
        model_path="ddqn_sumo/saves/sumo_single_intersection/ddqn_model_final.pth",
        output_folder="result_test/m",
        max_steps=4090,
        num_episodes=1,
        use_gui=True
    )
