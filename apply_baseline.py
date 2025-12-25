# ===========================================
#     APPLY BASELINE — MULTI EPISODE
# ===========================================

import os
import traci
import datetime
import numpy as np
from baseline.lanes import LaneGroups
from baseline.env import TLSEnv
from baseline.strategies import FixedTime, PHASE_NS


def run_baseline_multi_episode(
    sumo_config,
    output_folder,
    max_steps=4090,
    num_episodes=1,
    g_ns=42,
    y_ns=3,
    g_ew=42,
    y_ew=3,
    use_gui=True
):

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] OUTPUT FOLDER = {output_folder}")

    lg = LaneGroups()

    np.random.seed(5360)
    new_task = np.array([1.0,0.0])

    for ep in range(1, num_episodes + 1):
        print(f"\n========== EPISODE {ep}/{num_episodes} ==========")

        # Output paths per episode
        ep_tripinfo = os.path.join(output_folder, f"tripinfo_ep_{ep}.xml")
        ep_summary  = os.path.join(output_folder, f"summary_ep_{ep}.xml")
        ep_vehroute = os.path.join(output_folder, f"vehroute_ep_{ep}.xml")

        # Environment (new SUMO instance per episode)
        env = TLSEnv(
            cfg=sumo_config,
            lane_groups=lg,
            step_len=1.0,
            reward_type="neg_wait",
            tripinfo=ep_tripinfo,
            summary=ep_summary,
            vehroute=ep_vehroute,
            teleport_time=600,
            use_gui=use_gui,
            gui_delay_ms=150,
        )

        env.reset_task(new_task)

        # Fixed-time strategy
        class Args:
            pass

        args = Args()
        args.g_ns = g_ns
        args.y_ns = y_ns
        args.g_ew = g_ew
        args.y_ew = y_ew

        strat = FixedTime(args, lg)

        # RESET environment
        state = env.reset()
        strat.on_phase(PHASE_NS)

        done = False
        step = 0

        while not done and step < max_steps:
            action = strat.act(state, state["phase"])
            state, reward, done, info = env.step(action)
            step += 1

        print(f"[EP {ep}] STEPS = {step}")
        print(f"[EP {ep}] XML written successfully.")

        # Close SUMO
        env.close()
        try:
            traci.close(False)
        except:
            pass

    print("\n========== DONE BASELINE ==========\n")


# ===========================================
#                 MAIN CALL
# ===========================================

if __name__ == "__main__":

    run_baseline_multi_episode(
        sumo_config="nets/demo/run_60p2k.sumocfg",
        output_folder="result_test/e",
        max_steps=4090,
        num_episodes=1,
        use_gui=True
    )
