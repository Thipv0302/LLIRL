# controller/inspect.py
import traci, sys

def run(cfg="run.sumocfg", step_len=1.0):
    cmd = ["sumo", "-c", cfg, "--step-length", str(step_len)]
    traci.start(cmd)
    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids:
        traci.close(); raise RuntimeError("No traffic light found!")
    print(f"[INSPECT] TLS IDs: {list(tls_ids)}")
    for tls in tls_ids:
        print(f"\n=== TLS: {tls} ===")
        prog = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)
        for p in prog:
            print(f" programID={p.programID}, type={p.type}, phases={len(p.phases)}")
            for i, ph in enumerate(p.phases):
                print(f"  - phase {i}: duration={ph.duration}, state={ph.state}")
    traci.close()
    print("\n[INSPECT] OK. TLS programs are readable; no obvious errors.")

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "run.sumocfg"
    run(cfg)
