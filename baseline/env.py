import traci
import random
from baseline.strategies import (
    PHASE_NS, PHASE_Y_NS, PHASE_EW, PHASE_Y_EW,
    KEEP, SWITCH
)


class TLSEnv:
    def __init__(
        self,
        cfg="run.sumocfg",
        lane_groups=None,
        step_len=1.0,
        reward_type="neg_wait",
        tripinfo=None,
        summary=None,
        vehroute=None,
        teleport_time=None,
        extra_sumo_args=None,
        use_gui=False,
        gui_delay_ms=100,
    ):
        self.cfg = cfg
        self.lg = lane_groups
        self.step_len = step_len
        self.reward_type = reward_type
        self.tripinfo = tripinfo
        self.summary = summary
        self.vehroute = vehroute
        self.teleport_time = teleport_time

        self.extra_sumo_args = extra_sumo_args or []
        self.use_gui = use_gui
        self.gui_delay_ms = gui_delay_ms
        self.sumo_running = False

        # Default task params
        self.task_params = [1.0, 0.0]  # [inten, var]

        # Intersection variables
        self.tls = None
        self.phase = None
        self.switch_count = 0

    # =====================================================
    #                  TASK API
    # =====================================================
    def reset_task(self, task_params):
        """
        task_params = [inten, var]
        inten: scale traffic
        var:   randomness (unused for now)
        """
        self.task_params = task_params

    # =====================================================
    #               START SUMO PER EPISODE
    # =====================================================
    def _start_sumo(self):
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass

        inten, var = self.task_params

        sumo_bin = "sumo-gui" if self.use_gui else "sumo"
        cmd = [sumo_bin, "-c", self.cfg, "--step-length", str(self.step_len)]

        # GUI: auto-start + slow down to observe
        if self.use_gui:
            cmd += ["--start", "--delay", str(self.gui_delay_ms)]

        # Episode XML outputs
        if self.tripinfo:
            cmd += ["--tripinfo-output", self.tripinfo]
        if self.summary:
            cmd += ["--summary-output", self.summary]
        if self.vehroute:
            cmd += ["--vehroute-output", self.vehroute]

        # Prevent teleport if defined
        if self.teleport_time is not None:
            cmd += ["--time-to-teleport", str(self.teleport_time)]

        # SUMO scaling — identical to DDQN & LLIRL
        cmd += ["--scale", str(inten)]

        cmd += self.extra_sumo_args

        print(f"[SUMO] Starting with task inten={inten}, var={var}")
        traci.start(cmd)
        self.sumo_running = True

    # =====================================================
    #                      RESET
    # =====================================================
    def reset(self):
        self._start_sumo()

        tls_list = traci.trafficlight.getIDList()
        if not tls_list:
            traci.close()
            raise RuntimeError("No traffic light found!")

        self.tls = tls_list[0]
        # # ===== DEBUG: print phase table =====
        # logics = traci.trafficlight.getAllProgramLogics(self.tls)
        # logic = logics[0]
        # for i, ph in enumerate(logic.phases):
        #     print(i, ph.duration, ph.state)
        # # ===================================

        traci.trafficlight.setPhase(self.tls, PHASE_NS)
        self.phase = PHASE_NS
        self.switch_count = 0

        return self._get_state()

    # =====================================================
    #         TRAFFIC INJECTOR — DISABLED (DDQN STYLE)
    # =====================================================
    def _apply_task_traffic(self):
        """
        Injector disabled.
        DDQN & LLIRL DO NOT spawn extra vehicles,
        they only use SUMO scaling (--scale).
        Baseline must match that behavior.
        """
        return

    # =====================================================
    #                       STEP
    # =====================================================
    def step(self, action):
        # Handle SWITCH action
        if action == SWITCH:
            if self.phase == PHASE_NS:
                self._set_phase(PHASE_Y_NS)
            elif self.phase == PHASE_EW:
                self._set_phase(PHASE_Y_EW)
            elif self.phase == PHASE_Y_NS:
                self._set_phase(PHASE_EW)
            elif self.phase == PHASE_Y_EW:
                self._set_phase(PHASE_NS)

            self.switch_count += 1

        traci.simulationStep()
        # DEBUG: in phase hiện tại mỗi step
        # print("phase=", traci.trafficlight.getPhase(self.tls))
        # # Auto-yellow transition
        # p = traci.trafficlight.getPhase(self.tls)
        # print("phase=", traci.trafficlight.getPhase(self.tls))

        # if p == PHASE_Y_NS:
        #     self._set_phase(PHASE_EW)
        # elif p == PHASE_Y_EW:
        #     self._set_phase(PHASE_NS)

        # Injector disabled
        # self._apply_task_traffic()

        # Read state
        state = self._get_state()
        reward = self._reward(state)

        # Episode termination
        done = traci.simulation.getMinExpectedNumber() == 0

        info = {
            "phase": self.phase,
            "switch_count": self.switch_count,
            "task": self.task_params
        }

        return state, reward, done, info

    # =====================================================
    #                    HELPERS
    # =====================================================
    def _set_phase(self, phase):
        traci.trafficlight.setPhase(self.tls, phase)

        # ép duration theo đúng tlLogic
        logic = traci.trafficlight.getAllProgramLogics(self.tls)[0]
        dur = logic.phases[phase].duration
        traci.trafficlight.setPhaseDuration(self.tls, dur)

        self.phase = phase

    def _get_state(self):
        q_ns = self.lg.halting_on(traci, self.lg.IN_NS)
        q_ew = self.lg.halting_on(traci, self.lg.IN_EW)
        v_ns = self.lg.veh_on(traci, self.lg.IN_NS)
        v_ew = self.lg.veh_on(traci, self.lg.IN_EW)
        p_ns = q_ns - self.lg.veh_on(traci, self.lg.OUT_NS)
        p_ew = q_ew - self.lg.veh_on(traci, self.lg.OUT_EW)

        return {
            "q_ns": q_ns, "q_ew": q_ew,
            "v_ns": v_ns, "v_ew": v_ew,
            "p_ns": p_ns, "p_ew": p_ew,
            "phase": self.phase
        }

    def _reward(self, s):
        if self.reward_type == "neg_wait":
            return -(s["q_ns"] + s["q_ew"])

        if self.reward_type == "neg_queue_sq":
            q = s["q_ns"] + s["q_ew"]
            return -(q * q)

        if self.reward_type == "pressure":
            return s["p_ns"] + s["p_ew"]

        # Default reward
        return -(s["q_ns"] + s["q_ew"])

    def close(self):
        try:
            traci.close()
        except:
            pass
        self.sumo_running = False
