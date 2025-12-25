# ================================
#       strategies.py
# ================================

PHASE_NS, PHASE_Y_NS, PHASE_EW, PHASE_Y_EW = 0, 1, 2, 3
KEEP, SWITCH = 0, 1

class StrategyBase:
    def __init__(self, args, lane_groups):
        self.args = args
        self.lg = lane_groups
        self.green_since = 0
        self.t = 0

    def on_phase(self, phase):
        self.green_since = 0

    def act(self, state, phase):
        raise NotImplementedError


class FixedTime(StrategyBase):
    def __init__(self, args, lane_groups):
        super().__init__(args, lane_groups)
        self.plan = [
            (PHASE_NS, args.g_ns),
            (PHASE_Y_NS, args.y_ns),
            (PHASE_EW, args.g_ew),
            (PHASE_Y_EW, args.y_ew),
        ]
        self.idx = 0
        self.left = self.plan[0][1]

    def act(self, state, phase):
        self.green_since += 1
        self.left -= 1

        if self.left <= 0:
            self.idx = (self.idx + 1) % len(self.plan)
            self.left = self.plan[self.idx][1]
            return SWITCH
        return KEEP
