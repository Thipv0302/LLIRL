# controller/lanes.py
class LaneGroups:
    """
    Gom theo HƯỚNG (tối giản 2-pha: NS vs EW).
    Nếu bạn có nhiều lane per edge, có thể thay edge->lane tuỳ ý:
      self.IN_NS_LANES = ["N_in_0","N_in_1","S_in_0",...]
    """
    def __init__(self):
        self.IN_NS  = ["N_in","S_in"]
        self.OUT_NS = ["N_out","S_out"]
        self.IN_EW  = ["E_in","W_in"]
        self.OUT_EW = ["E_out","W_out"]

    # Helpers dùng trong env/strategies
    @staticmethod
    def halting_on(traci, edges):
        return sum(traci.edge.getLastStepHaltingNumber(e) for e in edges)

    @staticmethod
    def veh_on(traci, edges):
        return sum(traci.edge.getLastStepVehicleNumber(e) for e in edges)
