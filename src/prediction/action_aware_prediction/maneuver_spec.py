from decision_making.src.planning.types import FrenetState2D


class ManeuverSpec:
    def __init__(self, init_state: FrenetState2D, final_state: FrenetState2D, T_s: float, T_d: float):
        """
        :param T_d: time horizon to complete the action in the d (lateral) axis. In [sec]
        :param T_s: time horizon to complete the action in the s (longitudinal) axis. In [sec]
        :param init_state: initial Frenet state of object
        :param final_state: final Frenet state of object
        """
        self.T_d = T_d
        self.T_s = T_s
        self.final_state = final_state
        self.init_state = init_state

