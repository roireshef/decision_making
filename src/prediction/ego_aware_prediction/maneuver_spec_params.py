from decision_making.src.messages.str_serializable import StrSerializable


class ManeuverSpecParams(StrSerializable):
    def __init__(self, T_d: float, avg_s_a: float, s_a_final: float, relative_lane: float, lat_normalized: float):
        """
        Parametrization of Frenet based maneuver
        :param T_d: lateral maneuver duration in [sec]
        :param avg_s_a: average acceleration in [m/s^2]
        :param s_a_final: final acceleration in [m/s^2]
        :param relative_lane: relative lane (integer in [lanes] units)
        :param lat_normalized: lateral (normalized) position within lane. Between [-0.5, 0.5]
        """
        self.lat_normalized = lat_normalized
        self.relative_lane = relative_lane
        self.s_a_final = s_a_final
        self.avg_s_a = avg_s_a
        self.T_d = T_d