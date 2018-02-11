from enum import Enum


class RelativeLane(Enum):
    LEFT_LANE = -1
    SAME_LANE = 0
    RIGHT_LANE = 1


class RelativeLongitudinalPosition(Enum):
    REAR = -1
    PARALLEL = 0
    FRONT = 1


class ActionType(Enum):
    FOLLOW_LANE = 1
    FOLLOW_VEHICLE = 2
    TAKE_OVER_VEHICLE = 3


class AggressivenessLevel(Enum):
    CALM = 1
    AGGRESSIVE = 2


class ActionRecipe:
    def __init__(self, action_type: ActionType, lane: RelativeLane, aggressiveness: AggressivenessLevel):
        self.action_type = action_type
        self.lane = lane
        self.aggressiveness = aggressiveness


class StaticActionRecipe(ActionRecipe):
    def __init__(self, lane: RelativeLane, velocity: float, aggressiveness: AggressivenessLevel):
        super().__init__(ActionType.FOLLOW_LANE, lane, aggressiveness)
        self.velocity = velocity


class DynamicActionRecipe(ActionRecipe):
    def __init__(self, action_type: ActionType, lane: RelativeLane, aggressiveness: AggressivenessLevel,
                 lon_position: RelativeLongitudinalPosition):
        super().__init__(action_type, lane, aggressiveness)
        self.lon_position = lon_position


class ActionSpec:
    """
    Holds the actual translation of the semantic action in terms of trajectory specifications.
    """

    def __init__(self, t: float, v: float, s_rel: float, d_rel: float):
        """
        The trajectory specifications are defined by the target ego state
        :param t: time [sec]
        :param v: velocity [m/s]
        :param s_rel: relative longitudinal distance to ego in Frenet frame [m]
        :param d_rel: relative lateral distance to ego in Frenet frame [m]
        """
        self.t = t
        self.v = v
        self.s_rel = s_rel
        self.d_rel = d_rel
