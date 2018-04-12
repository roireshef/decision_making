from enum import Enum
from typing import Tuple, List

from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory


class RelativeLane(Enum):
    """"
    The lane associated with a certain Recipe, relative to ego
    """
    LEFT_LANE = -1
    SAME_LANE = 0
    RIGHT_LANE = 1


class RelativeLongitudinalPosition(Enum):
    """"
    The high-level longitudinal position associated with a certain Recipe, relative to ego
    """
    REAR = -1
    PARALLEL = 0
    FRONT = 1


class ActionType(Enum):
    """"
    Type of Recipe, when "follow lane" is a static action while "follow vehicle" and "takeover vehicle" are dynamic ones.
    """
    FOLLOW_LANE = 1
    FOLLOW_VEHICLE = 2
    TAKE_OVER_VEHICLE = 3


class AggressivenessLevel(Enum):
    """"
    Aggressiveness driving level, which affects the urgency in reaching the specified goal.
    """
    CALM = 0
    STANDARD = 1
    AGGRESSIVE = 2


# Define semantic cell
SemanticGridCell = Tuple[int, int]

# tuple indices
LAT_CELL, LON_CELL = 0, 1


class ActionRecipe:
    def __init__(self, relative_lane: RelativeLane, action_type: ActionType, aggressiveness: AggressivenessLevel):
        self.relative_lane = relative_lane
        self.action_type = action_type
        self.aggressiveness = aggressiveness

    @classmethod
    def from_args_list(cls, args: List):
        return cls(*args)


class StaticActionRecipe(ActionRecipe):
    """"
    Data object containing the fields needed for specifying a certain static action, together with the state.
    """
    def __init__(self, relative_lane: RelativeLane, velocity: float, aggressiveness: AggressivenessLevel):
        super().__init__(relative_lane, ActionType.FOLLOW_LANE, aggressiveness)
        self.velocity = velocity


class DynamicActionRecipe(ActionRecipe):
    """"
    Data object containing the fields needed for specifying a certain dynamic action, together with the state.
    """
    def __init__(self, relative_lane: RelativeLane, relative_lon: RelativeLongitudinalPosition,  action_type: ActionType, aggressiveness: AggressivenessLevel):
        super().__init__(relative_lane, action_type, aggressiveness)
        self.relative_lon = relative_lon


class ActionSpec:
    """
    Holds the actual translation of the semantic action in terms of trajectory specifications.
    """

    def __init__(self, t: float, v: float, s: float, d: float, samplable_trajectory: SamplableTrajectory = None):
        """
        The trajectory specifications are defined by the target ego state
        :param t: time [sec]
        :param v: velocity [m/s]
        :param s: relative longitudinal distance to ego in Frenet frame [m]
        :param d: relative lateral distance to ego in Frenet frame [m]
        :param samplable_trajectory: samplable reference trajectory.
        """
        self.t = t
        self.v = v
        self.s = s
        self.d = d
        self.samplable_trajectory = samplable_trajectory

    def __str__(self):
        return str({k: str(v) for (k, v) in self.__dict__.items()})