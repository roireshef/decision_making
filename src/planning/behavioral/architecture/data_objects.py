from enum import Enum
from typing import Tuple, TypeVar, Union

from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import QuarticPoly1D, QuinticPoly1D
from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory


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
    CALM = 0
    STANDARD = 1
    AGGRESSIVE = 2


# Define semantic cell
SemanticGridCell = Tuple[int, int]

# tuple indices
LAT_CELL, LON_CELL = 0, 1


class ActionRecipe:
    def __init__(self, rel_lane: RelativeLane, action_type: ActionType, aggressiveness: AggressivenessLevel):
        self.rel_lane = rel_lane
        self.action_type = action_type
        self.aggressiveness = aggressiveness


class StaticActionRecipe(ActionRecipe):
    def __init__(self, rel_lane: RelativeLane, velocity: float, aggressiveness: AggressivenessLevel):
        super().__init__(rel_lane, ActionType.FOLLOW_LANE, aggressiveness)
        self.velocity = velocity


class DynamicActionRecipe(ActionRecipe):
    def __init__(self, rel_lane: RelativeLane, rel_lon: RelativeLongitudinalPosition,  action_type: ActionType, aggressiveness: AggressivenessLevel):
        super().__init__(rel_lane, action_type, aggressiveness)
        self.rel_lon = rel_lon


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
