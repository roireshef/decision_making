from enum import Enum
from typing import List

from decision_making.src.planning.behavioral.behavioral_grid_state import RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory


class ActionType(Enum):
    """"
    Type of Recipe, when "follow lane" is a static action while "follow vehicle" and "takeover vehicle" are dynamic ones.
    """
    FOLLOW_LANE = 1
    FOLLOW_VEHICLE = 2
    OVERTAKE_VEHICLE = 3


class AggressivenessLevel(Enum):
    """"
    Aggressiveness driving level, which affects the urgency in reaching the specified goal.
    """
    CALM = 0
    STANDARD = 1
    AGGRESSIVE = 2


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
    def __init__(self, t: float, v: float, s: float, d: float):
        """
        The trajectory specifications are defined by the target ego state
        :param t: time [sec]
        :param v: velocity [m/s]
        :param s: global longitudinal position in Frenet frame [m]
        :param d: global lateral position in Frenet frame [m]
        """
        self.t = t
        self.v = v
        self.s = s
        self.d = d

    def __str__(self):
        return str({k: str(v) for (k, v) in self.__dict__.items()})

class NavigationGoal:
    def __init__(self, road_id: int, segment_id: int, lon: float, lanes: List[int]):
        self.road_id = road_id
        self.segment_id = segment_id
        self.lon = lon
        self.lanes_list = lanes
