from enum import Enum
from typing import List, Dict

from decision_making.src.planning.types import FS_SX
from decision_making.src.state.state import State
from decision_making.src.utils.map_utils import MapUtils


class GoalStatus(Enum):
    """
    status of achieving goal
    """
    REACHED = 1
    MISSED = 2
    NOT_YET = 3


class NavigationGoal:
    def __init__(self, road_segment_id: int, goal_longitude_per_ordinal: Dict[int, float]):
        """
        Holds parameters of a navigation goal: road id, longitude, list of lanes.
        :param road_segment_id: road segment id from the map
        :param goal_longitude_per_ordinal: a mapping between an ordinal in the goal road_segment and the longitude of the goal
                                relatively to that lane's beginning
        """
        self.road_segment_id = road_segment_id
        self.goal_longitude_per_ordinal = goal_longitude_per_ordinal

    def validate(self, state: State) -> GoalStatus:
        """
        check if the given state reached missed or yet not reached the goal
        :param state: the (next) State
        :return: GoalStatus (REACHED, MISSED or NOT_YET)
        """
        # TODO: use route planner to check whether current road_id != goal.road means MISSED or NOT_YET
        map_state = state.ego_state.map_state
        road_segment_id = MapUtils.get_road_segment_id_from_lane_id(map_state.lane_id)
        ego_ordinal = MapUtils.get_lane_ordinal(map_state.lane_id)
        if road_segment_id == self.road_segment_id and map_state.lane_fstate[FS_SX] >= self.goal_longitude_per_ordinal[ego_ordinal]:
            if ego_ordinal in self.goal_longitude_per_ordinal.keys():
                return GoalStatus.REACHED
            else:
                return GoalStatus.MISSED
        else:
            return GoalStatus.NOT_YET
