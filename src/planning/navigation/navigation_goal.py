from enum import Enum
from typing import List

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
    def __init__(self, road_id: int, lon: float, lane_indices: List[int]):
        """
        Holds parameters of a navigation goal: road id, longitude, list of lanes.
        :param road_id: road id from the map
        :param lon: [m] longitude of the goal relatively to the road's beginning
        :param lane_indices: list of lane indices of the goal
        """
        self.road_id = road_id
        self.lon = lon
        self.lane_indices = lane_indices

    def validate(self, state: State) -> GoalStatus:
        """
        check if the given state reached missed or yet not reached the goal
        :param state: the (next) State
        :return: GoalStatus (REACHED, MISSED or NOT_YET)
        """
        # TODO: use route planner to check whether current road_id != goal.road means MISSED or NOT_YET
        map_state = state.ego_state.map_state
        road_id = MapUtils.get_road_by_lane(map_state.lane_id)
        if road_id == self.road_id and map_state.lane_fstate[FS_SX] >= self.lon:
            if map_state.lane_index in self.lane_indices:
                return GoalStatus.REACHED
            else:
                return GoalStatus.MISSED
        else:
            return GoalStatus.NOT_YET
