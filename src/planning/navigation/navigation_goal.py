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
    def __init__(self, road_segment_id: int, lon: float, lanes_idxs: List[int]):
        """
        Holds parameters of a navigation goal: road id, longitude, list of lanes.
        :param road_segment_id: road segment id from the map
        :param lon: [m] longitude of the goal relatively to the road's beginning
        :param lanes_idxs: list of lane indices of the goal
        """
        # TODO: replace road & lane indices by list of lane_ids and lon will be per lane.
        self.road_segment_id = road_segment_id
        self.lon = lon
        self.lanes_idxs = lanes_idxs

    def validate(self, state: State) -> GoalStatus:
        """
        check if the given state reached missed or yet not reached the goal
        :param state: the (next) State
        :return: GoalStatus (REACHED, MISSED or NOT_YET)
        """
        # TODO: use route planner to check whether current road_id != goal.road means MISSED or NOT_YET
        map_state = state.ego_state.map_state
        road_segment_id = MapUtils.get_road_segment_id_from_lane_id(map_state.lane_id)
        # TODO: decide relatively to which lane self.lon is given
        if road_segment_id == self.road_segment_id and map_state.lane_fstate[FS_SX] >= self.lon:
            if MapUtils.get_lane_ordinal(map_state.lane_id) in self.lanes_idxs:
                return GoalStatus.REACHED
            else:
                return GoalStatus.MISSED
        else:
            return GoalStatus.NOT_YET
