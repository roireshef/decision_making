import numpy as np
from enum import Enum
from typing import List

from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.state.map_state import MapState


class GoalStatus(Enum):
    """
    status of achieving goal
    """
    REACHED = 1
    MISSED = 2
    NOT_YET = 3


class NavigationGoal:
    def __init__(self, road_id: int, lon: float, lanes: List[int]):
        """
        Holds parameters of a navigation goal: road id, longitude, list of lanes.
        :param road_id: road id from the map
        :param lon: [m] longitude of the goal relatively to the road's beginning
        :param lanes: list of lane indices of the goal
        """
        self.road_id = road_id
        self.lon = lon
        self.lanes = lanes

    def validate(self, samplable_trajectory: SamplableTrajectory, next_state_time: float) -> GoalStatus:
        """
        check if given samplable trajectory at next state time reaches misses or not reaches the goal
        :param samplable_trajectory: samplable trajectory (from TP)
        :param next_state_time: [sec] global time of the next state
        :return: GoalStatus (REACHED, MISSED or NOT_YET)
        """
        # reaching the goal may be validated only for SamplableWerlingTrajectory
        if type(samplable_trajectory) != SamplableWerlingTrajectory:
            return GoalStatus.NOT_YET

        goal_time = samplable_trajectory.get_time_from_longitude(self.lon)
        if goal_time is not None:
            if next_state_time < goal_time:
                return GoalStatus.NOT_YET
            fstate_at_goal = samplable_trajectory.sample_frenet(np.array([goal_time]))[0]
            if MapState(fstate_at_goal, self.road_id).lane_num in self.lanes:
                return GoalStatus.REACHED
            else:
                return GoalStatus.MISSED
        else:
            return GoalStatus.NOT_YET
