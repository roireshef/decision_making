import numpy as np
from logging import Logger

from decision_making.src.exceptions import MissingNavigationGoal
from decision_making.src.global_constants import BP_METRICS_LANE_DEVIATION_COST_WEIGHT, BP_MISSING_GOAL_COST, \
    BP_RIGHT_LANE_COST_WEIGHT, BP_EFFICIENCY_COST_WEIGHT, BP_CALM_LANE_CHANGE_TIME
from decision_making.src.planning.behavioral.evaluators.cost_functions import BP_CostFunctions
from decision_making.src.planning.behavioral.evaluators.value_approximator import ValueApproximator
from decision_making.src.planning.behavioral.data_objects import NavigationGoal, ActionSpec
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.types import FS_SV, FS_SX, FrenetState2D
from decision_making.src.planning.utils.map_utils import MapUtils
from mapping.src.service.map_service import MapService


class NaiveValueApproximator(ValueApproximator):
    """
    In this value function approximator we assume the agent continues to move with constant velocity on the same lane
    until reaching the goal.
    """
    def __init__(self, logger: Logger):
        super().__init__(logger)
        self.logger = logger

    def approximate(self, behavioral_state: BehavioralGridState, goal: NavigationGoal) -> float:
        """
        Naive approximation of value function for given state and goal.
        Here we assume the agent continues to move with constant velocity on the same lane until reaching the goal.
        :param behavioral_state: terminal behavioral state after an action
        :param goal: navigation goal
        :return: value function approximation
        """
        ego = behavioral_state.ego_state
        ego_loc = ego.road_localization
        (ego_lane, ego_lon, ego_length, road_id) = (ego_loc.lane_num, ego_loc.road_lon, ego.size.length, ego_loc.road_id)
        road_frenet = MapUtils.get_road_rhs_frenet(ego)
        ego_fstate = MapUtils.get_ego_road_localization(ego, road_frenet)
        lane_width = MapService.get_instance().get_road(ego_loc.road_id).lane_width

        # calculate time to goal
        map_based_nav_plan = MapService.get_instance().get_road_based_navigation_plan(road_id)
        # goal.lon - ego_loc.road_lon
        dist_to_goal = MapService.get_instance().get_longitudinal_difference(
            road_id, ego_loc.road_lon, goal.road_id, goal.lon, map_based_nav_plan)
        if ego.v_x > 0:
            time_to_goal = dist_to_goal / ego.v_x
        else:
            return np.inf

        # calculate efficiency and non-right lane costs
        efficiency_cost = NaiveValueApproximator._calc_efficiency_cost(ego.v_x, time_to_goal)
        right_lane_cost = NaiveValueApproximator._calc_rightlane_cost(ego_lane, time_to_goal)

        # calculate lane deviation and comfort cost for reaching the goal
        # in case of missing the goal, there is a missing goal cost
        comfort_cost = 0
        if len(goal.lanes_list) > 0 and ego_lane not in goal.lanes_list:  # outside of the lanes range of the goal
            if ego_lon >= goal.lon:  # we missed the goal
                raise MissingNavigationGoal("Missing a navigation goal on road_id=%d, longitude=%.2f, lanes=%s; "
                                            "ego_lon=%.2f ego_lane=%d" %
                                            (goal.road_id, goal.lon, goal.lanes_list, ego_lon, ego_lane))
            else:  # if still did not arrive to the goal, calculate lateral comfort for reaching the goal
                lanes_from_goal = np.min(np.abs(np.array(goal.lanes_list) - ego_lane))
                comfort_cost = NaiveValueApproximator._calc_comfort_cost(lanes_from_goal, goal.lon, ego_fstate,
                                                                         lane_width)

        cost = efficiency_cost + comfort_cost + right_lane_cost

        return cost

    @staticmethod
    def _calc_efficiency_cost(ego_vel: float, time_to_goal: float):
        if time_to_goal <= 0:
            return 0
        return BP_CostFunctions.calc_efficiency_cost_for_velocities(np.array([ego_vel]))[0] * \
               BP_EFFICIENCY_COST_WEIGHT * time_to_goal

    @staticmethod
    def _calc_rightlane_cost(ego_lane: int, time_to_goal: float):
        if time_to_goal <= 0:
            return 0
        return ego_lane * time_to_goal * BP_RIGHT_LANE_COST_WEIGHT

    @staticmethod
    def _calc_lane_deviation_cost(lanes_from_goal: int):
        return BP_METRICS_LANE_DEVIATION_COST_WEIGHT * lanes_from_goal

    @staticmethod
    def _calc_comfort_cost(lanes_from_goal: int, goal_lon: float, ego_fstate: FrenetState2D, lane_width: float):
        # calculate required time for one lane change
        if lanes_from_goal == 0 or ego_fstate[FS_SV] <= 0:
            return 0
        T_d_max_per_lane = (goal_lon - ego_fstate[FS_SX]) / (ego_fstate[FS_SV] * lanes_from_goal)
        if T_d_max_per_lane >= BP_CALM_LANE_CHANGE_TIME:
            return 0
        spec = ActionSpec(0, 0, 0, lane_width)
        _, goal_comfort_cost = lanes_from_goal * BP_CostFunctions.calc_comfort_cost(ego_fstate, spec, 0, T_d_max_per_lane)
        return min(BP_MISSING_GOAL_COST, goal_comfort_cost)
