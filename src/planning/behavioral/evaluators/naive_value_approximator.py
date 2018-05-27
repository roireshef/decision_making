import numpy as np
from logging import Logger

from decision_making.src.global_constants import BP_METRICS_LANE_DEVIATION_COST_WEIGHT, BP_MISSING_GOAL_COST, \
    BP_RIGHT_LANE_COST_WEIGHT, BP_EFFICIENCY_COST_WEIGHT, BP_CALM_LANE_CHANGE_TIME
from decision_making.src.planning.behavioral.evaluators.cost_functions import BP_EfficiencyMetric, BP_ComfortMetric
from decision_making.src.planning.behavioral.evaluators.value_approximator import ValueApproximator
from decision_making.src.planning.behavioral.data_objects import NavigationGoal, ActionSpec
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.types import FS_SX, FS_SV
from decision_making.src.planning.utils.map_utils import MapUtils
from decision_making.src.state.state import NewEgoState
from mapping.src.service.map_service import MapService


class NaiveValueApproximator(ValueApproximator):
    def __init__(self, logger: Logger):
        super().__init__(logger)
        self.logger = logger
        self.calm_lat_comfort_cost = None

    def approximate(self, behavioral_state: BehavioralGridState, goal: NavigationGoal) -> float:

        ego: NewEgoState = behavioral_state.ego_state
        ego_fstate = ego.map_state.road_state
        (ego_lane, ego_lon, ego_vel, ego_length, road_id) = \
            (ego.map_state.lane_id, ego_fstate[FS_SX], ego_fstate[FS_SV], ego.size.length, ego.map_state.road_id)
        TYPICAL_LANE_WIDTH = 3.5

        if self.calm_lat_comfort_cost is None:
            spec = ActionSpec(0, 0, 0, TYPICAL_LANE_WIDTH)
            _, self.calm_lat_comfort_cost = BP_ComfortMetric.calc_cost(ego_fstate, spec, BP_CALM_LANE_CHANGE_TIME)

        dist_to_goal = goal.lon - ego_lon  # TODO: use navigation plan
        # map_based_nav_plan = MapService.get_instance().get_road_based_navigation_plan(road_id)
        # dist_to_goal = MapService.get_instance().get_longitudinal_difference(
        #     road_id, ego_lon, goal.road_id, goal.lon, map_based_nav_plan)
        time_to_goal = dist_to_goal / max(ego_vel, 0.001)

        efficiency_cost = right_lane_cost = future_lane_deviation_cost = future_comfort_cost = goal_cost = 0
        if time_to_goal > 0:
            efficiency_cost = BP_EfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([ego_vel]))[0] * \
                              BP_EFFICIENCY_COST_WEIGHT * time_to_goal
            right_lane_cost = ego_lane * time_to_goal * BP_RIGHT_LANE_COST_WEIGHT

        if goal is None:  # no relevant goal
            if ego_lane > 0:  # distance in lanes from the rightest lane
                goal_cost = ego_lane * (BP_METRICS_LANE_DEVIATION_COST_WEIGHT + self.calm_lat_comfort_cost)
                future_lane_deviation_cost = future_comfort_cost = 0
        elif len(goal.lanes_list) > 0 and ego_lane not in goal.lanes_list:  # outside of the lanes range of the goal
            if ego_lon >= goal.lon:  # we missed the goal
                goal_cost = BP_MISSING_GOAL_COST
            else:  # still did not arrive to the goal
                lanes_from_goal = np.min(np.abs(np.array(goal.lanes_list) - ego_lane))
                T_d_max_per_lane = BP_CALM_LANE_CHANGE_TIME
                if ego_vel * lanes_from_goal > 0:
                    T_d_max_per_lane = (goal.lon - ego_lon) / (ego_vel * lanes_from_goal)  # required time for one lane change
                spec = ActionSpec(0, 0, 0, TYPICAL_LANE_WIDTH)
                _, goal_comfort_cost = lanes_from_goal * BP_ComfortMetric.calc_cost(ego_fstate, spec, T_d_max_per_lane)
                goal_cost = min(BP_MISSING_GOAL_COST, goal_comfort_cost)

        cost = efficiency_cost + future_comfort_cost + right_lane_cost + future_lane_deviation_cost + goal_cost

        # print('value=%.2f: eff=%.2f comf=%.2f rgt=%.2f dev=%.2f goal=%.2f' %
        #       (cost, efficiency_cost, future_comfort_cost, right_lane_cost, future_lane_deviation_cost, goal_cost))

        return cost
