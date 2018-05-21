import numpy as np
from logging import Logger

from decision_making.src.global_constants import BP_METRICS_LANE_DEVIATION_COST_WEIGHT, BP_MISSING_GOAL_COST, \
    BP_RIGHT_LANE_COST_WEIGHT, BP_EFFICIENCY_COST_WEIGHT, BP_CALM_LANE_CHANGE_TIME
from decision_making.src.planning.behavioral.evaluators.cost_functions import BP_EfficiencyMetric, BP_ComfortMetric
from decision_making.src.planning.behavioral.evaluators.value_approximator import ValueApproximator
from decision_making.src.planning.behavioral.data_objects import NavigationGoal, ActionSpec
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.utils.map_utils import MapUtils
from mapping.src.service.map_service import MapService


class NaiveValueApproximator(ValueApproximator):
    def __init__(self, logger: Logger):
        super().__init__(logger)
        self.logger = logger
        self.calm_lat_comfort_cost = None

    def approximate(self, behavioral_state: BehavioralGridState, goal: NavigationGoal) -> float:

        ego = behavioral_state.ego_state
        ego_loc = ego.road_localization
        (ego_lane, ego_lon, ego_length, road_id) = (ego_loc.lane_num, ego_loc.road_lon, ego.size.length, ego_loc.road_id)
        road_frenet = MapUtils.get_road_rhs_frenet(ego)
        ego_fstate = MapUtils.get_ego_road_localization(ego, road_frenet)
        lane_width = MapService.get_instance().get_road(ego_loc.road_id).lane_width

        if self.calm_lat_comfort_cost is None:
            spec = ActionSpec(0, 0, 0, lane_width)
            _, self.calm_lat_comfort_cost = BP_ComfortMetric.calc_cost(
                ego_fstate, spec, BP_CALM_LANE_CHANGE_TIME)

        map_based_nav_plan = MapService.get_instance().get_road_based_navigation_plan(road_id)
        # goal.lon - ego_loc.road_lon
        dist_to_goal = MapService.get_instance().get_longitudinal_difference(
            road_id, ego_loc.road_lon, goal.road_id, goal.lon, map_based_nav_plan)
        time_to_goal = dist_to_goal / max(ego.v_x, 0.001)

        efficiency_cost = right_lane_cost = future_lane_deviation_cost = future_comfort_cost = goal_cost = 0
        if time_to_goal > 0:
            efficiency_cost = BP_EfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([ego.v_x]))[0] * \
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
                if ego.v_x * lanes_from_goal > 0:
                    T_d_max_per_lane = (goal.lon - ego_lon) / (ego.v_x * lanes_from_goal)  # required time for one lane change
                spec = ActionSpec(0, 0, 0, lane_width)
                _, goal_comfort_cost = lanes_from_goal * BP_ComfortMetric.calc_cost(ego_fstate, spec, T_d_max_per_lane)
                goal_cost = min(BP_MISSING_GOAL_COST, goal_comfort_cost)

        cost = efficiency_cost + future_comfort_cost + right_lane_cost + future_lane_deviation_cost + goal_cost

        # print('value=%.2f: eff=%.2f comf=%.2f rgt=%.2f dev=%.2f goal=%.2f' %
        #       (cost, efficiency_cost, future_comfort_cost, right_lane_cost, future_lane_deviation_cost, goal_cost))

        return cost
