import numpy as np
from logging import Logger

from typing import Optional

from decision_making.src.global_constants import WERLING_TIME_RESOLUTION, EFFICIENCY_COST_WEIGHT, \
    RIGHT_LANE_COST_WEIGHT, BP_METRICS_LANE_DEVIATION_COST_WEIGHT, BP_MISSING_GOAL_COST
from decision_making.src.planning.behavioral.architecture.components.evaluators.cost_functions import \
    PlanEfficiencyMetric, PlanComfortMetric
from decision_making.src.planning.behavioral.architecture.components.evaluators.velocity_profile import VelocityProfile
from decision_making.src.planning.behavioral.architecture.data_objects import AggressivenessLevel, NavigationGoal
from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState
from mapping.src.model.localization import RoadLocalization


class ValueApproximator:
    def __init__(self, logger: Logger):
        self.logger = logger

    #def evaluate_state(self, behavioral_state: SemanticBehavioralGridState) -> float:
    def evaluate_state(self, road_localization: RoadLocalization, v_tar: float, time_period: float, T_d_full: float,
                       lane_width: float, goal: Optional[NavigationGoal]) -> float:

        ego_road_id = road_localization.road_id
        ego_lane = road_localization.lane_num
        ego_lon = road_localization.road_lon
        rel_ego_lat = road_localization.intra_road_lat / lane_width
        empty_vel_profile = VelocityProfile(0, 0, 0, 0, 0, 0)

        time_period = max(0., time_period)
        efficiency_cost = right_lane_cost = future_lane_deviation_cost = future_comfort_cost = goal_cost = 0
        if time_period > 0:
            efficiency_cost = PlanEfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([v_tar]))[0] * \
                              EFFICIENCY_COST_WEIGHT * time_period / WERLING_TIME_RESOLUTION
            right_lane_cost = ego_lane * time_period * RIGHT_LANE_COST_WEIGHT / WERLING_TIME_RESOLUTION

        if goal is None or ego_road_id != goal.road_id:  # no relevant goal
            if ego_lane > 0:  # distance in lanes from the rightest lane
                future_lane_deviation_cost = ego_lane * BP_METRICS_LANE_DEVIATION_COST_WEIGHT / WERLING_TIME_RESOLUTION
                future_comfort_cost = ego_lane * PlanComfortMetric.calc_cost(empty_vel_profile, T_d_full, T_d_full, AggressivenessLevel.CALM)
        elif ego_lane > goal.to_lane or ego_lane < goal.from_lane:  # outside of the lanes range of the goal
            if ego_lon >= goal.lon:  # we missed the goal
                goal_cost = BP_MISSING_GOAL_COST
            else:  # still did not arrive to the goal
                lanes_from_goal = max(rel_ego_lat - goal.to_lane, goal.from_lane - rel_ego_lat)
                T_d_max_per_lane = np.inf
                if v_tar * lanes_from_goal > 0:
                    T_d_max_per_lane = (goal.lon - ego_lon) / (v_tar * lanes_from_goal)  # required time for one lane change
                goal_comfort_cost = lanes_from_goal * PlanComfortMetric.calc_cost(
                    empty_vel_profile, T_d_full, T_d_max_per_lane, AggressivenessLevel.CALM)
                goal_cost = min(BP_MISSING_GOAL_COST, goal_comfort_cost)

        cost = efficiency_cost + future_comfort_cost + right_lane_cost + future_lane_deviation_cost + goal_cost
        return cost
