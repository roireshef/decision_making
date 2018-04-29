from logging import Logger
from typing import List, Optional

import numpy as np

from decision_making.src.global_constants import SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_SAME, \
    SEMANTIC_CELL_LAT_SAME, BP_METRICS_TIME_HORIZON, SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LAT_LEFT
from decision_making.src.global_constants import SEMANTIC_CELL_LON_REAR
from decision_making.src.planning.behavioral.architecture.components.evaluators.cost_functions import \
    PlanEfficiencyMetric, \
    PlanComfortMetric, PlanRightLaneMetric, VelocityProfile, PlanLaneDeviationMetric
from decision_making.src.planning.behavioral.architecture.components.evaluators.state_action_evaluator import \
    StateActionRecipeEvaluator
from decision_making.src.planning.behavioral.architecture.components.evaluators.value_approximator import \
    ValueApproximator
from decision_making.src.planning.behavioral.architecture.components.evaluators.velocity_profile import ProfileSafety
from decision_making.src.planning.behavioral.architecture.data_objects import ActionRecipe, ActionType, \
    AggressivenessLevel
from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState
from decision_making.src.planning.types import FP_SX, FP_DX
from mapping.src.service.map_service import MapService


class HeuristicStateActionRecipeEvaluator(StateActionRecipeEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate_recipes(self, behavioral_state: SemanticBehavioralGridState, action_recipes: List[ActionRecipe],
                         action_recipes_mask: List[bool]) -> np.ndarray:
        """
        Gets a list of actions to evaluate and returns a vector representing their costs.
        A set of actions is provided, enabling us to assess them independently.
        Note: the semantic actions were generated using the behavioral state and don't necessarily capture
        all relevant details in the scene. Therefore the evaluation is done using the behavioral state.
        :param behavioral_state: semantic actions grid behavioral state
        :param action_recipes: array of semantic actions
        :param action_recipes_mask: array of boolean values indicating whether the recipe was filtered or not
        :return: array of costs (one cost per action)
        """
        ego = behavioral_state.ego_state
        ego_lane = ego.road_localization.lane_num
        ego_lat_vel = ego.v_x * np.sin(ego.road_localization.intra_road_yaw)
        lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width

        # calculate full lane change time with (calm)
        T_d_full = VelocityProfile.calc_lateral_time(0, lane_width, lane_width, AggressivenessLevel.CALM)

        # get front dynamic objects' velocities from the occupancy grid
        front_objects_vel = self._get_front_objects_velocities(behavioral_state)

        print('time=%f ego_v=%f ego_lat=%f' % (ego.timestamp_in_sec, ego.v_x, ego.road_localization.intra_road_lat))

        action_costs = np.full(len(action_recipes), np.inf)

        for i, action_recipe in enumerate(action_recipes):
            if not action_recipes_mask[i]:
                continue

            target_lane = ego_lane + action_recipe.relative_lane.value
            lat_dist = (target_lane + 0.5) * lane_width - ego.road_localization.intra_road_lat
            # calculate lateral time according to the given aggressiveness level
            T_d = VelocityProfile.calc_lateral_time(ego_lat_vel, lat_dist, lane_width, action_recipe.aggressiveness)

            # create velocity profile, whose length is at least as the lateral time
            vel_profile = self._calc_velocity_profile(behavioral_state, action_recipe, T_d, front_objects_vel)
            if vel_profile is None:  # infeasible action
                continue

            # calculate the latest safe time
            T_d_max = self._calc_largest_safe_time(
                behavioral_state, action_recipe.relative_lane, vel_profile, ego.size.length / 2, T_d)
            if T_d_max <= 0:
                continue

            action_costs[i], sub_costs = self._calc_action_cost(vel_profile, target_lane, T_d, T_d_max, T_d_full,
                                                                action_recipe.aggressiveness)

            # FOR DEBUGGING
            dist = np.inf
            if action_recipe.action_type != ActionType.FOLLOW_LANE:
                target_obj = behavioral_state.road_occupancy_grid[(action_recipe.relative_lane.value, action_recipe.relative_lon.value)][0]
                dist = target_obj.road_localization.road_lon - ego.road_localization.road_lon
            print('action %d type %d lane %d: dist=%.1f tar_vel=%.2f [eff %.3f comf %.3f right %.2f dev %.2f value %.2f]; prof_time=%.2f safe_time=%.2f: tot %.2f' %
                  (i, action_recipe.action_type.value, action_recipe.relative_lane.value, dist, vel_profile.v_tar,
                   sub_costs[0], sub_costs[1], sub_costs[2], sub_costs[3], sub_costs[4], vel_profile.total_time(),
                   T_d_max, action_costs[i]))

        # FOR DEBUGGING
        best_action = int(np.argmin(action_costs))
        print('Best action %d; lane %d\n' % (best_action, ego_lane + action_recipes[best_action].relative_lane.value))

        return action_costs

    def _get_front_objects_velocities(self, behavioral_state: SemanticBehavioralGridState) -> np.array:
        """
        given occupancy grid, retrieve velocities of at most 3 front objects (from 3 relevant lanes)
        :param behavioral_state: behavioral state containing the occupancy grid
        :return: array of the front objects' velocities. If some object is missing, its velocity is np.inf
        """
        # get front dynamic objects from the occupancy grid
        front_objects_vel = np.array([np.inf, np.inf, np.inf])
        for lat in [SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LAT_LEFT]:
            if (lat, SEMANTIC_CELL_LON_FRONT) in behavioral_state.road_occupancy_grid:
                front_objects_vel[lat - SEMANTIC_CELL_LAT_RIGHT] = \
                    behavioral_state.road_occupancy_grid[(lat, SEMANTIC_CELL_LON_FRONT)][0].v_x
        return front_objects_vel

    def _calc_velocity_profile(self, behavioral_state: SemanticBehavioralGridState, action_recipe: ActionRecipe,
                               T_d: float, front_objects_vel: np.array) -> Optional[VelocityProfile]:
        """
        Given action recipe and behavioral state, calculate the longitudinal velocity profile for that action.
        :param behavioral_state:
        :param action_recipe: the input action
        :param T_d: the action's lateral time (as a lower bound for the vel_profile total time)
        :param front_objects_vel: array of the front objects' velocities from the occupancy grid
        :return: longitudinal velocity profile or None if the action is infeasible by given aggressiveness level
        """
        ego = behavioral_state.ego_state
        ego_fpoint = np.array([ego.road_localization.road_lon, ego.road_localization.intra_road_lat])
        target_acc = cars_size_margin = obj_lon = None

        if action_recipe.action_type == ActionType.FOLLOW_VEHICLE or action_recipe.action_type == ActionType.TAKE_OVER_VEHICLE:
            target_obj = behavioral_state.road_occupancy_grid[
                (action_recipe.relative_lane.value, action_recipe.relative_lon.value)][0]
            target_vel = target_obj.v_x
            target_acc = target_obj.acceleration_lon
            obj_lon = target_obj.road_localization.road_lon
            cars_size_margin = 0.5 * (ego.size.length + target_obj.size.length)
        else:  # static action (FOLLOW_LANE)
            target_vel = action_recipe.velocity

            # TODO: the following condition should be implemented as a filter
            # skip static action if its velocity is greater than velocity of the front object on the same lane,
            # since a dynamic action should treat this lane (value function after such static action is unclear)
            if target_vel > front_objects_vel[action_recipe.relative_lane.value - SEMANTIC_CELL_LAT_RIGHT]:
                return None

        return VelocityProfile.calc_velocity_profile(
            action_recipe.action_type, ego_fpoint[FP_SX], ego.v_x, obj_lon, target_vel, target_acc,
            action_recipe.aggressiveness, cars_size_margin, min_time=T_d)

    def _calc_largest_safe_time(self, behavioral_state: SemanticBehavioralGridState, action_lat_cell: int,
                                vel_profile: VelocityProfile, ego_half_size: float, T_d: float) -> float:
        """
        For a lane change action, given ego velocity profile and behavioral_state, get two cars that may
        require faster lateral movement (the front overtaken car and the back interfered car) and calculate the last
        time, for which the safety holds w.r.t. these two cars.
        :param behavioral_state: semantic actions grid behavioral state
        :param action_lat_cell: either right, same or left
        :param vel_profile: the velocity profile of ego
        :param ego_half_size: half ego length
        :param T_d: time for comfortable lane change
        :return: the latest time, when ego is still safe
        """
        ego_lon = behavioral_state.ego_state.road_localization.road_lon

        # check safety w.r.t. the front object on the target lane (if exists)
        if (action_lat_cell.value, SEMANTIC_CELL_LON_FRONT) in behavioral_state.road_occupancy_grid:
            forward_obj = behavioral_state.road_occupancy_grid[(action_lat_cell.value, SEMANTIC_CELL_LON_FRONT)]
            forward_safe_time = ProfileSafety.calc_last_safe_time(
                ego_lon, ego_half_size, vel_profile, forward_obj[0], np.inf)
            if forward_safe_time < vel_profile.total_time():
                return 0
        if action_lat_cell.value == SEMANTIC_CELL_LAT_SAME:  # continue on the same lane
            return np.inf  # don't check safety on other lanes

        # TODO: move it to a filter
        # check whether there is a car in the neighbor cell (same longitude)
        if (action_lat_cell.value, SEMANTIC_CELL_LON_SAME) in behavioral_state.road_occupancy_grid:
            return 0

        safe_time = np.inf
        # check safety w.r.t. the front object on the original lane (if exists)
        if (SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LON_FRONT) in behavioral_state.road_occupancy_grid:
            front_obj = behavioral_state.road_occupancy_grid[(SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LON_FRONT)]
            front_safe_time = ProfileSafety.calc_last_safe_time(
                ego_lon, ego_half_size, vel_profile, front_obj[0], T_d)
            safe_time = min(safe_time, front_safe_time)

        # check safety w.r.t. the back object on the original lane (if exists)
        if (action_lat_cell.value, SEMANTIC_CELL_LON_REAR) in behavioral_state.road_occupancy_grid:
            back_obj = behavioral_state.road_occupancy_grid[(action_lat_cell.value, SEMANTIC_CELL_LON_REAR)]
            back_safe_time = ProfileSafety.calc_last_safe_time(
                ego_lon, ego_half_size, vel_profile, back_obj[0], T_d)
            safe_time = min(safe_time, back_safe_time)

        # print('front_time=%f back_time=%f forward_time=%f safe_time=%f' % \
        # (front_safe_time, back_safe_time, forward_safe_time, safe_time))
        return safe_time

    def _calc_action_cost(self, vel_profile: VelocityProfile, target_lane: int, T_d: float, T_d_max: float,
                          T_d_full: float, aggressiveness: AggressivenessLevel) -> [float, np.array]:
        """
        Calculate the cost of the action
        :param vel_profile: longitudinal velocity profile
        :param target_lane: target lane index
        :param T_d: lateral time according to the given aggressiveness level
        :param T_d_max: largest lateral time limited by safety
        :param T_d_full: lateral time of full lane change by calm action
        :param aggressiveness: aggressiveness level of the action
        :return: the action's cost and the cost components array (for debugging)
        """
        vel_profile_time = vel_profile.total_time()
        efficiency_cost = comfort_cost = right_lane_cost = lane_deviation_cost = 0
        if vel_profile_time > 0:
            efficiency_cost = PlanEfficiencyMetric.calc_cost(vel_profile)
            comfort_cost = PlanComfortMetric.calc_cost(vel_profile, T_d, T_d_max, aggressiveness)
            right_lane_cost = PlanRightLaneMetric.calc_cost(vel_profile_time, target_lane)
            lane_deviation_cost = PlanLaneDeviationMetric.calc_cost(T_d)

        value_approximator = ValueApproximator(self.logger)
        value_function = value_approximator.evaluate_state(
            BP_METRICS_TIME_HORIZON - vel_profile_time, vel_profile.v_tar, target_lane, T_d_full, T_d_full)

        return efficiency_cost + comfort_cost + right_lane_cost + lane_deviation_cost + value_function, \
               np.array([efficiency_cost, comfort_cost, right_lane_cost, lane_deviation_cost, value_function])
