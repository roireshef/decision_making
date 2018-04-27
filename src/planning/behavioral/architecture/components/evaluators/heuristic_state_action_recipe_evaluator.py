from logging import Logger
from typing import List

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
from decision_making.src.planning.behavioral.architecture.data_objects import ActionRecipe, ActionType
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
        ego_fpoint = np.array([ego.road_localization.road_lon, ego.road_localization.intra_road_lat])
        ego_lane = ego.road_localization.lane_num
        ego_lat_vel = ego.v_x * np.sin(ego.road_localization.intra_road_yaw)
        lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width

        value_approximator = ValueApproximator(self.logger)
        comfort_lane_change_time = VelocityProfile.calc_lateral_time(0, lane_width, aggressiveness_level=0)

        action_costs = np.full(len(action_recipes), np.inf)

        # get front dynamic objects from the occupancy grid
        front_objects_vel = np.array([np.inf, np.inf, np.inf])
        for lat in [SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LAT_LEFT]:
            if (lat, SEMANTIC_CELL_LON_FRONT) in behavioral_state.road_occupancy_grid:
                front_objects_vel[lat - SEMANTIC_CELL_LAT_RIGHT] = \
                    behavioral_state.road_occupancy_grid[(lat, SEMANTIC_CELL_LON_FRONT)][0].v_x

        print('time=%f ego_v=%f ego_lat=%f' % (ego.timestamp_in_sec, ego.v_x, ego_fpoint[FP_DX]))

        for i, action_recipe in enumerate(action_recipes):

            if not action_recipes_mask[i]:
                continue

            target_acc = cars_size_margin = obj_lon = None
            target_lane = ego_lane + action_recipe.relative_lane.value
            target_lat = (target_lane + 0.5) * lane_width

            # create velocity profile, whose extent is at least as the lateral time
            lat_time = VelocityProfile.calc_lateral_time(ego_lat_vel, target_lat - ego_fpoint[FP_DX],
                                                         action_recipe.aggressiveness.value)

            if action_recipe.action_type == ActionType.FOLLOW_VEHICLE or action_recipe.action_type == ActionType.TAKE_OVER_VEHICLE:
                target_obj = behavioral_state.road_occupancy_grid[
                    (action_recipe.relative_lane.value, action_recipe.relative_lon.value)][0]
                target_vel = target_obj.v_x
                target_acc = target_obj.acceleration_lon
                obj_lon = target_obj.road_localization.road_lon
                cars_size_margin = 0.5 * (ego.size.length + target_obj.size.length)
                min_profile_time = lat_time
            else:  # static action (FOLLOW_LANE)
                target_vel = action_recipe.velocity
                min_profile_time = BP_METRICS_TIME_HORIZON

                # TODO: the following condition should be implemented as a filter
                # skip static action if its velocity is greater than velocity of the front object on the same lane,
                # since a dynamic action should treat this lane (value function after such static action is unclear)
                if target_vel > front_objects_vel[action_recipe.relative_lane.value - SEMANTIC_CELL_LAT_RIGHT]:
                    continue

            vel_profile = VelocityProfile.calc_velocity_profile(
                action_recipe.action_type, ego_fpoint[FP_SX], ego.v_x, obj_lon, target_vel, target_acc,
                action_recipe.aggressiveness.value, cars_size_margin, min_time=min_profile_time)

            if vel_profile is None:  # infeasible action
                continue

            vel_profile_time = vel_profile.total_time()
            # calculate the latest safe time
            largest_safe_time = HeuristicStateActionRecipeEvaluator._calc_largest_safe_time(
                behavioral_state, action_recipe.relative_lane, ego_fpoint, vel_profile, ego.size.length / 2, lat_time,
                min_profile_time)
            # if largest_safe_time <= comfort_lat_time/3:
            #     continue

            efficiency_cost = PlanEfficiencyMetric.calc_cost(vel_profile)
            comfort_cost = PlanComfortMetric.calc_cost(vel_profile, lat_time, largest_safe_time)
            right_lane_cost = PlanRightLaneMetric.calc_cost(vel_profile_time, target_lane)
            lane_deviation_cost = PlanLaneDeviationMetric.calc_cost(lat_time)

            value_function = value_approximator.evaluate_state(BP_METRICS_TIME_HORIZON - vel_profile_time, target_vel,
                                                               target_lane, comfort_lane_change_time)

            action_costs[i] = efficiency_cost + right_lane_cost + comfort_cost + lane_deviation_cost + value_function


            dist = np.inf
            if obj_lon is not None:
                dist = obj_lon - ego_fpoint[FP_SX]
            print('action %d type %d lane %d: dist=%f target_vel=%.2f eff %.3f comf %.3f right %.2f dev %.2f time=%.2f safe_time=%f value %.2f: tot %.2f' %
                  (i, action_recipe.action_type.value, action_recipe.relative_lane.value, dist, target_vel,
                   efficiency_cost, comfort_cost, right_lane_cost, lane_deviation_cost, vel_profile_time,
                   largest_safe_time, value_function, action_costs[i]))

        #end = time.time()
        #print('tot_time = %f: init=%f lat_time=%f vel_prof=%f safe_time=%f metrics=%f' %
        #      (end - start, init_time, lat_time_time, vel_prof_time, safe_time_time, metrics_time))

        best_action = int(np.argmin(action_costs))
        print('Best action %d; lane %d\n' % (best_action, ego_lane + action_recipes[best_action].relative_lane.value))
        return action_costs

    @staticmethod
    def _calc_largest_safe_time(behavioral_state: SemanticBehavioralGridState, action_lat_cell: int,
                                ego_fpoint: np.array, vel_profile: VelocityProfile, ego_half_size: float,
                                lat_time: float, min_forward_safe_time: float) -> float:
        """
        For a lane change action, given ego velocity profile and behavioral_state, get two cars that may
        require faster lateral movement (the front overtaken car and the back interfered car) and calculate the last
        time, for which the safety holds w.r.t. these two cars.
        :param behavioral_state: semantic actions grid behavioral state
        :param action_lat_cell: either right, same or left
        :param ego_fpoint: ego in Frenet coordinates
        :param vel_profile: the velocity profile of ego
        :param ego_half_size: half ego length
        :param lat_time: time for comfortable lane change
        :return: the latest time, when ego is still safe
        """
        # check safety w.r.t. the front object on the target lane (if exists)
        if (action_lat_cell.value, SEMANTIC_CELL_LON_FRONT) in behavioral_state.road_occupancy_grid:
            forward_obj = behavioral_state.road_occupancy_grid[(action_lat_cell.value, SEMANTIC_CELL_LON_FRONT)]
            forward_safe_time = ProfileSafety.calc_last_safe_time(
                ego_fpoint, ego_half_size, vel_profile, forward_obj[0], np.inf)
            if forward_safe_time < max(min_forward_safe_time, vel_profile.total_time()):
                return 0
        if action_lat_cell.value == SEMANTIC_CELL_LAT_SAME:  # continue on the same lane
            return np.inf  # don't check safety on other lanes

        #TODO: move it to a filter
        # check whether there is a car in the neighbor cell (same longitude)
        if (action_lat_cell.value, SEMANTIC_CELL_LON_SAME) in behavioral_state.road_occupancy_grid:
            return 0

        safe_time = np.inf
        # check safety w.r.t. the front object on the original lane (if exists)
        if (SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LON_FRONT) in behavioral_state.road_occupancy_grid:
            front_obj = behavioral_state.road_occupancy_grid[(SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LON_FRONT)]
            front_safe_time = ProfileSafety.calc_last_safe_time(
                ego_fpoint, ego_half_size, vel_profile, front_obj[0], lat_time)
            safe_time = min(safe_time, front_safe_time)

        # check safety w.r.t. the back object on the original lane (if exists)
        if (action_lat_cell.value, SEMANTIC_CELL_LON_REAR) in behavioral_state.road_occupancy_grid:
            back_obj = behavioral_state.road_occupancy_grid[(action_lat_cell.value, SEMANTIC_CELL_LON_REAR)]
            back_safe_time = ProfileSafety.calc_last_safe_time(
                ego_fpoint, ego_half_size, vel_profile, back_obj[0], lat_time)
            safe_time = min(safe_time, back_safe_time)

        # print('front_time=%f back_time=%f forward_time=%f safe_time=%f' % \
        # (front_safe_time, back_safe_time, forward_safe_time, safe_time))
        return safe_time
