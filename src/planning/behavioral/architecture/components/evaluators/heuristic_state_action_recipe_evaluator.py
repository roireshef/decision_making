from logging import Logger
from typing import List

import numpy as np

from decision_making.src.global_constants import BP_ACTION_T_LIMITS, \
    SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_SAME, \
    SEMANTIC_CELL_LAT_SAME
from decision_making.src.global_constants import SEMANTIC_CELL_LON_REAR
from decision_making.src.planning.behavioral.architecture.components.evaluators.state_action_evaluator import \
    StateActionRecipeEvaluator
from decision_making.src.planning.behavioral.architecture.data_objects import ActionRecipe, ActionType
from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState
from decision_making.src.planning.performance_metrics.behavioral.cost_functions import PlanEfficiencyMetric, \
    PlanComfortMetric, ValueFunction, PlanRightLaneMetric, VelocityProfile
from decision_making.src.planning.performance_metrics.behavioral.velocity_profile import ProfileSafety
from decision_making.src.planning.types import FS_SX, FS_DX, FS_DV, FP_SX, FP_DX
from decision_making.src.planning.types import LIMIT_MAX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
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
        road_id = ego.road_localization.road_id
        road = MapService.get_instance().get_road(road_id)
        road_points = MapService.get_instance()._shift_road_points_to_latitude(road_id, 0.0)
        road_frenet = FrenetSerret2DFrame(road_points)  # TODO: it's heavy (10 ms), bring road_frenet from outside
        ego_cstate = np.array([ego.x, ego.y, ego.yaw, ego.v_x, ego.acceleration_lon, ego.curvature])
        ego_fstate = road_frenet.cstate_to_fstate(ego_cstate)
        ego_fpoint = np.array([ego_fstate[FS_SX], ego_fstate[FS_DX]])
        ego_lane = ego.road_localization.lane_num
        lane_width = road.lane_width

        time_horizon = BP_ACTION_T_LIMITS[LIMIT_MAX]

        action_costs = np.full(len(action_recipes), np.inf)

        for i, action_recipe in enumerate(action_recipes):

            if not action_recipes_mask[i]:
                continue

            target_acc = cars_size_margin = obj_lon = None

            if action_recipe.action_type == ActionType.FOLLOW_VEHICLE or action_recipe.action_type == ActionType.TAKE_OVER_VEHICLE:
                target_obj = behavioral_state.road_occupancy_grid[(action_recipe.relative_lane.value,
                                                                   action_recipe.relative_lon.value)][0]
                target_vel = target_obj.v_x
                target_acc = target_obj.acceleration_lon
                obj_lon = road_frenet.cpoint_to_fpoint(np.array([target_obj.x, target_obj.y]))[FP_SX]
                cars_size_margin = 0.5 * (ego.size.length + target_obj.size.length)
            else:
                target_vel = action_recipe.velocity

            target_lane = ego_lane + action_recipe.relative_lane.value
            target_lat = (target_lane + 0.5) * lane_width

            # create velocity profile, whose extent is at least as the lateral time
            comfort_lat_time = VelocityProfile.calc_lateral_time(ego_fstate[FS_DV], target_lat - ego_fpoint[FP_DX])
            vel_profile = VelocityProfile.calc_velocity_profile(action_recipe.action_type, ego_fpoint[FP_SX], ego.v_x,
                                                                obj_lon, target_vel, target_acc,
                                                                action_recipe.aggressiveness.value,
                                                                cars_size_margin, comfort_lat_time)

            if vel_profile is None:  # infeasible action
                continue

            vel_profile_time = vel_profile.total_time()

            # efficiency cost
            efficiency_cost = PlanEfficiencyMetric.calc_cost(ego.v_x, target_vel, vel_profile)

            # comfort cost
            largest_safe_time = HeuristicStateActionRecipeEvaluator._calc_largest_safe_time(
                behavioral_state, action_recipe.relative_lane, ego_fpoint, vel_profile, road_frenet,
                ego.size.length / 2, comfort_lat_time)

            comfort_cost = PlanComfortMetric.calc_cost(vel_profile, comfort_lat_time, largest_safe_time)

            right_lane_cost = PlanRightLaneMetric.calc_cost(vel_profile_time, target_lane)

            value_function = ValueFunction.calc_cost(time_horizon - vel_profile_time, target_vel, target_lane)

            action_costs[i] = efficiency_cost + right_lane_cost + comfort_cost + value_function

            # print('time %f; action %d: obj_vel=%s eff %s comf %s right %s value %f: tot %s' %
            #       (ego.timestamp_in_sec, action_recipe.relative_lane, target_vel, efficiency_cost, comfort_cost,
            #        right_lane_cost, value_function, action_costs[i]))

        # best_action = np.argmin(action_costs)[0]
        # print('Best action %d; lane %d\n' % (best_action, ego_lane + action_recipes[best_action].relative_lane))
        return action_costs

    @staticmethod
    def _calc_largest_safe_time(behavioral_state: SemanticBehavioralGridState, action_lat_cell: int,
                                ego_fpoint: np.array, vel_profile: VelocityProfile,
                                road_frenet: FrenetSerret2DFrame, ego_half_size: float,
                                comfort_lat_time: float) -> float:
        """
        For a lane change action, given ego velocity profile and behavioral_state, get two cars that may
        require faster lateral movement (the front overtaken car and the back interfered car) and calculate the last
        time, for which the safety holds w.r.t. these two cars.
        :param behavioral_state: semantic actions grid behavioral state
        :param action_lat_cell: either right, same or left
        :param ego_fpoint: ego in Frenet coordinates
        :param vel_profile: the velocity profile of ego
        :param road_frenet: Frenet frame
        :param ego_half_size: half ego length
        :param comfort_lat_time: time for comfortable lane change
        :return: the latest time, when ego is still safe
        """
        # check safety w.r.t. the front object on the target lane (if exists)
        if (action_lat_cell, SEMANTIC_CELL_LON_FRONT) in behavioral_state.road_occupancy_grid:
            forward_obj = behavioral_state.road_occupancy_grid[(action_lat_cell, SEMANTIC_CELL_LON_FRONT)]
            forward_safe_time = ProfileSafety.calc_last_safe_time(
                ego_fpoint, ego_half_size, vel_profile, forward_obj[0], road_frenet, np.inf)
            if forward_safe_time < vel_profile.total_time():
                return 0
        if action_lat_cell == SEMANTIC_CELL_LAT_SAME:  # continue on the same lane
            return np.inf  # don't check safety on other lanes

        safe_time = np.inf
        # check safety w.r.t. the front object on the original lane (if exists)
        if (SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LON_FRONT) in behavioral_state.road_occupancy_grid:
            front_obj = behavioral_state.road_occupancy_grid[(SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LON_FRONT)]
            front_safe_time = ProfileSafety.calc_last_safe_time(
                ego_fpoint, ego_half_size, vel_profile, front_obj[0], road_frenet, comfort_lat_time)
            safe_time = min(safe_time, front_safe_time)

        # check safety w.r.t. the back object on the original lane (if exists)
        if (action_lat_cell, SEMANTIC_CELL_LON_REAR) in behavioral_state.road_occupancy_grid:
            back_obj = behavioral_state.road_occupancy_grid[(action_lat_cell, SEMANTIC_CELL_LON_REAR)]
            back_safe_time = ProfileSafety.calc_last_safe_time(
                ego_fpoint, ego_half_size, vel_profile, back_obj[0], road_frenet, comfort_lat_time)
            safe_time = min(safe_time, back_safe_time)

        # print('front_time=%f back_time=%f forward_time=%f safe_time=%f' % \
        # (front_safe_time, back_safe_time, forward_safe_time, safe_time))
        return safe_time
