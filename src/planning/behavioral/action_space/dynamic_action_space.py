from logging import Logger
from typing import Optional

import numpy as np
from decision_making.src.planning.behavioral.filtering import recipe_filter_bank
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe, \
    ActionType, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from sklearn.utils.extmath import cartesian

from decision_making.src.global_constants import BP_ACTION_T_LIMITS, BP_ACTION_T_RES, SAFE_DIST_TIME_DELAY, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.semantic_actions_utils import SemanticActionsUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.planning.trajectory.werling_planner import SamplableWerlingTrajectory
from decision_making.src.planning.types import FP_SX, LIMIT_MAX, FS_SV, FS_SX, LIMIT_MIN, FS_SA, FS_DA, FS_DV, FS_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.optimal_control.quintic_poly_formulas import create_action_time_cost_func_deriv
from decision_making.src.prediction.predictor import Predictor
from mapping.src.service.map_service import MapService


class DynamicActionSpace(ActionSpace):

    def __init__(self, logger: Logger, predictor: Predictor):
        super().__init__(logger,
                         recipes=[DynamicActionRecipe.from_args_list(comb)
                                  for comb in cartesian([RelativeLane,
                                                         RelativeLongitudinalPosition,
                                                         [ActionType.FOLLOW_VEHICLE, ActionType.TAKE_OVER_VEHICLE],
                                                         AggressivenessLevel])],
                         recipe_filtering=RecipeFiltering(recipe_filter_bank.dynamic_filters))

        self.predictor = predictor

    def specify_goal_faster(self, action_recipe: DynamicActionRecipe,
                            behavioral_state: BehavioralGridState) -> Optional[ActionSpec]:
        """
        Given a state and a high level SemanticAction towards an object, generate a SemanticActionSpec.
        Internally, the reference route here is the RHS of the road, and the ActionSpec is specified with respect to it.
        :param action_recipe:
        :param behavioral_state: Frenet state of ego at initial point
        :return: semantic action specification
        """
        ego = behavioral_state.ego_state
        ego_init_cstate = np.array([ego.x, ego.y, ego.yaw, ego.v_x, ego.acceleration_lon, ego.curvature])
        road_id = ego.road_localization.road_id
        road_points = MapService.get_instance()._shift_road_points_to_latitude(road_id, 0.0)  # TODO: use nav_plan
        road_frenet = FrenetSerret2DFrame(road_points)
        ego_init_fstate = road_frenet.cstate_to_fstate(ego_init_cstate)

        recipe_cell = (action_recipe.relative_lane.value, action_recipe.relative_lon.value)
        target_obj = behavioral_state.road_occupancy_grid[recipe_cell][0]
        target_obj_fpoint = road_frenet.cpoint_to_fpoint(np.array([target_obj.x, target_obj.y]))
        _, _, _, road_curvature_at_obj_location, _ = road_frenet._taylor_interp(target_obj_fpoint[FP_SX])
        obj_init_fstate = road_frenet.cstate_to_fstate(np.array([
            target_obj.x, target_obj.y,
            target_obj.yaw,
            target_obj.total_speed,
            target_obj.acceleration_lon,
            road_curvature_at_obj_location  # We don't care about other agent's curvature, only the road's
        ]))

        # Extract relevant details from state on Reference-Object
        obj_on_road = target_obj.road_localization
        road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(road_id=obj_on_road.road_id)
        obj_center_lane_latitude = road_lane_latitudes[obj_on_road.lane_num]

        optimum_horizon_search_margin = 0.2
        T_vals = np.arange(BP_ACTION_T_LIMITS[LIMIT_MIN] - optimum_horizon_search_margin,
                           BP_ACTION_T_LIMITS[LIMIT_MAX] + optimum_horizon_search_margin + np.finfo(np.float16).eps,
                           0.001)

        w_Js, w_Jd, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[action_recipe.aggressiveness.value]
        v_0, a_0 = ego_init_fstate[FS_SV], ego_init_fstate[FS_SA]
        v_T = obj_init_fstate[FS_SV]

        obj_dist = BehavioralGridState.nonoverlapping_longitudinal_distance(ego, road_frenet, target_obj)
        lon_margin = SemanticActionsUtils.get_ego_lon_margin(ego.size) + target_obj.size.length / 2

        # Note: create_action_time_cost_func_deriv assumes constant-velocity moving objects
        if action_recipe.action_type == ActionType.FOLLOW_VEHICLE:
            lon_action_time_cost_func_deriv = create_action_time_cost_func_deriv(w_T, w_Js, a_0, v_0, v_T,
                                                                                 obj_dist - lon_margin,
                                                                                 T_m=SAFE_DIST_TIME_DELAY)
        elif action_recipe.action_type == ActionType.TAKE_OVER_VEHICLE:
            lon_action_time_cost_func_deriv = create_action_time_cost_func_deriv(w_T, w_Js, a_0, v_0, v_T,
                                                                                 obj_dist + lon_margin,
                                                                                 T_m=-SAFE_DIST_TIME_DELAY)
        else:
            raise NotImplemented("Action Type %s is not handled in DynamicActionSpace specification",
                                 action_recipe.action_type)

        der_lon = lon_action_time_cost_func_deriv(T_vals)
        ind_lon = np.argwhere(np.abs(der_lon) < 0.01)
        T_s = T_vals[ind_lon[0]]

        # TODO: Do the same as above for lateral movement and check if trajectory is feasible(?)

        lat_action_time_cost_func_deriv = create_action_time_cost_func_deriv(w_T, w_Jd, ego_init_fstate[FS_DA],
                                                                             ego_init_fstate[FS_DV], 0,
                                                                             ego_init_fstate[
                                                                                 FS_DX] - obj_center_lane_latitude,
                                                                             T_m=SAFE_DIST_TIME_DELAY)

        der_lat = lat_action_time_cost_func_deriv(T_vals)
        ind_lat = np.argwhere(np.abs(der_lat) < 0.01)
        T_d = T_vals[ind_lat[0]]

        # Note: We create the samplable trajectory as a reference trajectory of the current action.from
        # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.
        samplable_trajectory = SamplableWerlingTrajectory(timestamp_in_sec=ego.timestamp_in_sec,
                                                          T_s=T_vals[optimum_time_idx],
                                                          T_d=T_vals[optimum_time_idx],
                                                          frenet_frame=road_frenet,
                                                          poly_s_coefs=poly_coefs_s[optimum_time_idx],
                                                          poly_d_coefs=poly_coefs_d[optimum_time_idx])

        return ActionSpec(t=T_vals[optimum_time_idx], v=obj_svT[optimum_time_idx],
                          s=constraints_s[optimum_time_idx, 3],
                          d=constraints_d[optimum_time_idx, 3],
                          samplable_trajectory=samplable_trajectory)

    def specify_goal(self, action_recipe: DynamicActionRecipe,
                     behavioral_state: BehavioralGridState) -> Optional[ActionSpec]:
        """
        Given a state and a high level SemanticAction towards an object, generate a SemanticActionSpec.
        Internally, the reference route here is the RHS of the road, and the ActionSpec is specified with respect to it.
        :param action_recipe:
        :param behavioral_state: Frenet state of ego at initial point
        :return: semantic action specification
        """
        ego = behavioral_state.ego_state
        ego_init_cstate = np.array([ego.x, ego.y, ego.yaw, ego.v_x, ego.acceleration_lon, ego.curvature])
        road_id = ego.road_localization.road_id
        road_points = MapService.get_instance()._shift_road_points_to_latitude(road_id, 0.0)  # TODO: use nav_plan
        road_frenet = FrenetSerret2DFrame(road_points)
        ego_init_fstate = road_frenet.cstate_to_fstate(ego_init_cstate)

        target_obj = behavioral_state.road_occupancy_grid[(action_recipe.relative_lane.value,
                                                           action_recipe.relative_lon.value)][0]
        target_obj_fpoint = road_frenet.cpoint_to_fpoint(np.array([target_obj.x, target_obj.y]))
        _, _, _, road_curvature_at_obj_location, _ = road_frenet._taylor_interp(target_obj_fpoint[FP_SX])
        obj_init_fstate = road_frenet.cstate_to_fstate(np.array([
            target_obj.x, target_obj.y,
            target_obj.yaw,
            target_obj.total_speed,
            target_obj.acceleration_lon,
            road_curvature_at_obj_location  # We don't care about other agent's curvature, only the road's
        ]))

        # Extract relevant details from state on Reference-Object
        obj_on_road = target_obj.road_localization
        road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(road_id=obj_on_road.road_id)
        obj_center_lane_latitude = road_lane_latitudes[obj_on_road.lane_num]

        T_vals = np.arange(BP_ACTION_T_LIMITS[LIMIT_MIN], BP_ACTION_T_LIMITS[LIMIT_MAX] + np.finfo(np.float16).eps,
                           BP_ACTION_T_RES)

        # TODO: should be swapped with current implementation of Predictor.predict_object_on_road
        obj_saT = 0  # obj_init_fstate[FS_SA]
        obj_svT = obj_init_fstate[FS_SV] + obj_saT * T_vals
        obj_sxT = obj_init_fstate[FS_SX] + obj_svT * T_vals + obj_saT * T_vals ** 2 / 2

        safe_lon_dist = obj_svT * SAFE_DIST_TIME_DELAY
        lon_margin = SemanticActionsUtils.get_ego_lon_margin(ego.size) + target_obj.size.length / 2

        if action_recipe.action_type == ActionType.FOLLOW_VEHICLE:
            desired_lon = obj_sxT - safe_lon_dist - lon_margin
        elif action_recipe.action_type == ActionType.TAKE_OVER_VEHICLE:
            desired_lon = obj_sxT + safe_lon_dist + lon_margin
        else:
            raise NotImplemented("Action Type %s is not handled in DynamicActionSpace specification",
                                 action_recipe.action_type)

        constraints_s = ActionSpace.define_lon_constraints(len(T_vals), ego_init_fstate, obj_saT, obj_svT, desired_lon)
        constraints_d = ActionSpace.define_lat_constraints(len(T_vals), ego_init_fstate, obj_center_lane_latitude)
        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T_vals))

        # solve for s(t) and d(t)
        poly_coefs_s = QuinticPoly1D.zip_solve(A_inv, constraints_s)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv, constraints_d)

        optimum_time_idx, optimum_time_satisfies_constraints = ActionSpace.find_optimum_planning_time(T_vals,
                                                                                                      poly_coefs_s,
                                                                                                      QuinticPoly1D,
                                                                                                      poly_coefs_d,
                                                                                                      QuinticPoly1D,
                                                                                                      action_recipe.aggressiveness)

        if not optimum_time_satisfies_constraints:
            # self.logger.debug("Can\'t specify Recipe %s given ego state %s ", str(action_recipe), str(ego))
            return None

        # Note: We create the samplable trajectory as a reference trajectory of the current action.from
        # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.
        samplable_trajectory = SamplableWerlingTrajectory(timestamp_in_sec=ego.timestamp_in_sec,
                                                          T_s=T_vals[optimum_time_idx],
                                                          T_d=T_vals[optimum_time_idx],
                                                          frenet_frame=road_frenet,
                                                          poly_s_coefs=poly_coefs_s[optimum_time_idx],
                                                          poly_d_coefs=poly_coefs_d[optimum_time_idx])

        return ActionSpec(t=T_vals[optimum_time_idx], v=obj_svT[optimum_time_idx],
                          s=constraints_s[optimum_time_idx, 3],
                          d=constraints_d[optimum_time_idx, 3],
                          samplable_trajectory=samplable_trajectory)
