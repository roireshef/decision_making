from typing import Optional, List

import numpy as np
from sklearn.utils.extmath import cartesian

from decision_making.src.global_constants import BP_ACTION_T_LIMITS, EPS, BP_JERK_S_JERK_D_TIME_WEIGHTS, \
    BP_ACTION_T_RES, LON_ACC_LIMITS, LAT_ACC_LIMITS
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.constants import VELOCITY_STEP, MAX_VELOCITY, MIN_VELOCITY
from decision_making.src.planning.behavioral.data_objects import ActionSpec, StaticActionRecipe
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering import recipe_filter_bank
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import LIMIT_MAX, LIMIT_MIN, FS_SV, FS_SA, FS_DX, FS_DA, FS_DV, FS_SX
from decision_making.src.planning.utils.map_utils import MapUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D
from mapping.src.service.map_service import MapService


class StaticActionSpace(ActionSpace):
    def __init__(self, logger):
        self._velocity_grid = np.arange(MIN_VELOCITY, MAX_VELOCITY + np.finfo(np.float16).eps, VELOCITY_STEP)
        super().__init__(logger,
                         recipes=[StaticActionRecipe.from_args_list(comb)
                                  for comb in cartesian([RelativeLane, self._velocity_grid, AggressivenessLevel])],
                         recipe_filtering=RecipeFiltering(recipe_filter_bank.static_filters))

    def specify_goals(self, action_recipes: List[StaticActionRecipe], behavioral_state: BehavioralGridState) -> \
            List[Optional[ActionSpec]]:
        ego = behavioral_state.ego_state
        road_frenet = MapUtils.get_road_rhs_frenet(ego)

        # project ego vehicle onto the road
        ego_init_fstate = MapUtils.get_ego_road_localization(ego, road_frenet)

        # get the relevant desired center lane latitude (from road's RHS)
        road_id = ego.road_localization.road_id
        road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(road_id)
        relative_lane = np.array([action_recipe.relative_lane.value for action_recipe in action_recipes])
        desired_lane = ego.road_localization.lane_num + relative_lane
        desired_center_lane_latitude = road_lane_latitudes[desired_lane]

        # get relevant aggressiveness weights for all actions
        aggressiveness = np.array([action_recipe.aggressiveness.value for action_recipe in action_recipes])
        weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggressiveness]

        # get desired terminal velocity
        v_T = np.array([action_recipe.velocity for action_recipe in action_recipes])

        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuarticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 0], a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV], v_T=v_T)
        roots_s = ActionSpace.find_roots(cost_coeffs_s, LON_ACC_LIMITS)
        T_s = np.fmin.reduce(roots_s, axis=-1)

        # latitudinal difference to target
        latitudinal_difference = desired_center_lane_latitude - ego_init_fstate[FS_DX]

        # T_d <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_d = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 1], a_0=ego_init_fstate[FS_DA], v_0=ego_init_fstate[FS_DV], v_T=0,
            ds_0=latitudinal_difference, T_m=0)
        roots_d = ActionSpace.find_roots(cost_coeffs_d, LAT_ACC_LIMITS)
        T_d = np.fmin.reduce(roots_d, axis=-1)

        # if both T_d[i] and T_s[i] are defined for i, then take maximum. otherwise leave it nan.
        T = np.maximum(T_d, T_s)

        # Calculate resulting distance from sampling the state at time T from the Quartic polynomial solution
        distance_s = QuarticPoly1D.distance_profile_function(a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV], v_T=v_T, T=T)(T)
        target_s = distance_s + ego_init_fstate[FS_SX]

        action_specs = [ActionSpec(t, v_T[i], target_s[i], desired_center_lane_latitude[i])
                        if ~np.isnan(t) else None
                        for i, t in enumerate(T)]

        return action_specs

    def specify_goal(self, action_recipe: StaticActionRecipe, behavioral_state: BehavioralGridState) -> \
            Optional[ActionSpec]:

        ego = behavioral_state.ego_state
        road_frenet = MapUtils.get_road_rhs_frenet(ego)

        # project ego vehicle onto the road
        ego_init_fstate = MapUtils.get_ego_road_localization(ego, road_frenet)

        road_id = ego.road_localization.road_id
        road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(road_id)
        desired_lane = ego.road_localization.lane_num + action_recipe.relative_lane.value
        desired_center_lane_latitude = road_lane_latitudes[desired_lane]

        T_s_vals = np.arange(BP_ACTION_T_LIMITS[LIMIT_MIN], BP_ACTION_T_LIMITS[LIMIT_MAX] + EPS, BP_ACTION_T_RES)

        w_Js, w_Jd, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[action_recipe.aggressiveness.value]
        v_0, a_0 = ego_init_fstate[FS_SV], ego_init_fstate[FS_SA]
        v_T = action_recipe.velocity

        lon_time_cost_func_der = QuarticPoly1D.time_cost_function_derivative(w_T, w_Js, a_0, v_0, v_T)

        T_s = ActionSpace.find_roots(lon_time_cost_func_der, T_s_vals)
        # If roots were found out of the desired region, this action won't be specified
        if len(T_s) == 0:
            return None

        # TODO: Do the same as above for lateral movement
        # TODO: check if lateral trajectory is feasible(?)

        latitudinal_difference = desired_center_lane_latitude - ego_init_fstate[FS_DX]
        lat_time_cost_func_der = QuinticPoly1D.time_cost_function_derivative(w_T, w_Jd,
                                                                             ego_init_fstate[FS_DA],
                                                                             ego_init_fstate[FS_DV], 0,
                                                                             latitudinal_difference,
                                                                             T_m=0)

        T_d_vals = np.arange(0.1, BP_ACTION_T_LIMITS[LIMIT_MAX] + EPS, BP_ACTION_T_RES)
        T_d = ActionSpace.find_roots(lat_time_cost_func_der, T_d_vals)
        # If roots were found out of the desired region, this action won't be specified
        if len(T_d) == 0:
            return None

        # This stems from the assumption we've made about independency between d and s
        planning_time = float(max(T_s[0], T_d[0]))

        distance_func = QuarticPoly1D.distance_profile_function(a_0, v_0, v_T, planning_time)
        target_s = distance_func(planning_time) + ego_init_fstate[FS_SX]

        return ActionSpec(t=planning_time, v=v_T,
                          s=target_s,
                          d=desired_center_lane_latitude)


