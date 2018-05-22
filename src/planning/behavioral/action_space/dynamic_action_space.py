from logging import Logger
from typing import Optional, List, Type

import numpy as np
from sklearn.utils.extmath import cartesian

from decision_making.src.global_constants import BP_ACTION_T_LIMITS, SAFE_DIST_TIME_DELAY, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe, \
    ActionType, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering import recipe_filter_bank
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import LIMIT_MAX, FS_SV, FS_SX, LIMIT_MIN, FS_SA, FS_DA, FS_DV, FS_DX
from decision_making.src.planning.utils.map_utils import MapUtils
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.predictor import Predictor
from mapping.src.service.map_service import MapService


class DynamicActionSpace(ActionSpace):
    def __init__(self, logger: Logger, predictor: Predictor, filtering: RecipeFiltering):
        super().__init__(logger,
                         recipes=[DynamicActionRecipe.from_args_list(comb)
                                  for comb in cartesian([RelativeLane,
                                                         RelativeLongitudinalPosition,
                                                         [ActionType.FOLLOW_VEHICLE, ActionType.OVERTAKE_VEHICLE],
                                                         AggressivenessLevel])],
                         recipe_filtering=filtering)
        self.predictor = predictor

    @property
    def recipe_classes(self) -> List[Type]:
        return [DynamicActionRecipe]

    def specify_goals(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) -> \
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

        targets = [behavioral_state.road_occupancy_grid[(action_recipe.relative_lane, action_recipe.relative_lon)][0]
                   for action_recipe in action_recipes]
        target_length = np.array([target.dynamic_object.size.length for target in targets])
        target_fstate = np.array([target.fstate for target in targets])

        # get desired terminal velocity
        v_T = target_fstate[:, FS_SV]

        # latitudinal difference to target
        init_latitudinal_difference = desired_center_lane_latitude - ego_init_fstate[FS_DX]

        # T_d <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_d = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 1], ds=init_latitudinal_difference,
            a_0=ego_init_fstate[FS_DA], v_0=ego_init_fstate[FS_DV], v_T=0, T_m=SAFE_DIST_TIME_DELAY)
        roots_d = Math.find_real_roots_in_limits(cost_coeffs_d, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_d = np.fmin.reduce(roots_d, axis=-1)

        # longitudinal difference between object and ego at t=0 (positive if obj in front of ego)
        init_longitudinal_difference = target_fstate[:, FS_SX] - ego_init_fstate[FS_SX]
        # margin_sign is -1 for FOLLOW_VEHICLE (behind target) and +1 for OVER_TAKE_VEHICLE (in front of target)
        margin_sign = np.array([action_recipe.action_type.value * 2 - 5 for action_recipe in action_recipes])

        ds = init_longitudinal_difference + margin_sign * (
            LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + ego.size.length / 2 + target_length / 2)

        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 0], ds=ds,
            a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV], v_T=v_T, T_m=SAFE_DIST_TIME_DELAY)
        roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_s = np.fmin.reduce(roots_s, axis=-1)

        # voids (setting <np.nan>) all non-Calm actions with T_s < (minimal allowed T_s)
        # this still leaves some values of T_s which are smaller than (minimal allowed T_s) and will be replaced later
        # when setting T
        T_s[(T_s < BP_ACTION_T_LIMITS[LIMIT_MIN]) & (aggressiveness > AggressivenessLevel.CALM.value)] = np.nan

        # if both T_d[i] and T_s[i] are defined for i, then take maximum. otherwise leave it nan.
        T = np.maximum(np.maximum(T_d, T_s), BP_ACTION_T_LIMITS[LIMIT_MIN])

        # Calculate resulting distance from sampling the state at time T from the Quartic polynomial solution
        distance_s = QuinticPoly1D.distance_profile_function(a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV],
                                                             v_T=v_T, T=T, ds=ds, T_m=SAFE_DIST_TIME_DELAY)(T)
        target_s = distance_s + ego_init_fstate[FS_SX]

        action_specs = [ActionSpec(t, v_T[i], target_s[i], desired_center_lane_latitude[i])
                        if ~np.isnan(t) else None
                        for i, t in enumerate(T)]

        return action_specs
