from logging import Logger
from typing import Optional, List, Type

import numpy as np
from sklearn.utils.extmath import cartesian

import rte.python.profiler as prof
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, SPECIFICATION_MARGIN_TIME_DELAY, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, BEHAVIORAL_PLANNING_TIME_RESOLUTION, \
    LON_ACC_LIMITS
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe, \
    ActionType, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import LIMIT_MAX, FS_SV, FS_SX, LIMIT_MIN, FS_SA, FS_DA, FS_DV, FS_DX
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor


class DynamicActionSpace(ActionSpace):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, filtering: RecipeFiltering):
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
        """a list of Recipe classes this action space can handle with"""
        return [DynamicActionRecipe]

    @prof.ProfileFunction()
    def specify_goals(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) -> \
            List[Optional[ActionSpec]]:
        """
        This method's purpose is to specify the enumerated actions (recipes) that the agent can take.
        Each semantic action (ActionRecipe) is translated into a terminal state specification (ActionSpec).
        :param action_recipes: a list of enumerated semantic actions [ActionRecipe].
        :param behavioral_state: a Frenet state of ego at initial point
        :return: semantic action specification [ActionSpec] or [None] if recipe can't be specified.
        """
        # pick ego initial fstates projected on all target frenet_frames
        projected_ego_fstates = np.array([behavioral_state.projected_ego_fstates[recipe.relative_lane] for recipe in action_recipes])

        # collect targets' lengths, lane_ids and fstates
        targets = [behavioral_state.road_occupancy_grid[(action_recipe.relative_lane, action_recipe.relative_lon)][0]
                   for action_recipe in action_recipes]
        target_length = np.array([target.dynamic_object.size.length for target in targets])
        target_map_states = [target.dynamic_object.map_state for target in targets]
        # get desired terminal velocity
        v_T = np.array([map_state.lane_fstate[FS_SV] for map_state in target_map_states])
        v_0 = np.full(shape=v_T.shape, fill_value=behavioral_state.ego_state.map_state.lane_fstate[FS_SV])
        a_0 = np.full(shape=v_T.shape, fill_value=behavioral_state.ego_state.map_state.lane_fstate[FS_SA])
        zeros = np.zeros(shape=v_T.shape)

        # get relevant aggressiveness weights for all actions
        aggressiveness = np.array([action_recipe.aggressiveness.value for action_recipe in action_recipes])
        weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggressiveness]

        # calculate initial longitudinal differences between all target objects and ego along target lanes
        longitudinal_differences = behavioral_state.calculate_longitudinal_differences(target_map_states)
        assert not np.isinf(longitudinal_differences).any()

        # margin_sign is -1 for FOLLOW_VEHICLE (behind target) and +1 for OVER_TAKE_VEHICLE (in front of target)
        margin_sign = np.array([-1 if action_recipe.action_type == ActionType.FOLLOW_VEHICLE else +1
                                for action_recipe in action_recipes])

        ds = longitudinal_differences + margin_sign * (
            LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + behavioral_state.ego_state.size.length / 2 + target_length / 2)

        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 0], dx=ds,
            a_0=projected_ego_fstates[:, FS_SA], v_0=projected_ego_fstates[:, FS_SV], v_T=v_T, T_m=SPECIFICATION_MARGIN_TIME_DELAY)
        roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_s = np.fmin.reduce(roots_s, axis=-1)

        # Agent is in tracking mode, meaning the required velocity change is negligible and action time is actually
        # zero. This degenerate action is valid but can't be solved analytically thus we probably got nan for T_s
        # although it should be zero.
        T_s[np.logical_and(np.isclose(v_T, v_0, atol=1e-3, rtol=0), np.isclose(a_0, zeros, atol=1e-3, rtol=0))] = 0

        # T_d <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_d = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 1], dx=-projected_ego_fstates[:, FS_DX],
            a_0=projected_ego_fstates[:, FS_DA], v_0=projected_ego_fstates[:, FS_DV], v_T=0, T_m=SPECIFICATION_MARGIN_TIME_DELAY)
        roots_d = Math.find_real_roots_in_limits(cost_coeffs_d, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_d = np.fmin.reduce(roots_d, axis=-1)

        # if both T_d[i] and T_s[i] are defined for i, then take maximum. otherwise leave it nan.
        T = np.maximum(T_d, T_s)

        # Calculate resulting distance from sampling the state at time T from the Quintic polynomial solution.
        # distance_s also takes into account the safe distance that depends on target vehicle velocity that we want
        # to keep from the target vehicle.
        distance_s = QuinticPoly1D.distance_profile_function(a_0=projected_ego_fstates[:, FS_SA],
                                                             v_0=projected_ego_fstates[:, FS_SV],
                                                             v_T=v_T, T=T, dx=ds,
                                                             T_m=SPECIFICATION_MARGIN_TIME_DELAY)(T)
        # Absolute longitudinal position of target
        target_s = distance_s + projected_ego_fstates[:, FS_SX]

        # lane center has latitude = 0, i.e. spec.d = 0
        action_specs = [ActionSpec(t, v_T[i], target_s[i], 0, action_recipes[i].relative_lane)
                        if ~np.isnan(t) else None
                        for i, t in enumerate(T)]

        return action_specs

    def specify_aggressive_braking(self, action_recipes: List[DynamicActionRecipe], recipes_mask: List[bool],
                                   behavioral_state: BehavioralGridState) -> (int, ActionSpec):

        rel_lane, rel_lon = RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT
        if (rel_lane, rel_lon) not in behavioral_state.road_occupancy_grid:
            return None, None
        aggressive_same_lane_follow_vehicle_idx = [i for i, recipe in enumerate(action_recipes)
                                                   if recipe.action_type == ActionType.FOLLOW_VEHICLE and
                                                   recipe.relative_lane == rel_lane and recipe.relative_lon == rel_lon and
                                                   recipe.aggressiveness == AggressivenessLevel.AGGRESSIVE]
        # if the recipe passed the filter, return None
        if len(aggressive_same_lane_follow_vehicle_idx) == 0 or recipes_mask[aggressive_same_lane_follow_vehicle_idx[0]]:
            return None, None
        object = behavioral_state.road_occupancy_grid[(rel_lane, rel_lon)][0].dynamic_object
        ego_fstate_s = behavioral_state.projected_ego_fstates[rel_lane][:FS_DX]
        frenet = behavioral_state.extended_lane_frames[rel_lane]
        obj_fstate_s = frenet.convert_from_segment_state(object.map_state.lane_fstate, object.map_state.lane_id)[:FS_DX]

        # if the object is faster than ego or too far from it, return None
        margin = LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + behavioral_state.ego_state.size.length / 2 + object.size.length / 2
        ds = obj_fstate_s[FS_SX] - ego_fstate_s[FS_SX] - SPECIFICATION_MARGIN_TIME_DELAY * obj_fstate_s[FS_SX] - margin
        dv = ego_fstate_s[FS_SV] - obj_fstate_s[FS_SV]
        if dv <= 0 or ds > BP_ACTION_T_LIMITS[LIMIT_MAX] * dv:
            print('FAILED AGGRESSIVE BRAKING: ds > BP_ACTION_T_LIMITS[LIMIT_MAX] * dv: ego.v=%.3f obj.v=%.3f ds=%.2f' %
                  (ego_fstate_s[1], obj_fstate_s[1], ds))
            return None, None
        # calculate minimal time for the braking
        action_spec = DynamicActionSpace._specify_most_aggressive_action(ego_fstate_s, obj_fstate_s, margin)
        if action_spec is None:
            return None, None
        return aggressive_same_lane_follow_vehicle_idx[0], action_spec

    @staticmethod
    def _specify_most_aggressive_action(ego_fstate_s: np.array, obj_fstate_s: np.array, margin: float) \
            -> Optional[ActionSpec]:
        """
        This method's purpose is to specify the the most aggressive action (typically braking behind of a close slower
        front vehicle) that the agent can take.
        The semantic action (ActionRecipe) is translated into a terminal state specification (ActionSpec).
        :return: semantic action specification [ActionSpec] or [None] if recipe can't be specified.
        """
        # create time horizons grid
        time_horizons = BEHAVIORAL_PLANNING_TIME_RESOLUTION + np.arange(0, 8, BEHAVIORAL_PLANNING_TIME_RESOLUTION)
        v_T = obj_fstate_s[FS_SV]
        # calculate initial longitudinal differences between the object and ego along the target lane
        target_fstates_s = np.c_[obj_fstate_s[FS_SX] - margin + (time_horizons - SPECIFICATION_MARGIN_TIME_DELAY) * v_T,
                                 np.repeat(v_T, time_horizons.shape[0]),
                                 np.zeros_like(time_horizons)]
        duplicated_ego_fstates_s = np.tile(ego_fstate_s, time_horizons.shape[0]).reshape(-1, 3)
        constraints_s = np.concatenate((duplicated_ego_fstates_s, target_fstates_s), axis=-1)

        # check longitudinal acceleration limits for all time horizons
        A_inv = QuinticPoly1D.inverse_time_constraints_tensor(time_horizons)
        poly_coefs = QuinticPoly1D.zip_solve(A_inv, constraints_s)
        acc_in_limits = QuinticPoly1D.are_accelerations_in_limits(poly_coefs, time_horizons, LON_ACC_LIMITS)
        if not acc_in_limits.any():
            print('FAILED AGGRESSIVE BRAKING: no acc_in_limits: ego.v=%.3f obj.v=%.3f dist=%.2f' %
                  (ego_fstate_s[1], obj_fstate_s[1], obj_fstate_s[0] - ego_fstate_s[0] - margin))
            return None
        # choose the minimal T_s, for which the accelerations are in limits
        chosen_time_idx = np.argmax(acc_in_limits)
        T_s = time_horizons[chosen_time_idx]
        chosen_target_fstate = target_fstates_s[chosen_time_idx]

        # Calculate resulting distance from sampling the state at time T from the Quintic polynomial solution.
        # distance_s also takes into account the safe distance that depends on target vehicle velocity that we want
        # to keep from the target vehicle.
        ds = chosen_target_fstate[FS_SX] - ego_fstate_s[FS_SX]
        distance_s = QuinticPoly1D.distance_profile_function(a_0=ego_fstate_s[FS_SA], v_0=ego_fstate_s[FS_SV],
                                                             v_T=chosen_target_fstate[FS_SV], T=T_s, dx=ds,
                                                             T_m=SPECIFICATION_MARGIN_TIME_DELAY)(T_s)
        # lane center has latitude = 0, i.e. spec.d = 0
        return ActionSpec(T_s, chosen_target_fstate[FS_SV], ego_fstate_s[FS_SX] + distance_s, 0, RelativeLane.SAME_LANE)
