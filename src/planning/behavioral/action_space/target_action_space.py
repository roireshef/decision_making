from abc import abstractmethod

import numpy as np
from logging import Logger
from typing import Optional, List, Type

import rte.python.profiler as prof
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, SPECIFICATION_HEADWAY, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS, SAFETY_HEADWAY
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, TargetActionRecipe
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import FS_SV, FS_SX, FS_SA, FS_DA, FS_DV, FS_DX
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor


class TargetActionSpace(ActionSpace):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, recipes: List[TargetActionRecipe],
                 filtering: RecipeFiltering, margin_to_keep_from_targets: float):
        """
        Abstract class for Target-Action-Space implementations. Implementations should include actions enumeration,
        filtering and specification.
        :param logger: dedicated logger implementation
        :param predictor: a predictor of target state that is aware of the ego
        :param recipes: list of recipes that define the scope of an ActionSpace implementation
        :param filtering: RecipeFiltering object that holds the logic for filtering recipes
        :param margin_to_keep_from_targets: longitudinal margin to keep from the targets in meters
        """
        super().__init__(logger,
                         recipes=recipes,
                         recipe_filtering=filtering)
        self.predictor = predictor
        self.margin_to_keep_from_targets = margin_to_keep_from_targets

    @abstractmethod
    def _get_target_lengths(self, action_recipes: List[TargetActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        """
        Should return the length of the targets
        :param action_recipes: list of action recipes from which the targets should be extracted
        :param behavioral_state: current state of the world
        :return: array of floats describing the length of the targets
        """
        pass

    @abstractmethod
    def _get_target_velocities(self, action_recipes: List[TargetActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        """
        Should return the velocities of the targets
        :param action_recipes: list of action recipes from which the targets should be extracted
        :param behavioral_state: current state of the world
        :return: array of floats describing the velocities of the targets
        """
        pass

    @abstractmethod
    def _get_end_target_relative_position(self, action_recipes: List[TargetActionRecipe]) -> np.ndarray:
        """
        Should return the relative longitudinal position of the ego relative to the targets at the end of the action
        For example: -1 for FOLLOW_VEHICLE (behind target) and +1 for OVER_TAKE_VEHICLE (in front of target)
        :param action_recipes: list of action recipes from which the targets should be extracted
        :return: array of ints describing the relative positions of the ego relative to the targets at the end of the action
        """
        pass

    @abstractmethod
    def _get_distance_to_targets(self, action_recipes: List[TargetActionRecipe], behavioral_state: BehavioralGridState)\
            -> np.ndarray:
        """
        Should return the distance of the ego from the targets before the action is taken
        :param action_recipes: list of action recipes from which the targets should be extracted
        :param behavioral_state: current state of the world
        :return: array of floats describing the distance of the ego from the targets before the action is taken
        """
        pass

    @prof.ProfileFunction()
    def specify_goals(self, action_recipes: List[TargetActionRecipe], behavioral_state: BehavioralGridState) -> \
            List[Optional[ActionSpec]]:
        """
        This method's purpose is to specify the enumerated actions (recipes) that the agent can take.
        Each semantic action (ActionRecipe) is translated into a terminal state specification (ActionSpec).
        :param action_recipes: a list of enumerated semantic actions [ActionRecipe].
        :param behavioral_state: a Frenet state of ego at initial point
        :return: semantic action specification [ActionSpec] or [None] if recipe can't be specified.
        """
        # pick ego initial fstates projected on all target frenet_frames
        relative_lanes = np.array([recipe.relative_lane for recipe in action_recipes])
        projected_ego_fstates = np.array([behavioral_state.projected_ego_fstates[lane] for lane in relative_lanes])

        # collect targets' lengths, lane_ids and fstates
        # Targets are other vehicles, on the target grid box that ego plans to enter, sorted by S
        target_lengths = self._get_target_lengths(action_recipes, behavioral_state)
        v_T = self._get_target_velocities(action_recipes, behavioral_state)
        margin_sign = self._get_end_target_relative_position(action_recipes)

        # calculate initial longitudinal differences between all target objects and ego along target lanes
        longitudinal_differences = self._get_distance_to_targets(action_recipes, behavioral_state)

        # get relevant aggressiveness weights for all actions
        aggressiveness = np.array([action_recipe.aggressiveness.value for action_recipe in action_recipes])
        weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggressiveness]

        # here we deduct from the distance to progress: half of lengths of host and target (so we can stay in center-host
        # to center-target distance, plus another margin that will represent the stopping distance, when headway is
        # irrelevant due to 0 velocity
        ds = longitudinal_differences + margin_sign * (
                self.margin_to_keep_from_targets + behavioral_state.ego_state.size.length / 2 + target_lengths / 2)

        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        v_0 = projected_ego_fstates[:, FS_SV]
        MIN_VEL = 0.1
        MAX_VEL_RATIO = 3.0
        EXTEND_RATIO = 3.0
        MAX_VAL = ((MAX_VEL_RATIO - 1) * EXTEND_RATIO + 1)
        accelerating_idxs = v_T + MIN_VEL <= v_0
        T_m = np.full(len(v_0), SPECIFICATION_HEADWAY)
        v_ratio = (v_0[accelerating_idxs] / (v_T[accelerating_idxs] + MIN_VEL))
        headway_extension = ((np.minimum(v_ratio, MAX_VEL_RATIO) - 1) * EXTEND_RATIO + 1)
        desired_headway = SPECIFICATION_HEADWAY * headway_extension
        # make sure this is not longer than physically possible - keep 1 second margin for action
        time_to_match_speed = (v_0[accelerating_idxs] - v_T[accelerating_idxs]) / 5.0 # this is the time it takes us to match target speed with max deceleraton
        target_location_at_match_speed = ds[accelerating_idxs] + (time_to_match_speed *v_T[accelerating_idxs])
        ego_location_at_match_speed = (v_0[accelerating_idxs] + v_T[accelerating_idxs]) / 2 *time_to_match_speed
        current_headway = (target_location_at_match_speed - ego_location_at_match_speed) / (v_T[accelerating_idxs] + MIN_VEL)
        T_m[accelerating_idxs] = np.maximum(np.minimum(desired_headway, current_headway), SAFETY_HEADWAY)
        # if np.any(accelerating_idxs):
        #     print("++++ v_0", v_0, "v_T", v_T, "T_m", T_m)
        if len(desired_headway)>0:
            des_hw = desired_headway[0]
            curr_hw = max(current_headway[0], 0)
        else:
            des_hw = SPECIFICATION_HEADWAY
            curr_hw = SPECIFICATION_HEADWAY
        self.logger.debug("Headway set %1.2f, %1.2f ,%1.2f,  %f", des_hw, curr_hw, T_m[0], behavioral_state.ego_state.timestamp_in_sec)
        # T_m = SPECIFICATION_HEADWAY
        cost_coeffs_s = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 0], dx=ds, a_0=projected_ego_fstates[:, FS_SA],
            v_0=projected_ego_fstates[:, FS_SV], v_T=v_T, T_m=T_m)
        # TODO see https://confluence.gm.com/display/ADS133317/Stop+at+Geo+location+remaining+issues for possibly extending the allowed action time
        roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, BP_ACTION_T_LIMITS)
        T_s = np.fmin.reduce(roots_s, axis=-1)

        # Agent is in tracking mode, meaning the required velocity change is negligible and action time is actually
        # zero. This degenerate action is valid but can't be solved analytically thus we probably got nan for T_s
        # although it should be zero. Here we can't find a local minima as the equation is close to a linear line,
        # intersecting in T=0.
        # TODO: this creates 3 actions (different aggressiveness levels) which are the same, in case of tracking mode
        v_0 = behavioral_state.ego_state.map_state.lane_fstate[FS_SV]
        a_0 = behavioral_state.ego_state.map_state.lane_fstate[FS_SA]
        T_s[QuinticPoly1D.is_tracking_mode(v_0, v_T, a_0, ds, SPECIFICATION_HEADWAY)] = 0

        # T_d <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_d = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 1], dx=-projected_ego_fstates[:, FS_DX],
            a_0=projected_ego_fstates[:, FS_DA], v_0=projected_ego_fstates[:, FS_DV], v_T=0, T_m=0)
        roots_d = Math.find_real_roots_in_limits(cost_coeffs_d, BP_ACTION_T_LIMITS)
        T_d = np.fmin.reduce(roots_d, axis=-1)

        # if both T_d[i] and T_s[i] are defined for i, then take maximum. otherwise leave it nan.
        T = np.maximum(T_d, T_s)

        # Calculate resulting distance from sampling the state at time T from the Quartic polynomial solution.
        # distance_s also takes into account the safe distance that depends on target vehicle velocity that we want
        # to keep from the target vehicle.
        distance_s = QuinticPoly1D.distance_profile_function(a_0=projected_ego_fstates[:, FS_SA],
                                                             v_0=projected_ego_fstates[:, FS_SV],
                                                             v_T=v_T, T=T, dx=ds,
                                                             T_m=T_m)(T)
        # Absolute longitudinal position of target
        target_s = distance_s + projected_ego_fstates[:, FS_SX]

        # lane center has latitude = 0, i.e. spec.d = 0
        action_specs = [ActionSpec(t, vt, st, 0, recipe)
                        if ~np.isnan(t) else None
                        for recipe, t, vt, st in zip(action_recipes, T, v_T, target_s)]

        return action_specs
