from abc import abstractmethod

import numpy as np
from logging import Logger
from typing import Optional, List, Type

import rte.python.profiler as prof
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, SPECIFICATION_HEADWAY, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import FS_SV, FS_SX, FS_SA, FS_DA, FS_DV, FS_DX
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor


class DistanceFacingActionSpace(ActionSpace):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, recipes: List[DynamicActionRecipe],
                 filtering: RecipeFiltering):
        super().__init__(logger,
                         recipes=recipes,
                         recipe_filtering=filtering)
        self.predictor = predictor

    @property
    def recipe_classes(self) -> List[Type]:
        """a list of Recipe classes this action space can handle with"""
        return [DynamicActionRecipe]

    @abstractmethod
    def perform_common(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState):
        """ do any calculation necessary for several abstract methods, to avoid duplication """
        pass

    @abstractmethod
    def get_target_length(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        """ Should return the length of the target object (e.g. cars) for the objects which the actions are
        relative to """
        pass

    @abstractmethod
    def get_target_velocities(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        """ Should return the velocities of the target object (e.g. cars) for the objects which the actions are
        relative to """
        pass

    @abstractmethod
    def get_end_target_relative_position(self, action_recipes: List[DynamicActionRecipe]) -> np.ndarray:
        """ Should return the relative longitudinal position of the target object (e.g. cars) relative to the ego at the
        end of the action, for the objects which the actions are relative to
        For example: -1 for FOLLOW_VEHICLE (behind target) and +1 for OVER_TAKE_VEHICLE (in front of target)  """
        pass

    @abstractmethod
    def get_distance_to_targets(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState)\
            -> np.ndarray:
        """ Should return the distance of the ego from the target object (e.g. cars) for the objects which the actions
        are relative to """
        pass

    @abstractmethod
    def get_margin_to_keep_from_targets(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState)\
            -> float:
        """ Should return the margin the ego should keep from the target object for the objects which the actions
        are relative to """
        pass

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
        relative_lanes = np.array([recipe.relative_lane for recipe in action_recipes])
        projected_ego_fstates = np.array([behavioral_state.projected_ego_fstates[lane] for lane in relative_lanes])

        self.perform_common(action_recipes, behavioral_state)
        # collect targets' lengths, lane_ids and fstates
        # Targets are other vehicles, on the target grid box that ego plans to enter, sorted by S
        target_length = self.get_target_length(action_recipes, behavioral_state)
        v_T = self.get_target_velocities(action_recipes, behavioral_state)
        margin_sign = self.get_end_target_relative_position(action_recipes)

        # calculate initial longitudinal differences between all target objects and ego along target lanes
        longitudinal_differences = self.get_distance_to_targets(action_recipes, behavioral_state)

        # get relevant aggressiveness weights for all actions
        aggressiveness = np.array([action_recipe.aggressiveness.value for action_recipe in action_recipes])
        weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggressiveness]

        # here we deduct from the distance to progress: half of lengths of host and target (so we can stay in center-host
        # to center-target distance, plus another margin that will represent the stopping distance, when headway is
        # irrelevant due to 0 velocity
        ds = longitudinal_differences + margin_sign * (
                self.get_margin_to_keep_from_targets(action_recipes, behavioral_state) +
                behavioral_state.ego_state.size.length / 2 + target_length / 2)

        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 0], dx=ds, a_0=projected_ego_fstates[:, FS_SA],
            v_0=projected_ego_fstates[:, FS_SV], v_T=v_T, T_m=SPECIFICATION_HEADWAY)
        # TODO may be a better idea to to keep actions even if they take "too long".
        #  Instead, let the action evaluator reject such actions.
        #  That is, use here a value larger than BP_ACTION_T_LIMITS[1] = 15 seconds,
        #  or at least, keep the knowledge that this was why they were filtered. This is a weaker result, as the action spec may be filtered by other filters later on.
        #  It will help the evaluator understand that the action may be taken later, and possibly base its decision on that.
        #  For example, consider the case that a STOP at geolocation is possible when an aggressive action is taken, but not when a calm action is employed, due to the time restriction.
        #  This can tell the evaluator that it is better to keep driving normally and start the stopping procedure later on.
        #  On the other hand, if the calm STOP action was removed since the vehicle is too close to the geolocation,
        #  then the evaluator can understand it should start the stopping procedure immediately.
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
                                                             T_m=SPECIFICATION_HEADWAY)(T)
        # Absolute longitudinal position of target
        target_s = distance_s + projected_ego_fstates[:, FS_SX]

        # lane center has latitude = 0, i.e. spec.d = 0
        action_specs = [ActionSpec(t, vt, st, 0, recipe)
                        if ~np.isnan(t) else None
                        for recipe, t, vt, st in zip(action_recipes, T, v_T, target_s)]

        return action_specs
