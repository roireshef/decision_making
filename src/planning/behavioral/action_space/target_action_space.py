from abc import abstractmethod

import numpy as np
from scipy.interpolate.interpolate import interp1d
from logging import Logger
from typing import Optional, List

import rte.python.profiler as prof
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS, MAX_IMMEDIATE_DECEL, SLOW_DOWN_FACTOR, CLOSE_TO_ZERO_NEGATIVE_VELOCITY, \
    ZERO_SPEED, LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT

from decision_making.src.messages.gap_setting_message import GapSettingState
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, TargetActionRecipe
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import FS_SV, FS_SX, FS_SA, FS_DA, FS_DV, FS_DX
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor


class TargetActionSpace(ActionSpace):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, recipes: List[TargetActionRecipe],
                 filtering: RecipeFiltering):
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

        # Get the headway specifications (+- Comfort_hdw_min/max) based on the user options that were specified
        planner_user_option_state = behavioral_state.ego_state.get_planner_user_option_state()
        headway, min_headway, max_headway = planner_user_option_state.get_headway()

        margin_to_keep_from_targets = LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT

        # here we deduct from the distance to progress: half of lengths of host and target (so we can stay in center-host
        # to center-target distance, plus another margin that will represent the stopping distance, when headway is
        # irrelevant due to 0 velocity
        ds = longitudinal_differences + margin_sign * (
                margin_to_keep_from_targets + behavioral_state.ego_length / 2 + target_lengths / 2)

        T_m = headway
        self.logger.debug(f"Planner Setting Headway: {behavioral_state}")
        self.logger.debug(f"Using Headway: {headway}")

        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        v_0 = projected_ego_fstates[:, FS_SV]
        v_T_mod = self._modify_target_speed_if_ego_is_faster_than_target(behavioral_state, ds, v_0, v_T, headway)
        cost_coeffs_s = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 0], dx=ds, a_0=projected_ego_fstates[:, FS_SA],
            v_0=projected_ego_fstates[:, FS_SV], v_T=v_T_mod, T_m=T_m)
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
        T_s[QuinticPoly1D.is_tracking_mode(v_0, v_T_mod, a_0, ds, T_m)] = 0

        # T_d <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        T_d = KinematicUtils.specify_lateral_planning_time(
            projected_ego_fstates[:, FS_DA], projected_ego_fstates[:, FS_DV], -projected_ego_fstates[:, FS_DX])

        # if both T_d[i] and T_s[i] are defined for i, then take maximum. otherwise leave it nan.
        T = np.maximum(T_d, T_s)

        # Calculate resulting distance from sampling the state at time T from the Quartic polynomial solution.
        # distance_s also takes into account the safe distance that depends on target vehicle velocity that we want
        # to keep from the target vehicle.
        distance_s = QuinticPoly1D.distance_profile_function(a_0=projected_ego_fstates[:, FS_SA],
                                                             v_0=projected_ego_fstates[:, FS_SV],
                                                             v_T=v_T_mod, T=T, dx=ds,
                                                             T_m=T_m)(T)
        # Absolute longitudinal position of target
        target_s = distance_s + projected_ego_fstates[:, FS_SX]

        # lane center has latitude = 0, i.e. spec.d = 0
        action_specs = [ActionSpec(t, vt, st, 0, recipe)
                        if ~np.isnan(t) else None
                        for recipe, t, vt, st in zip(action_recipes, T, v_T_mod, target_s)]

        return action_specs

    def _modify_target_speed_if_ego_is_faster_than_target(self, behavioral_state: BehavioralGridState, ds: np.ndarray,
                                                          v_0: np.ndarray, v_T: np.ndarray, headway: float):
        """
        If the speed of the ego is higher than that of the leading vehicle, set the v_T to be lower than the real v_T.
        This will limit the ego's speed and help in case of sudden brake by the leading vehicle.
        Does not impact RoadSign actions, as its v_T is already 0, so it won't be reduced
        On top of this, need to make sure the v_T is not too low, as this might cause it to be filtered by the
        kinematics filter due to too large deceleration.
        Assume the ego starts at v_0 and breaks at deceleration A. Calculate at what speed the lead vehicle should drive
        in order for the ego to match its speed at time t, and for the distance between vehicles to be the SAFETY
        distance at the same time t.
        This gives us the following 2 equations:
        1. v_T = v_0 - A * t
        2. S0 - (t *(v_T+v_0)/2) + (t * v_T) = SPECIFICATION_HEADWAY * v_T + LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT
        Exchanging t from eq. 1 into eq. 2 gives a quadratic eq in v_T which we solve below.
        We use the smaller root as a lower bound to the speed we can set.
        There are a few special cases to note:
        1. No roots to the equation: This happens when the initial S0 is too small relative to the required headway.
        In this case we keep the v_T unchanged.
        2. Root is larger than v_T. Happens if the leading vehicle is moving too slow and we want it to move faster.
        In this case we keep the v_T unchanged.
        3. Negative root. This happens if the initial S0 is too large relative to v_T. The lead vehicle needs to
        drive in reverse to be at the desired location. In this case we ignore the bound.
        :param behavioral_state: world's state. Used for debug only.
        :param ds: distance between vehicles after subtracting the car's length and the specify margin
        :param v_0: initial ego speed [m/s]
        :param v_T: real target speed [m/s]
        :return: target speed to set for the action
        """
        v_diff = v_T - v_0
        # indices where the leading vehicle is slower than the ego
        mod_idx = np.where(v_diff < CLOSE_TO_ZERO_NEGATIVE_VELOCITY)[0]
        v_T_mod = v_T.copy()
        lower_root = Math.solve_quadratic(np.c_[np.ones(len(mod_idx)),
                                                2 * (MAX_IMMEDIATE_DECEL * headway - v_0[mod_idx]),
                                                v_0[mod_idx] * v_0[mod_idx] - 2 * MAX_IMMEDIATE_DECEL * ds[mod_idx]])[:, 0]
        valid_root_idx = np.where(~np.isnan(lower_root) & (lower_root < v_T[mod_idx]))[0]
        valid_idx = mod_idx[valid_root_idx]
        # select the reduced value where possible
        v_T_mod[valid_idx] = np.maximum(0, v_T_mod[valid_idx] + (v_diff[valid_idx] * SLOW_DOWN_FACTOR))
        # limit by the lower bound
        v_T_mod[valid_idx] = np.maximum(v_T_mod[valid_idx], lower_root[valid_idx])
        self.logger.debug("SlowDown %1.2f, %1.2f, %1.2f, %1.2f, %1.2f, %f", v_T[0], v_0[0], max(0, v_T[0] + (v_diff[0] * SLOW_DOWN_FACTOR)), v_T_mod[0],
                          lower_root[0] if (len(mod_idx) > 0 and ~np.isnan(lower_root[0])) else 0,
                          behavioral_state.ego_state.timestamp_in_sec)
        return v_T_mod
