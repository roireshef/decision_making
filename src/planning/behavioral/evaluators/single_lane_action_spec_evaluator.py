from decision_making.src.exceptions import NoActionsLeftForBPError
from logging import Logger
from typing import List

import numpy as np
from decision_making.src.global_constants import LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_HEADWAY, EPS, \
    LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, ActionType, RelativeLane, \
    StaticActionRecipe, DynamicActionRecipe, RelativeLongitudinalPosition, AggressivenessLevel
from decision_making.src.planning.behavioral.evaluators.action_evaluator import \
    ActionSpecEvaluator
from decision_making.src.planning.types import LAT_CELL, FS_SA, FS_SV, FS_DX, C_V, FS_SX
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D


class SingleLaneActionSpecEvaluator(ActionSpecEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate(self, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec], action_specs_mask: List[bool]) -> np.ndarray:
        """
        Evaluates Action-Specifications based on the following logic:
        * Only takes into account actions on RelativeLane.SAME_LANE
        * If there's a leading vehicle, try following it (ActionType.FOLLOW_VEHICLE, lowest aggressiveness possible)
        * If no action from the previous bullet is found valid, find the ActionType.FOLLOW_ROAD_SIGN action with lowest
        * aggressiveness, and save it.
        * Find the ActionType.FOLLOW_LANE action with maximal allowed velocity and lowest aggressiveness possible,
        * and save it.
        * Compare the saved FOLLOW_ROAD_SIGN and FOLLOW_LANE actions, and choose between them.
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_recipes: semantic actions list.
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """
        costs = np.full(len(action_recipes), 1)

        # first try to find a valid dynamic action (FOLLOW_VEHICLE) for SAME_LANE
        follow_vehicle_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                            if action_specs_mask[i]
                                            and recipe.relative_lane == RelativeLane.SAME_LANE
                                            and recipe.action_type == ActionType.FOLLOW_VEHICLE]
        # The selection is only by aggressiveness, since it relies on the fact that we only follow a vehicle on the
        # SAME lane, which means there is only 1 possible vehicle to follow, so there is only 1 target vehicle speed.
        if len(follow_vehicle_valid_action_idxs) > 0:
            # choose aggressiveness level for dynamic action according to the headway safety margin from the front car
            dynamic_specs = [action_specs[action_idx] for action_idx in follow_vehicle_valid_action_idxs]
            print(">>>>>>> AGGR_LEVELS & SAFETY MARGINS >>>>>")
            min_headways = SingleLaneActionSpecEvaluator.calc_minimal_headways(dynamic_specs, behavioral_state)
            aggr_levels = np.array(
                [action_recipes[idx].aggressiveness.value for idx in follow_vehicle_valid_action_idxs])
            spec_t = [action_specs[action_idx].t for action_idx in follow_vehicle_valid_action_idxs]
            calm_idx = np.where(aggr_levels == 0)[0]
            standard_idx = np.where(aggr_levels == 1)[0]
            aggr_idx = np.where(aggr_levels == 2)[0]
            print(">>>>>>> AGGR_LEVELS & MIN_HEADWAYS: spec.t", aggr_levels, min_headways, spec_t, "calm_idx", calm_idx,
                  "standard_idx", standard_idx)
            if len(calm_idx) > 0 and min_headways[calm_idx[0]] > SAFETY_HEADWAY + 0.7:
                chosen_level = calm_idx[0]
            elif len(standard_idx) > 0 and min_headways[standard_idx[0]] > SAFETY_HEADWAY + 0.5:
                chosen_level = standard_idx[0]
            else:
                chosen_level = -1  # the most aggressive
            # chosen_level = -1       # TODO OVERRIDE FOR TESTING
            chosen_level = follow_vehicle_valid_action_idxs[0]  # BASIC - calmest

            self.logger.debug("Headway min %1.2f, %1.2f ,%1.2f,  %d, %f", -1 if len(calm_idx) == 0 else min_headways[calm_idx[0]],
                              -1 if len(standard_idx) == 0 else min_headways[standard_idx[0]],
                              -1 if len(aggr_idx) == 0 else min_headways[aggr_idx[0]],
                              chosen_level,
                              behavioral_state.ego_state.timestamp_in_sec)
            print(">>>>>>> SAFETY MARGINS chosen level",
                  action_recipes[follow_vehicle_valid_action_idxs[chosen_level]].aggressiveness)
            costs[follow_vehicle_valid_action_idxs[chosen_level]] = 0  # choose the found dynamic action
            return costs

        # next try to find a valid road sign action (FOLLOW_ROAD_SIGN) for SAME_LANE.
        # Selection only needs to consider aggressiveness level, as all the target speeds are ZERO_SPEED.
        # Tentative decision is kept in selected_road_sign_idx, to be compared against STATIC actions
        follow_road_sign_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                              if action_specs_mask[i]
                                              and recipe.relative_lane == RelativeLane.SAME_LANE
                                              and recipe.action_type == ActionType.FOLLOW_ROAD_SIGN]
        if len(follow_road_sign_valid_action_idxs) > 0:
            # choose the found action, which is least aggressive.
            selected_road_sign_idx = follow_road_sign_valid_action_idxs[0]
        else:
            selected_road_sign_idx = -1

        # last, look for valid static action
        filtered_follow_lane_idxs = [i for i, recipe in enumerate(action_recipes)
                            if action_specs_mask[i] and isinstance(recipe, StaticActionRecipe)
                            and recipe.relative_lane == RelativeLane.SAME_LANE]
        if len(filtered_follow_lane_idxs) > 0:
            # find the minimal aggressiveness level among valid static recipes
            min_aggr_level = min([action_recipes[idx].aggressiveness.value for idx in filtered_follow_lane_idxs])

            # among the minimal aggressiveness level, find the fastest action
            follow_lane_valid_action_idxs = [idx for idx in filtered_follow_lane_idxs
                                             if action_recipes[idx].aggressiveness.value == min_aggr_level]

            selected_follow_lane_idx = follow_lane_valid_action_idxs[-1]
        else:
            selected_follow_lane_idx = -1

        # finally decide between the road sign and the static action
        if selected_road_sign_idx < 0 and selected_follow_lane_idx < 0:
            # if no action of either type was found, raise an error
            raise NoActionsLeftForBPError("All actions were filtered in BP. timestamp_in_sec: %f" %
                                          behavioral_state.ego_state.timestamp_in_sec)
        elif selected_road_sign_idx < 0:
            # if no road sign action is found, select the static action
            costs[selected_follow_lane_idx] = 0
            return costs
        elif selected_follow_lane_idx < 0:
            # if no static action is found, select the road sign action
            costs[selected_road_sign_idx] = 0
            return costs
        else:
            # if both road sign and static actions are valid - choose
            if self._is_static_action_preferred(action_recipes, selected_road_sign_idx, selected_follow_lane_idx):
                costs[selected_follow_lane_idx] = 0
                return costs
            else:
                costs[selected_road_sign_idx] = 0
                return costs

    def _is_static_action_preferred(self, action_recipes: List[ActionRecipe], road_sign_idx: int, follow_lane_idx: int):
        """
        Selects if a STATIC or ROAD_SIGN action is preferred.
        This can be based on any criteria.
        For example:
            always prefer 1 type of action,
            select action type if aggressiveness is as desired
            Toggle between the 2
        :param action_recipes: of all possible actions
        :param road_sign_idx: of calmest road sign action
        :param follow_lane_idx: of calmest fastest static action
        :return: True if static action is preferred, False otherwise
        """
        road_sign_action = action_recipes[road_sign_idx]
        # Avoid AGGRESSIVE stop. TODO relax the restriction of not selective an aggressive road sign
        return road_sign_action.aggressiveness != AggressivenessLevel.CALM

    @staticmethod
    def calc_minimal_headways(action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[int]:
        """ This is a temporary filter that replaces a more comprehensive test suite for safety w.r.t the target vehicle
         of a dynamic action or towards a leading vehicle in a static action. The condition under inspection is of
         maintaining the required safety-headway + constant safety-margin"""
        # Extract the grid cell relevant for that action (for static actions it takes the front cell's actor,
        # so this filter is actually applied to static actions as well). Then query the cell for the target vehicle
        relative_cells = [(spec.recipe.relative_lane,
                           spec.recipe.relative_lon if isinstance(spec.recipe, DynamicActionRecipe) else RelativeLongitudinalPosition.FRONT)
                          for spec in action_specs]
        target_vehicles = [behavioral_state.road_occupancy_grid[cell][0]
                           if len(behavioral_state.road_occupancy_grid[cell]) > 0 else None
                           for cell in relative_cells]
        T = np.array([spec.t for spec in action_specs])

        # represent initial and terminal boundary conditions (for s axis)
        initial_fstates = np.array([behavioral_state.projected_ego_fstates[cell[LAT_CELL]] for cell in relative_cells])
        terminal_fstates = np.array([spec.as_fstate() for spec in action_specs])

        poly_coefs_s = QuinticPoly1D.s_profile_coefficients(
            initial_fstates[:, FS_SA], initial_fstates[:, FS_SV], terminal_fstates[:, FS_SV],
            terminal_fstates[:, FS_SX] - initial_fstates[:, FS_SX], T)
        poly_coefs_s[:, -1] = initial_fstates[:, FS_SX]

        min_headways = []
        for poly_s, cell, target, spec in zip(poly_coefs_s, relative_cells, target_vehicles, action_specs):
            if target is None:
                min_headways.append(np.inf)
                continue

            target_fstate = behavioral_state.extended_lane_frames[cell[LAT_CELL]].convert_from_segment_state(
                target.dynamic_object.map_state.lane_fstate, target.dynamic_object.map_state.lane_id)
            target_poly_s, _ = KinematicUtils.create_linear_profile_polynomial_pair(target_fstate)

            # minimal margin used in addition to headway (center-to-center of both objects) #LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
            margin = LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT + \
                     behavioral_state.ego_state.size.length / 2 + target.dynamic_object.size.length / 2

            # calculate safety margin (on frenet longitudinal axis)
            min_headway = KinematicUtils.calc_safety_margin(poly_s, target_poly_s, margin, SAFETY_HEADWAY, np.array([0, spec.t]))
            min_headways.append(min_headway)

        return min_headways
