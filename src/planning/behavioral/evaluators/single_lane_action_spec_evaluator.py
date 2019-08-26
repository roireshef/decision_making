from decision_making.src.exceptions import NoActionsLeftForBPError
from logging import Logger
from typing import List

import numpy as np
from decision_making.src.global_constants import LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_HEADWAY, EPS

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, ActionType, RelativeLane, \
    StaticActionRecipe, DynamicActionRecipe, RelativeLongitudinalPosition
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
        * If there's a leading vehicle, try following it (ActionType.FOLLOW_LANE, lowest aggressiveness possible)
        * If no action from the previous bullet is found valid, find the ActionType.FOLLOW_LANE action with maximal
        allowed velocity and lowest aggressiveness possible.
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_recipes: semantic actions list.
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """
        costs = np.full(len(action_recipes), 1)

        # first try to find a valid dynamic action for SAME_LANE
        follow_vehicle_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                            if action_specs_mask[i]
                                            and recipe.relative_lane == RelativeLane.SAME_LANE
                                            and recipe.action_type == ActionType.FOLLOW_VEHICLE]
        if len(follow_vehicle_valid_action_idxs) > 0:
            # choose aggressiveness level for dynamic action according to the headway safety margin from the front car
            dynamic_specs = [action_specs[action_idx] for action_idx in follow_vehicle_valid_action_idxs]
            print(">>>>>>> AGGR_LEVELS & SAFETY MARGINS >>>>>")
            min_headways = SingleLaneActionSpecEvaluator.calc_minimal_headways(dynamic_specs, behavioral_state)
            aggr_levels = [action_recipes[idx].aggressiveness.value for idx in follow_vehicle_valid_action_idxs]
            spec_t = [action_specs[action_idx].t for action_idx in follow_vehicle_valid_action_idxs]
            print(">>>>>>> AGGR_LEVELS & MIN_HEADWAYS: spec.t", aggr_levels, min_headways, spec_t)
            calm_idx = np.where(aggr_levels == 0)[0]
            standard_idx = np.where(aggr_levels == 1)[0]
            if len(calm_idx) > 0 and min_headways[calm_idx[0]] > SAFETY_HEADWAY + 0.5:
                chosen_level = calm_idx[0]
            elif len(standard_idx) > 0 and min_headways[standard_idx[0]] > SAFETY_HEADWAY + 0.3:
                chosen_level = standard_idx[0]
            else:
                chosen_level = -1  # the most aggressive

            print(">>>>>>> SAFETY MARGINS chosen level", action_recipes[follow_vehicle_valid_action_idxs[chosen_level]].aggressiveness)
            costs[follow_vehicle_valid_action_idxs[chosen_level]] = 0  # choose the found dynamic action
            return costs

        filtered_indices = [i for i, recipe in enumerate(action_recipes)
                            if action_specs_mask[i] and isinstance(recipe, StaticActionRecipe)
                            and recipe.relative_lane == RelativeLane.SAME_LANE]
        if len(filtered_indices) == 0:
            raise NoActionsLeftForBPError("All actions were filtered in BP. timestamp_in_sec: %f" %
                                          behavioral_state.ego_state.timestamp_in_sec)

        # find the minimal aggressiveness level among valid static recipes
        min_aggr_level = min([action_recipes[idx].aggressiveness.value for idx in filtered_indices])

        # find the most fast action with the minimal aggressiveness level
        follow_lane_valid_action_idxs = [idx for idx in filtered_indices
                                         if action_recipes[idx].aggressiveness.value == min_aggr_level]

        # choose the most fast action among the calmest actions;
        # it's last in the recipes list since the recipes are sorted in the increasing order of velocities
        costs[follow_lane_valid_action_idxs[-1]] = 0
        return costs

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

            # minimal margin used in addition to headway (center-to-center of both objects)
            margin = LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + \
                     behavioral_state.ego_state.size.length / 2 + target.dynamic_object.size.length / 2

            # calculate safety margin (on frenet longitudinal axis)
            min_headway = KinematicUtils.calc_safety_margin(poly_s, target_poly_s, margin, SAFETY_HEADWAY, np.array([0, spec.t]))
            min_headways.append(min_headway / max(behavioral_state.ego_state.cartesian_state[C_V], EPS))

        return min_headways
