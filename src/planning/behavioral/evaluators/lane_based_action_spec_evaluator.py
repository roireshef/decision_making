from decision_making.src.exceptions import NoActionsLeftForBPError
from logging import Logger
from typing import List

import numpy as np

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, ActionType, RelativeLane, \
    StaticActionRecipe, AggressivenessLevel, DynamicActionRecipe, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.evaluators.action_evaluator import \
    ActionSpecEvaluator
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.global_constants import LANE_END_COST_IND, PREFER_LEFT_SPLIT_OVER_RIGHT_SPLIT, EPS, \
    TRAJECTORY_TIME_RESOLUTION, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, REQUIRED_HEADWAY_FOR_CALM_DYNAMIC_ACTION, \
    REQUIRED_HEADWAY_FOR_STANDARD_DYNAMIC_ACTION
from decision_making.src.exceptions import AugmentedGffCreatedIncorrectly
from decision_making.src.planning.types import LIMIT_MIN, LIMIT_MAX, LAT_CELL, FS_SA, FS_SX, FS_SV, Limits
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GFFType
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D


class LaneBasedActionSpecEvaluator(ActionSpecEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate(self, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec], action_specs_mask: List[bool], route_plan: RoutePlan) -> np.ndarray:
        pass


    def _get_follow_vehicle_valid_action_idx(self, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                                             action_specs: List[ActionSpec], action_specs_mask: List[bool], target_lane: RelativeLane) -> int:
        """
        Try to find a valid dynamic action (FOLLOW_VEHICLE)
        The selection is only by aggressiveness, since it relies on the fact that we only follow a vehicle on the
        target lane, which means there is only 1 possible vehicle to follow, so there is only 1 target vehicle speed.
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_recipes: semantic actions list.
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :param target_lane: lane to choose actions from
        :return: index of the chosen action_recipe within action_recipes. If there are no valid actions, -1 is returned
        """
        follow_vehicle_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                            if action_specs_mask[i]
                                            and recipe.relative_lane == target_lane
                                            and recipe.action_type == ActionType.FOLLOW_VEHICLE]
        if len(follow_vehicle_valid_action_idxs) > 0:
            return self._choose_aggressiveness_of_dynamic_action_by_headway(action_recipes, action_specs,
                                                                            behavioral_state, follow_vehicle_valid_action_idxs)
        else:
            return -1

    def _get_follow_road_sign_valid_action_idx(self, action_recipes: List[ActionRecipe], action_specs_mask: List[bool], target_lane: RelativeLane) -> int:
        """
        Try to find a valid road sign action (FOLLOW_ROAD_SIGN) for target_lane
        Selection only needs to consider aggressiveness level, as all the target speeds are ZERO_SPEED.
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :param target_lane: lane to choose actions from
        :return: index of the chosen action_recipe within action_recipes. If there are no valid actions, -1 is returned
        """
        follow_road_sign_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                              if action_specs_mask[i]
                                              and recipe.relative_lane == target_lane
                                              and recipe.action_type == ActionType.FOLLOW_ROAD_SIGN]
        return follow_road_sign_valid_action_idxs[0] if len(follow_road_sign_valid_action_idxs) > 0 else -1

    def _get_follow_lane_valid_action_idx(self, action_recipes: List[ActionRecipe], action_specs_mask: List[bool], target_lane: RelativeLane) -> int:
        """
        Look for valid static action with the minimal aggressiveness level and fastest speed
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :param target_lane: lane to choose actions from
        :return: index of the chosen action_recipe within action_recipes. If there are no valid actions, -1 is returned
        """
        filtered_follow_lane_idxs = [i for i, recipe in enumerate(action_recipes)
                                     if action_specs_mask[i] and isinstance(recipe, StaticActionRecipe)
                                     and recipe.relative_lane == target_lane]

        if len(filtered_follow_lane_idxs) > 0:
            # find the minimal aggressiveness level among valid static recipes
            min_aggr_level = min([action_recipes[idx].aggressiveness.value for idx in filtered_follow_lane_idxs])

            # among the minimal aggressiveness level, find the fastest action
            follow_lane_valid_action_idxs = [idx for idx in filtered_follow_lane_idxs
                                             if action_recipes[idx].aggressiveness.value == min_aggr_level]

            selected_follow_lane_idx = follow_lane_valid_action_idxs[-1]
        else:
            selected_follow_lane_idx = -1
        return selected_follow_lane_idx


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

    def _find_min_cost_augmented_lane(self, behavioral_state: BehavioralGridState, route_plan: RoutePlan) -> RelativeLane:
        """
        Finds the lane with the minimal cost based on route plan costs. Only the SAME_LANE/augmented lanes are considered.
        Only the first upcoming split is considered. The minimum is defined as the lane whose downstream lane at that split
        has the minimal cost.
        :param behavioral_state: Behavioral Grid State which contains the GFFs to be considered
        :param route_plan: Route plan that contains the route costs
        :return: RelativeLane of the BehavioralGridState which has the minimal cost.
        """
        gffs = behavioral_state.extended_lane_frames
        route_costs_dict = route_plan.to_costs_dict()

        # initialize the SAME_LANE to be the one that is chosen
        minimum_cost_lane = RelativeLane.SAME_LANE

        is_left_augmented = RelativeLane.LEFT_LANE in gffs and (
                gffs[RelativeLane.LEFT_LANE].gff_type == GFFType.Augmented or
                gffs[RelativeLane.LEFT_LANE].gff_type == GFFType.AugmentedPartial)

        is_right_augmented = RelativeLane.RIGHT_LANE in gffs and (
                gffs[RelativeLane.RIGHT_LANE].gff_type == GFFType.Augmented or
                gffs[RelativeLane.RIGHT_LANE].gff_type == GFFType.AugmentedPartial)

        diverging_indices = {}

        if is_left_augmented:
            # Find where gffs[RelativeLane.LEFT_LANE].segment_ids and gffs[RelativeLane.SAME_LANE].segment_ids begin to diverge
            max_index = min(len(gffs[RelativeLane.SAME_LANE].segment_ids),
                            len(gffs[RelativeLane.LEFT_LANE].segment_ids))

            try:
                diverging_indices[RelativeLane.LEFT_LANE] = \
                    np.argwhere(gffs[RelativeLane.LEFT_LANE].segment_ids[:max_index] !=
                                gffs[RelativeLane.SAME_LANE].segment_ids[:max_index])[0][0]
            except IndexError:
                is_left_augmented = False
                self.logger.warning(AugmentedGffCreatedIncorrectly(
                    f"Augmented LEFT_LANE and SAME_LANE GFFs contain identical lane segments."))

        if is_right_augmented:
            # Find where gffs[RelativeLane.RIGHT_LANE].segment_ids and gffs[RelativeLane.SAME_LANE].segment_ids begin to diverge
            max_index = min(len(gffs[RelativeLane.SAME_LANE].segment_ids),
                            len(gffs[RelativeLane.RIGHT_LANE].segment_ids))

            try:
                diverging_indices[RelativeLane.RIGHT_LANE] = \
                    np.argwhere(gffs[RelativeLane.RIGHT_LANE].segment_ids[:max_index] !=
                                gffs[RelativeLane.SAME_LANE].segment_ids[:max_index])[0][0]
            except IndexError:
                is_right_augmented = False
                self.logger.warning(AugmentedGffCreatedIncorrectly(
                    f"Augmented RIGHT_LANE and SAME_LANE GFFs contain identical lane segments."))

        # Define lambda function to get route cost of lane
        cost_after_diverge = lambda target_lane, split_side: \
            route_costs_dict[gffs[target_lane].segment_ids[diverging_indices[split_side]]][LANE_END_COST_IND]

        # After determining if the left or right lane diverges first, look at the lane end costs for the lanes where the divergence occurs.
        # Target the lane with the lower cost.
        if is_left_augmented and is_right_augmented:
            # splits are at the same point
            if diverging_indices[RelativeLane.LEFT_LANE] == diverging_indices[RelativeLane.RIGHT_LANE]:
                # Since the splits occur at the same location, the index where the split occurs in the right lane is just used below for
                # all lanes.
                lane_end_costs = {rel_lane: cost_after_diverge(rel_lane, RelativeLane.RIGHT_LANE)
                                  for rel_lane in RelativeLane}
                minimum_cost_lane = min(lane_end_costs, key=lane_end_costs.get)

                # if minimum isn't unique, prefer same_lane if it is a minimum
                # otherwise, choose between left and right based on global_constant
                if lane_end_costs[RelativeLane.SAME_LANE] == lane_end_costs[minimum_cost_lane]:
                    minimum_cost_lane = RelativeLane.SAME_LANE
                elif lane_end_costs[RelativeLane.LEFT_LANE] == lane_end_costs[RelativeLane.RIGHT_LANE]:
                    # If the lane end cost for SAME_LANE is not equal to the minimum lane end cost, then either the left or right lane was returned
                    # above as the lane with the minimum lane end cost. Therefore, if the left and right lane end costs are equal, they are both
                    # minimums, and one of the lanes needs to be chosen.
                    if PREFER_LEFT_SPLIT_OVER_RIGHT_SPLIT:
                        minimum_cost_lane = RelativeLane.LEFT_LANE
                    else:
                        minimum_cost_lane = RelativeLane.RIGHT_LANE

            # splits are at different places
            # left lane splits off first
            elif diverging_indices[RelativeLane.LEFT_LANE] < diverging_indices[RelativeLane.RIGHT_LANE]:
                if cost_after_diverge(RelativeLane.LEFT_LANE, RelativeLane.LEFT_LANE) \
                        < cost_after_diverge(RelativeLane.SAME_LANE, RelativeLane.LEFT_LANE):
                    minimum_cost_lane = RelativeLane.LEFT_LANE

            # right lane splits off first
            else:
                if cost_after_diverge(RelativeLane.RIGHT_LANE, RelativeLane.RIGHT_LANE) \
                        < cost_after_diverge(RelativeLane.SAME_LANE, RelativeLane.RIGHT_LANE):
                    minimum_cost_lane = RelativeLane.RIGHT_LANE

        # only the left lane is augmented
        elif is_left_augmented and \
                (cost_after_diverge(RelativeLane.LEFT_LANE, RelativeLane.LEFT_LANE)
                 < cost_after_diverge(RelativeLane.SAME_LANE, RelativeLane.LEFT_LANE)):
            minimum_cost_lane = RelativeLane.LEFT_LANE

        # only the right lane is augmented
        elif is_right_augmented and \
                (cost_after_diverge(RelativeLane.RIGHT_LANE, RelativeLane.RIGHT_LANE)
                 < cost_after_diverge(RelativeLane.SAME_LANE, RelativeLane.RIGHT_LANE)):
            minimum_cost_lane = RelativeLane.RIGHT_LANE

        return minimum_cost_lane

    def _choose_aggressiveness_of_dynamic_action_by_headway(self, action_recipes: List[ActionRecipe], action_specs: List[ActionSpec],
                                                            behavioral_state: BehavioralGridState, follow_vehicle_valid_action_idxs: List[int]) -> int:
        """ choose aggressiveness level for dynamic action according to the headway safety margin from the front car.
        If the headway becomes small, be more aggressive.
        :param action_recipes: recipes
        :param action_specs: specs
        :param behavioral_state: state of the world
        :param follow_vehicle_valid_action_idxs: indices of valid dynamic actions
        :return: the index of the chosen dynamic action
        """
        dynamic_specs = [action_specs[action_idx] for action_idx in follow_vehicle_valid_action_idxs]
        min_headways = LaneBasedActionSpecEvaluator._calc_minimal_headways(dynamic_specs, behavioral_state)
        aggr_levels = np.array([action_recipes[idx].aggressiveness.value for idx in follow_vehicle_valid_action_idxs])
        calm_idx = np.where(aggr_levels == AggressivenessLevel.CALM.value)[0]
        standard_idx = np.where(aggr_levels == AggressivenessLevel.STANDARD.value)[0]
        aggr_idx = np.where(aggr_levels == AggressivenessLevel.AGGRESSIVE.value)[0]
        if len(calm_idx) > 0 and min_headways[calm_idx[0]] > REQUIRED_HEADWAY_FOR_CALM_DYNAMIC_ACTION:
            chosen_level = calm_idx[0]
        elif len(standard_idx) > 0 and min_headways[standard_idx[0]] > REQUIRED_HEADWAY_FOR_STANDARD_DYNAMIC_ACTION:
            chosen_level = standard_idx[0]
        else:
            chosen_level = -1  # the most aggressive
        self.logger.debug("Headway min %1.2f, %1.2f ,%1.2f,  %d, %f",
                          -1 if len(calm_idx) == 0 else min_headways[calm_idx[0]],
                          -1 if len(standard_idx) == 0 else min_headways[standard_idx[0]],
                          -1 if len(aggr_idx) == 0 else min_headways[aggr_idx[0]],
                          chosen_level,
                          behavioral_state.ego_state.timestamp_in_sec)
        return follow_vehicle_valid_action_idxs[chosen_level]

    @staticmethod
    def _calc_minimal_headways(action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[int]:
        """
        Calculate the minimal headway between ego and targets over the whole trajectory of the action.
        :param action_specs: action specs, defining the trajectories.
        :param behavioral_state: state of the world, from which the target state is extracted.
        :return:
        """
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

        poly_coefs_s = QuinticPoly1D.position_profile_coefficients(
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
            # Uses LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT and not LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT, as otherwise
            # when approaching the leading vehicle, the distance becomes 0, and so does the headway,
            # leading to a selection of AGGRESSIVE action for no reason. Especially noticeable in stop&go tests
            margin = LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + \
                     behavioral_state.ego_state.size.length / 2 + target.dynamic_object.size.length / 2

            # calculate safety margin (on frenet longitudinal axis)
            min_headway = LaneBasedActionSpecEvaluator.calc_minimal_headway_over_trajectory(poly_s, target_poly_s, margin, np.array([0, spec.t]))
            min_headways.append(min_headway)

        return min_headways

    @staticmethod
    def calc_minimal_headway_over_trajectory(poly_host: np.array, poly_target: np.array, margin: float, time_range: Limits):
        """
        Given two s(t) longitudinal polynomials (one for host, one for target), this function calculates the minimal
        headway over the whole trajectory specified by <time_range>.
        Restriction: only relevant for positive velocities.
        :param poly_host: 1d numpy array - coefficients of host's polynomial s(t)
        :param poly_target: 1d numpy array - coefficients of target's polynomial s(t)
        :param margin: the minimal stopping distance to keep in meters (in addition to headway, highly relevant for stopping)
        :param time_range: the relevant range of t for checking the polynomials, i.e. [0, T]
        :return: minimal (on time axis) difference between min. safe distance and actual distance
        """
        # coefficients of host vehicle velocity v_h(t) of host
        vel_poly = np.polyder(poly_host, 1)

        # poly_diff is the polynomial of the distance between poly2 and poly1 with subtracting the required distance
        poly_diff = poly_target - poly_host
        poly_diff[-1] -= margin

        suspected_times = np.arange(time_range[LIMIT_MIN], time_range[LIMIT_MAX] + EPS, TRAJECTORY_TIME_RESOLUTION)

        # This calculates the margin in headway time by checking 64 points evenly spaced in the time range
        # selects the time at which the headway time is minimal
        distances = np.polyval(poly_diff, suspected_times)
        velocities = np.polyval(vel_poly, suspected_times)
        min_headway = np.min(distances / np.maximum(velocities, EPS))
        return min_headway
