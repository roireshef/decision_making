from decision_making.src.exceptions import NoActionsLeftForBPError
from logging import Logger
from typing import List

import numpy as np

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, ActionType, RelativeLane, \
    StaticActionRecipe, AggressivenessLevel
from decision_making.src.planning.behavioral.evaluators.action_evaluator import \
    ActionSpecEvaluator
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.global_constants import LANE_END_COST_IND, PREFER_LEFT_SPLIT_OVER_RIGHT_SPLIT
from decision_making.src.exceptions import AugmentedGffCreatedIncorrectly
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GFFType




class LaneBasedActionSpecEvaluator(ActionSpecEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate(self, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec], action_specs_mask: List[bool], route_plan: RoutePlan) -> np.ndarray:
        pass


    def _get_follow_vehicle_valid_action_idx(self, action_recipes: List[ActionRecipe], action_specs_mask: List[bool], target_lane: RelativeLane) -> int:
        """
        Try to find a valid dynamic action (FOLLOW_VEHICLE)
        The selection is only by aggressiveness, since it relies on the fact that we only follow a vehicle on the
        target lane, which means there is only 1 possible vehicle to follow, so there is only 1 target vehicle speed.
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :param target_lane: lane to choose actions from
        :return: index of the chosen action_recipe within action_recipes. If there are no valid actions, -1 is returned
        """
        follow_vehicle_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                            if action_specs_mask[i]
                                            and recipe.relative_lane == target_lane
                                            and recipe.action_type == ActionType.FOLLOW_VEHICLE]
        return follow_vehicle_valid_action_idxs[0] if len(follow_vehicle_valid_action_idxs) > 0 else -1

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
