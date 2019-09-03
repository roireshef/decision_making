from decision_making.src.exceptions import NoActionsLeftForBPError
from logging import Logger
from typing import List

import numpy as np

from decision_making.src.exceptions import AugmentedGffCreatedIncorrectly
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, ActionType, RelativeLane, \
    StaticActionRecipe
from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionSpecEvaluator
from decision_making.src.global_constants import LANE_END_COST_IND, PREFER_LEFT_SPLIT_OVER_RIGHT_SPLIT
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GFF_Type


class AugmentedLaneActionSpecEvaluator(ActionSpecEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate(self, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec], action_specs_mask: List[bool], route_plan: RoutePlan) -> np.ndarray:
        """
        Evaluates Action-Specifications based on the following logic:
        * First chooses the relative with the lowest cost based on route_plan
        * Only looks at actions from this minimal cost lane
        * If there's a leading vehicle, try following it (ActionType.FOLLOW_LANE, lowest aggressiveness possible)
        * If no action from the previous bullet is found valid, find the ActionType.FOLLOW_LANE action with maximal
        allowed velocity and lowest aggressiveness possible.
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_recipes: semantic actions list.
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :param route_plan: the route plan which contains lane costs
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """

        # choose the minimum cost lane based on route plan costs
        # the minimum cost lane is defined as the lane who has the minimum cost at the first point where
        # it diverges from the SAME_LANE
        minimum_cost_lane = self.find_min_cost_augmented_lane(behavioral_state, route_plan)

        costs = np.full(len(action_recipes), 1)

        # if an augmented lane is chosen to be the minimum_cost_lane, also allow the possibility of choosing an action
        # on the straight lane if no actions are available on the augmented lane

        # A set is used to prevent duplicates when minimum_cost_lane==RelativeLane.SAME_LANE
        lanes_to_try = {minimum_cost_lane, RelativeLane.SAME_LANE}


        for target_lane in lanes_to_try:
            # first try to find a valid dynamic action (FOLLOW_VEHICLE) for SAME_LANE
            follow_vehicle_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                                if action_specs_mask[i]
                                                and recipe.relative_lane == target_lane
                                                and recipe.action_type == ActionType.FOLLOW_VEHICLE]
            # The selection is only by aggressiveness, since it relies on the fact that we only follow a vehicle on the
            # SAME lane, which means there is only 1 possible vehicle to follow, so there is only 1 target vehicle speed.
            if len(follow_vehicle_valid_action_idxs) > 0:
                costs[follow_vehicle_valid_action_idxs[0]] = 0  # choose the found dynamic action, which is least aggressive
                return costs

            # next try to find a valid road sign action (FOLLOW_ROAD_SIGN) for SAME_LANE.
            # Selection only needs to consider aggressiveness level, as all the target speeds are ZERO_SPEED.
            # Tentative decision is kept in selected_road_sign_idx, to be compared against STATIC actions
            follow_road_sign_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                                  if action_specs_mask[i]
                                                  and recipe.relative_lane == target_lane
                                                  and recipe.action_type == ActionType.FOLLOW_ROAD_SIGN]
            if len(follow_road_sign_valid_action_idxs) > 0:
                # choose the found action, which is least aggressive.
                selected_road_sign_idx = follow_road_sign_valid_action_idxs[0]
            else:
                selected_road_sign_idx = -1

            # last, look for valid static action
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

            # finally decide between the road sign and the static action
            if selected_road_sign_idx < 0 and selected_follow_lane_idx < 0:
                # if no action of either type was found, skip checking for actions on the lane
                continue
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

        # if the for loop is exited without having returned anything, no valid actions were found in both
        # minimum_cost_lane as well as the SAME_LANE
        raise NoActionsLeftForBPError("All actions were filtered in BP. timestamp_in_sec: %f" %
                                      behavioral_state.ego_state.timestamp_in_sec)

    def find_min_cost_augmented_lane(self, behavioral_state: BehavioralGridState, route_plan: RoutePlan):
        gffs = behavioral_state.extended_lane_frames
        route_costs_dict = route_plan.to_costs_dict()

        # initialize the SAME_LANE to be the one that is chosen
        minimum_cost_lane = RelativeLane.SAME_LANE

        is_left_augmented = RelativeLane.LEFT_LANE in gffs and (
                    gffs[RelativeLane.LEFT_LANE].gff_type == GFF_Type.Augmented or
                    gffs[RelativeLane.LEFT_LANE].gff_type == GFF_Type.AugmentedPartial)

        is_right_augmented = RelativeLane.RIGHT_LANE in gffs and (
                    gffs[RelativeLane.RIGHT_LANE].gff_type == GFF_Type.Augmented or
                    gffs[RelativeLane.RIGHT_LANE].gff_type == GFF_Type.AugmentedPartial)

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
