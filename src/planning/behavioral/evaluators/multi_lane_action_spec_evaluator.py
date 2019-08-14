from decision_making.src.exceptions import NoActionsLeftForBPError
from logging import Logger
from typing import List

import numpy as np

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, ActionType, RelativeLane, \
    StaticActionRecipe
from decision_making.src.planning.behavioral.evaluators.action_evaluator import \
    ActionSpecEvaluator
from decision_making.src.global_constants import LANE_END_COST_IND
from decision_making.src.messages.route_plan_message import RoutePlan



class MultiLaneActionSpecEvaluator(ActionSpecEvaluator):
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
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """

        route_costs_dict = route_plan.to_costs_dict()
        # initialize dict to store costs for the different relative_lanes
        lane_costs_dict = {rel_lane: 0. for rel_lane in behavioral_state.extended_lane_frames}

        # get index on each GFF from projected fstates
        lane_index = {rel_lane: behavioral_state.extended_lane_frames[rel_lane].
                                get_closest_index_on_frame(behavioral_state.projected_ego_fstates[rel_lane][0])
                      for rel_lane in behavioral_state.extended_lane_frames}

        # initialize the SAME_LANE to be the one that is chosen
        minimum_cost_lane = RelativeLane.SAME_LANE

        # loop until all segment_ids inside each gff have been looked at
        while np.all([lane_index[rel_lane] < len(behavioral_state.extended_lane_frames[rel_lane].segment_ids)
                      for rel_lane in behavioral_state.extended_lane_frames]):
            # get costs for next lane
            for rel_lane in behavioral_state.extended_lane_frames:
                lane_costs_dict[rel_lane] += route_costs_dict[
                    behavioral_state.extended_lane_frames[rel_lane].segment_ids[lane_index[rel_lane]]][LANE_END_COST_IND]
                lane_index[rel_lane] += 1

            # check if there is a minimum cost lane (if only one element equals the minimum, it is a unique minimum)
            lane_cost_values = list(lane_costs_dict.values())
            if [cost == min(lane_cost_values) for cost in lane_cost_values].count(True) == 1:
                minimum_cost_lane = min(lane_costs_dict, key=lane_costs_dict.get)
                break

        # look at the actions that are in the minimum cost lane
        costs = np.full(len(action_recipes), 1)

        # first try to find a valid dynamic action for SAME_LANE
        follow_vehicle_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                            if action_specs_mask[i]
                                            and recipe.relative_lane == minimum_cost_lane
                                            and recipe.action_type == ActionType.FOLLOW_VEHICLE]
        if len(follow_vehicle_valid_action_idxs) > 0:
            costs[follow_vehicle_valid_action_idxs[0]] = 0  # choose the found dynamic action
            return costs

        filtered_indices = [i for i, recipe in enumerate(action_recipes)
                            if action_specs_mask[i] and isinstance(recipe, StaticActionRecipe)
                            and recipe.relative_lane == minimum_cost_lane]
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
