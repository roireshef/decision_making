import numpy as np
from decision_making.src.exceptions import NoActionsLeftForBPError
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, DynamicActionRecipe, \
    ActionSpec, AggressivenessLevel, RelativeLane, ActionType, ActionRecipe
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING, \
    DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from logging import Logger
from typing import List

from decision_making.src.planning.types import ActionSpecArray
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor


class SingleStepBehavioralPlanner(BasePlanner):
    """
    For each received current-state:
     1.A behavioral, semantic state is created and its value is approximated.
     2.The full action-space is enumerated (recipes are generated).
     3.Recipes are filtered according to some pre-defined rules.
     4.Recipes are specified so ActionSpecs are created.
     5.Action Specs are evaluated.
     6.Lowest-Cost ActionSpec is chosen and its parameters are sent to TrajectoryPlanner.
    """
    def __init__(self, behavioral_state: BehavioralGridState, logger: Logger):
        super().__init__(behavioral_state, logger)
        self.predictor = RoadFollowingPredictor(logger)
        self.action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                          DynamicActionSpace(logger, self.predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
        self.logger.debug('ActionSpec Filters List: %s', [filter.__str__() for filter in action_spec_validator._filters])

    def _create_actions(self) -> ActionSpecArray:
        action_recipes = self.action_space.recipes

        # Recipe filtering
        recipes_mask = self.action_space.filter_recipes(action_recipes, self.behavioral_state)
        self.logger.debug('Number of actions originally: %d, valid: %d',
                          self.action_space.action_space_size, np.sum(recipes_mask))

        action_specs = np.full(len(action_recipes), None)
        valid_action_recipes = [action_recipe for i, action_recipe in enumerate(action_recipes) if recipes_mask[i]]
        action_specs[recipes_mask] = self.action_space.specify_goals(valid_action_recipes, self.behavioral_state)

        # TODO: FOR DEBUG PURPOSES!
        num_of_considered_static_actions = sum(isinstance(x, StaticActionRecipe) for x in valid_action_recipes)
        num_of_considered_dynamic_actions = sum(isinstance(x, DynamicActionRecipe) for x in valid_action_recipes)
        num_of_specified_actions = sum(x is not None for x in action_specs)
        self.logger.debug('Number of actions specified: %d (#%dS,#%dD)',
                          num_of_specified_actions, num_of_considered_static_actions, num_of_considered_dynamic_actions)
        return action_specs

    def _filter_actions(self, action_specs: ActionSpecArray) -> ActionSpecArray:
        action_specs_mask = self.action_spec_validator.filter_action_specs(action_specs, self.behavioral_state)
        filtered_action_specs = np.full(len(action_specs), None)
        filtered_action_specs[action_specs_mask] = action_specs[action_specs_mask]
        return filtered_action_specs

    def _evaluate(self, action_specs: ActionSpecArray) -> np.ndarray:
        """
        Evaluates Action-Specifications based on the following logic:
        * Only takes into account actions on RelativeLane.SAME_LANE
        * If there's a leading vehicle, try following it (ActionType.FOLLOW_VEHICLE, lowest aggressiveness possible)
        * If no action from the previous bullet is found valid, find the ActionType.FOLLOW_ROAD_SIGN action with lowest
        * aggressiveness, and save it.
        * Find the ActionType.FOLLOW_LANE action with maximal allowed velocity and lowest aggressiveness possible,
        * and save it.
        * Compare the saved FOLLOW_ROAD_SIGN and FOLLOW_LANE actions, and choose between them.
        :param action_specs: specifications of action_recipes.
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """
        costs = np.full(len(action_specs), 1)

        # first try to find a valid dynamic action (FOLLOW_VEHICLE) for SAME_LANE
        follow_vehicle_valid_action_idxs = [i for i, spec in enumerate(action_specs)
                                            if spec is not None
                                            and spec.recipe.relative_lane == RelativeLane.SAME_LANE
                                            and spec.recipe.action_type == ActionType.FOLLOW_VEHICLE]
        # The selection is only by aggressiveness, since it relies on the fact that we only follow a vehicle on the
        # SAME lane, which means there is only 1 possible vehicle to follow, so there is only 1 target vehicle speed.
        if len(follow_vehicle_valid_action_idxs) > 0:
            costs[follow_vehicle_valid_action_idxs[0]] = 0  # choose the found dynamic action, which is least aggressive
            return costs

        # next try to find a valid road sign action (FOLLOW_ROAD_SIGN) for SAME_LANE.
        # Selection only needs to consider aggressiveness level, as all the target speeds are ZERO_SPEED.
        # Tentative decision is kept in selected_road_sign_idx, to be compared against STATIC actions
        follow_road_sign_valid_action_idxs = [i for i, spec in enumerate(action_specs)
                                              if spec is not None
                                              and spec.recipe.relative_lane == RelativeLane.SAME_LANE
                                              and spec.recipe.action_type == ActionType.FOLLOW_ROAD_SIGN]
        if len(follow_road_sign_valid_action_idxs) > 0:
            # choose the found action, which is least aggressive.
            selected_road_sign_idx = follow_road_sign_valid_action_idxs[0]
        else:
            selected_road_sign_idx = -1

        # last, look for valid static action
        filtered_follow_lane_idxs = [i for i, spec in enumerate(action_specs)
                                     if spec is not None and isinstance(spec.recipe, StaticActionRecipe)
                                     and spec.recipe.relative_lane == RelativeLane.SAME_LANE]
        if len(filtered_follow_lane_idxs) > 0:
            # find the minimal aggressiveness level among valid static recipes
            min_aggr_level = min([action_specs[idx].recipe.aggressiveness.value for idx in filtered_follow_lane_idxs])

            # among the minimal aggressiveness level, find the fastest action
            follow_lane_valid_action_idxs = [idx for idx in filtered_follow_lane_idxs
                                             if action_specs[idx].recipe.aggressiveness.value == min_aggr_level]

            selected_follow_lane_idx = follow_lane_valid_action_idxs[-1]
        else:
            selected_follow_lane_idx = -1

        # finally decide between the road sign and the static action
        if selected_road_sign_idx < 0 and selected_follow_lane_idx < 0:
            # if no action of either type was found, raise an error
            raise NoActionsLeftForBPError("All actions were filtered in BP. timestamp_in_sec: %f" %
                                          self.behavioral_state.ego_state.timestamp_in_sec)
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
            if SingleStepBehavioralPlanner._is_static_action_preferred(action_specs, selected_road_sign_idx):
                costs[selected_follow_lane_idx] = 0
                return costs
            else:
                costs[selected_road_sign_idx] = 0
                return costs

    @staticmethod
    def _is_static_action_preferred(action_specs: List[ActionSpec], road_sign_idx: int):
        """
        Selects if a STATIC or ROAD_SIGN action is preferred.
        This can be based on any criteria.
        For example:
            always prefer 1 type of action,
            select action type if aggressiveness is as desired
            Toggle between the 2
        :param action_specs: of all possible actions
        :param road_sign_idx: of calmest road sign action
        :return: True if static action is preferred, False otherwise
        """
        road_sign_action = action_specs[road_sign_idx]
        # Avoid AGGRESSIVE stop. TODO relax the restriction of not selective an aggressive road sign
        return road_sign_action.recipe.aggressiveness != AggressivenessLevel.CALM

    def _choose_action(self, action_specs: ActionSpecArray, costs: np.array) -> [ActionRecipe, ActionSpec]:
        selected_action_index = np.argmin(costs)
        return self.action_space.recipes[selected_action_index], action_specs[selected_action_index]
