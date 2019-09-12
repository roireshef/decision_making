import numpy as np
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.evaluators.augmented_lane_action_spec_evaluator import \
    AugmentedLaneActionSpecEvaluator
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, DynamicActionRecipe, \
    ActionSpec, ActionRecipe
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
    def __init__(self, behavioral_state: BehavioralGridState, route_plan: RoutePlan, logger: Logger):
        super().__init__(behavioral_state, logger)
        self.route_plan = route_plan
        self.predictor = RoadFollowingPredictor(logger)
        self.action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                          DynamicActionSpace(logger, self.predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
        self.logger.debug('ActionSpec Filters List: %s', [filter.__str__() for filter in self.action_spec_validator._filters])

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
        action_spec_evaluator = AugmentedLaneActionSpecEvaluator(self.logger)
        return action_spec_evaluator.evaluate(self.behavioral_state, self.action_space.recipes, action_specs,
                                              list(action_specs != None), self.route_plan)

    def _choose_action(self, action_specs: ActionSpecArray, costs: np.array) -> [ActionRecipe, ActionSpec]:
        selected_action_index = np.argmin(costs)
        return self.action_space.recipes[selected_action_index], action_specs[selected_action_index]
