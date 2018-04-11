from abc import abstractmethod
from logging import Logger
from typing import List

from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec, ActionRecipe
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_state import SemanticActionsGridState


class StateActionEvaluator:

    def __init__(self, logger: Logger):
        self.logger = logger

    def evaluate_recipes(self, behavioral_state: SemanticActionsGridState, action_recipes: List[ActionRecipe],
                         action_recipes_mask: List[bool])->List[float or None]:
        return [self.evaluate_recipe(behavioral_state, action_recipe) for action_recipe in action_recipes]

    @abstractmethod
    def evaluate_recipe(self, behavioral_state: SemanticActionsGridState, action_recipe: ActionRecipe)->float:
        pass

    def evaluate_action_specs(self, behavioral_state: SemanticActionsGridState, action_specs: List[ActionSpec],
                              action_specs_mask: List[bool])->List[float or None]:
        return [self.evaluate_action_spec(behavioral_state, action_spec) for action_spec in action_specs]

    @abstractmethod
    def evaluate_action_spec(self, behavioral_state: SemanticActionsGridState, action_spec: ActionSpec)->float:
        pass
