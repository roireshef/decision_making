from abc import abstractmethod
from logging import Logger
from typing import List

import numpy as np

from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec, ActionRecipe
from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState


class StateActionSpecEvaluator:

    def __init__(self, logger: Logger):
        self.logger = logger

    @abstractmethod
    def evaluate_action_specs(self, behavioral_state: SemanticBehavioralGridState,
                              action_recipes: List[ActionRecipe],
                              action_specs: List[ActionSpec],
                              action_specs_mask: List[bool]) -> np.ndarray:
        # return [self.evaluate_action_spec(behavioral_state, action_spec) for action_spec in action_specs]
        pass


class StateActionRecipeEvaluator:
    def __init__(self, logger: Logger):
        self.logger = logger

    @abstractmethod
    def evaluate_recipes(self, behavioral_state: SemanticBehavioralGridState,
                         action_recipes: List[ActionRecipe],
                         action_recipes_mask: List[bool]) -> np.ndarray:
        # return [self.evaluate_recipe(behavioral_state, action_recipe) for action_recipe in action_recipes]
        pass
