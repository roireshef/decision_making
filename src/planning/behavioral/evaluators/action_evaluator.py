from abc import abstractmethod, ABCMeta
from logging import Logger
from typing import List

import numpy as np
import six

import rte.python.profiler as prof
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, ActionRecipe
from decision_making.src.state.state import State


@six.add_metaclass(ABCMeta)
class ActionSpecEvaluator:
    def __init__(self, logger: Logger):
        self.logger = logger

    @staticmethod
    @abstractmethod
    def evaluate(action_recipes: List[ActionRecipe]) -> np.ndarray:
        pass


@six.add_metaclass(ABCMeta)
class ActionRecipeEvaluator:
    def __init__(self, logger: Logger):
        self.logger = logger

    @abstractmethod
    @prof.ProfileFunction()
    def evaluate(self, state: State, behavioral_state: BehavioralGridState,
                 action_recipes: List[ActionRecipe],
                 action_recipes_mask: List[bool],
                 policy) -> np.ndarray:
        pass
