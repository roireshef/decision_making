from abc import abstractmethod, ABCMeta
from logging import Logger
from typing import List

import numpy as np
import six

import rte.python.profiler as prof
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, ActionRecipe
from decision_making.src.messages.route_plan_message import RoutePlan


@six.add_metaclass(ABCMeta)
class ActionSpecEvaluator:
    def __init__(self, logger: Logger):
        self.logger = logger

    @abstractmethod
    @prof.ProfileFunction()
    def evaluate(self, behavioral_state: BehavioralGridState,
                 action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec],
                 action_specs_mask: List[bool],
                 route_plan: RoutePlan) -> np.ndarray:
        pass


@six.add_metaclass(ABCMeta)
class ActionRecipeEvaluator:
    def __init__(self, logger: Logger):
        self.logger = logger

    @abstractmethod
    @prof.ProfileFunction()
    def evaluate(self, behavioral_state: BehavioralGridState,
                 action_recipes: List[ActionRecipe],
                 action_recipes_mask: List[bool]) -> np.ndarray:
        pass

