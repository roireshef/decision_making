from abc import abstractmethod
from typing import List
import numpy as np
from logging import Logger

from decision_making.src.exceptions import BehavioralPlanningException
from decision_making.src.global_constants import SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LON_FRONT, \
    SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LAT_RIGHT, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, MIN_OVERTAKE_VEL, \
    SEMANTIC_CELL_LON_SAME, SEMANTIC_CELL_LON_REAR, SAFE_DIST_TIME_DELAY, LON_ACC_LIMITS
from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec, ActionRecipe, ActionType, \
    SemanticGridCell, LAT_CELL, LON_CELL
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_state import SemanticActionsGridState
from decision_making.src.planning.types import FrenetPoint, FP_SX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from mapping.src.service.map_service import MapService


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

    def evaluate(self, behavioral_state: SemanticActionsGridState,
                 action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec],
                 action_specs_mask: List[bool]) -> np.ndarray:
        """
        Evaluate the generated actions using the actions' spec and SemanticBehavioralState containing semantic grid.
        Gets a list of actions to evaluate and returns a vector representing their costs.
        A set of actions is provided, enabling us to assess them independently.
        Note: the semantic actions were generated using the behavioral state and don't necessarily capture
         all relevant details in the scene. Therefore the evaluation is done using the behavioral state.
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_recipes: semantic actions list.
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """

        pass
