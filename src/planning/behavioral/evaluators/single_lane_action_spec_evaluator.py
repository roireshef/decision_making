from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionSpecEvaluator
from logging import Logger
from typing import List

import numpy as np

from decision_making.src.exceptions import BehavioralPlanningException
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, MIN_OVERTAKE_VEL, \
    SPECIFICATION_MARGIN_TIME_DELAY, LON_ACC_LIMITS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, SemanticGridCell
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, ActionType, RelativeLane, \
    RelativeLongitudinalPosition, StaticActionRecipe
from decision_making.src.planning.behavioral.evaluators.action_evaluator import \
    ActionSpecEvaluator
from decision_making.src.planning.types import FrenetPoint, FP_SX, LAT_CELL, FP_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.utils.map_utils import MapUtils


class SingleLaneActionSpecEvaluator(ActionSpecEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate(self, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec], action_specs_mask: List[bool]) -> np.ndarray:
        costs = np.full(len(action_recipes), 1)

        follow_vehicle_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                            if action_specs_mask[i]
                                            and recipe.relative_lane == RelativeLane.SAME_LANE
                                            and recipe.action_type == ActionType.FOLLOW_VEHICLE]

        terminal_velocities = np.unique([recipe.velocity for recipe in action_recipes if isinstance(recipe, StaticActionRecipe)])
        maximal_allowed_velocity = max(terminal_velocities[terminal_velocities <= BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED])

        follow_lane_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                         if action_specs_mask[i] and isinstance(recipe, StaticActionRecipe)
                                         and recipe.relative_lane == RelativeLane.SAME_LANE
                                         and recipe.action_type == ActionType.FOLLOW_LANE
                                         and recipe.velocity == maximal_allowed_velocity]

        if len(follow_vehicle_valid_action_idxs) > 0:
            costs[follow_vehicle_valid_action_idxs[0]] = 0
        else:
            costs[follow_lane_valid_action_idxs[0]] = 0

        return costs
