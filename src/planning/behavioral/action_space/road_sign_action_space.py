from logging import Logger
from typing import List, Type

import numpy as np
from decision_making.src.global_constants import ROAD_SIGN_LENGTH, LONGITUDINAL_SPECIFY_MARGIN_FROM_STOP_BAR
from decision_making.src.planning.behavioral.action_space.target_action_space import TargetActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionType, RelativeLongitudinalPosition, \
    RoadSignActionRecipe
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterStopActionIfTooSoon
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from sklearn.utils.extmath import cartesian


class RoadSignActionSpace(TargetActionSpace):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, filtering: RecipeFiltering):
        super().__init__(logger,
                         predictor=predictor,
                         recipes=[RoadSignActionRecipe.from_args_list(comb)
                                  for comb in cartesian([RelativeLane,
                                                         [RelativeLongitudinalPosition.FRONT],
                                                         [ActionType.FOLLOW_ROAD_SIGN],
                                                         AggressivenessLevel])
                                  ],
                         filtering=filtering,
                         margin_to_keep_from_targets=LONGITUDINAL_SPECIFY_MARGIN_FROM_STOP_BAR)

    @property
    def recipe_classes(self) -> List[Type]:
        return [RoadSignActionRecipe]

    def get_target_lengths(self, action_recipes: List[RoadSignActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        return np.full(len(action_recipes), ROAD_SIGN_LENGTH)

    def get_target_velocities(self, action_recipes: List[RoadSignActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        return np.zeros(len(action_recipes))  # TODO: will need modification for other road signs

    def get_end_target_relative_position(self, action_recipes: List[RoadSignActionRecipe]) -> np.ndarray:
        return np.full(len(action_recipes), -1)

    def get_distance_to_targets(self, action_recipes: List[RoadSignActionRecipe], behavioral_state: BehavioralGridState)\
            -> np.ndarray:
        longitudinal_differences = np.array([FilterStopActionIfTooSoon.get_closest_stop_bar_distance(action_recipe,
                                                                                                     behavioral_state)
                                             for action_recipe in action_recipes])
        assert not np.isnan(longitudinal_differences).any()
        return longitudinal_differences
