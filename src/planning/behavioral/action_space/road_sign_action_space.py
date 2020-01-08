from logging import Logger
from typing import List, Type, Optional

import numpy as np
from decision_making.src.global_constants import ROAD_SIGN_LENGTH, LONGITUDINAL_SPECIFY_MARGIN_FROM_STOP_BAR, \
    STOP_BAR_DISTANCE_IND, LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.action_space.target_action_space import TargetActionSpace
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionType, RelativeLongitudinalPosition, \
    RoadSignActionRecipe, PlannerUserOptions
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import FS_SX
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from sklearn.utils.extmath import cartesian


class RoadSignActionSpace(TargetActionSpace):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, filtering: RecipeFiltering, user_options: Optional[PlannerUserOptions] = None):
        super().__init__(logger,
                         predictor=predictor,
                         recipes=[RoadSignActionRecipe.from_args_list(comb)
                                  for comb in cartesian([RelativeLane,
                                                         [RelativeLongitudinalPosition.FRONT],
                                                         [ActionType.FOLLOW_ROAD_SIGN],
                                                         AggressivenessLevel])
                                  ],
                         filtering=filtering,
                         user_options=user_options)

    @property
    def recipe_classes(self) -> List[Type]:
        return [RoadSignActionRecipe]

    def _get_target_lengths(self, action_recipes: List[RoadSignActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        return np.full(len(action_recipes), ROAD_SIGN_LENGTH)

    def _get_target_velocities(self, action_recipes: List[RoadSignActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        return np.zeros(len(action_recipes))  # TODO: will need modification for other road signs

    def _get_end_target_relative_position(self, action_recipes: List[RoadSignActionRecipe]) -> np.ndarray:
        return np.full(len(action_recipes), -1)

    def _get_distance_to_targets(self, action_recipes: List[RoadSignActionRecipe], behavioral_state: BehavioralGridState)\
            -> np.ndarray:
        longitudinal_differences = np.array([self._get_closest_stop_bar_distance(action_recipe, behavioral_state)
                                             for action_recipe in action_recipes])
        self.logger.debug("STOP RoadSign distance: %1.2f" % longitudinal_differences[0])
        assert not np.isnan(longitudinal_differences).any(), "Distance to STOP BAR not found. Should not happen"
        return longitudinal_differences

    def _get_closest_stop_bar_distance(self, action: RoadSignActionRecipe, behavioral_state: BehavioralGridState) \
            -> float:
        """
        Returns the s value of the closest Stop Bar or Stop sign.
        If both types exist, prefer stop bar, if close enough to stop sign.
        No existence checks necessary, as it was already tested by FilterActionsTowardsCellsWithoutRoadSigns
        :param action: the action recipe to be considered
        :param behavioral_state: BehavioralGridState in context
        :return: distance to closest stop bar
        """
        ego_location = behavioral_state.projected_ego_fstates[action.relative_lane][FS_SX]
        closest_TCB_and_its_distance = behavioral_state.get_closest_stop_bar(action.relative_lane)
        return closest_TCB_and_its_distance[STOP_BAR_DISTANCE_IND] - ego_location \
            if closest_TCB_and_its_distance is not None else np.nan
