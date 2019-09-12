from logging import Logger
from typing import List, Type

import numpy as np
from decision_making.src.global_constants import LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.action_space.target_action_space import TargetActionSpace
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, \
    ActionType, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import FS_SV
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.planning.behavioral.state.state import DynamicObject
from sklearn.utils.extmath import cartesian


class DynamicActionSpace(TargetActionSpace):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, filtering: RecipeFiltering):
        super().__init__(logger,
                         predictor=predictor,
                         recipes=[DynamicActionRecipe.from_args_list(comb)
                                  for comb in cartesian([RelativeLane,
                                                         RelativeLongitudinalPosition,
                                                         [ActionType.FOLLOW_VEHICLE, ActionType.OVERTAKE_VEHICLE],
                                                         AggressivenessLevel])],
                         filtering=filtering,
                         margin_to_keep_from_targets=LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT)

    # TODO FOLLOW_VEHICLE for REAR vehicle isn't really supported for 2 reasons:
    #   1. We have no way to guarantee the trajectory we construct does not collide with the rear vehicle
    #   2. Even if (1) was solved, we don't account for other vehicles following the REAR car, so we might collide with them.
    #   This is currently filtered by FilterActionsTowardBackAndParallelCells and FilterLaneChanging

    @property
    def recipe_classes(self) -> List[Type]:
        return [DynamicActionRecipe]

    def _get_closest_target(self, action_recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> DynamicObject:
        """
        Get the closest target on the BGS for the grid state to which the action is pointing.
        :param action_recipe: pointing to the relevant grid state
        :param behavioral_state: from which the target object is taken
        :return: the closest object for the given action
        """
        return behavioral_state.road_occupancy_grid[(action_recipe.relative_lane, action_recipe.relative_lon)][0].\
            dynamic_object

    def _get_target_lengths(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        target_lengths = np.array([self._get_closest_target(action_recipe, behavioral_state).size.length
                                  for action_recipe in action_recipes])

        return target_lengths

    def _get_target_velocities(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        v_T = np.array([self._get_closest_target(action_recipe, behavioral_state).map_state.lane_fstate[FS_SV]
                        for action_recipe in action_recipes])
        return v_T

    def _get_end_target_relative_position(self, action_recipes: List[DynamicActionRecipe]) -> np.ndarray:
        margin_sign = np.array([-1 if action_recipe.action_type == ActionType.FOLLOW_VEHICLE else +1
                                for action_recipe in action_recipes])
        return margin_sign

    def _get_distance_to_targets(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState)\
            -> np.ndarray:
        target_map_states = [self._get_closest_target(action_recipe, behavioral_state).map_state
                             for action_recipe in action_recipes]
        longitudinal_differences = behavioral_state.calculate_longitudinal_differences(target_map_states)
        assert not np.isinf(longitudinal_differences).any()
        return longitudinal_differences
