from logging import Logger
from typing import List, Type

import numpy as np
from decision_making.src.global_constants import LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.action_space.distance_facing_action_space import DistanceFacingActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, \
    ActionType, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import FS_SV
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from sklearn.utils.extmath import cartesian


class DynamicActionSpace(DistanceFacingActionSpace):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, filtering: RecipeFiltering):
        super().__init__(logger,
                         predictor=predictor,
                         recipes=[DynamicActionRecipe.from_args_list(comb)
                                  for comb in cartesian([RelativeLane,
                                                         RelativeLongitudinalPosition,
                                                         [ActionType.FOLLOW_VEHICLE, ActionType.OVERTAKE_VEHICLE],
                                                         AggressivenessLevel])],
                         filtering=filtering)
        self.targets = None
        self.target_map_states = None

    # TODO FOLLOW_VEHICLE for REAR vehicle isn't really supported for 2 reasons:
    #   1. We have no way to guarantee the trajectory we construct does not collide with the rear vehicle
    #   2. Even if (1) was solved, we don't account for other vehicles following the REAR car, so we might collide with them.
    #   This is currently filtered by FilterActionsTowardBackAndParallelCells and FilterLaneChanging

    @property
    def recipe_classes(self) -> List[Type]:
        """a list of Recipe classes this action space can handle with"""
        return [DynamicActionRecipe]

    def perform_common(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState):
        """ do any calculation necessary for several abstract methods, to avoid duplication """
        self.targets = [behavioral_state.road_occupancy_grid[(action_recipe.relative_lane, action_recipe.relative_lon)][0]
                        for action_recipe in action_recipes]
        self.target_map_states = [target.dynamic_object.map_state for target in self.targets]

    def get_target_length(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        """ Should return the length of the target object (e.g. cars) for the objects which the actions are
        relative to """
        target_length = np.array([target.dynamic_object.size.length for target in self.targets])
        return target_length

    def get_target_velocities(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        """ Should return the velocities of the target object (e.g. cars) for the objects which the actions are
        relative to """
        v_T = np.array([map_state.lane_fstate[FS_SV] for map_state in self.target_map_states])
        return v_T

    def get_end_target_relative_position(self, action_recipes: List[DynamicActionRecipe]) -> np.ndarray:
        """ Should return the relative longitudinal position of the target object (e.g. cars) relative to the ego at the
        end of the action, for the objects which the actions are relative to
        For example: -1 for FOLLOW_VEHICLE (behind target) and +1 for OVER_TAKE_VEHICLE (in front of target)  """
        margin_sign = np.array([-1 if action_recipe.action_type == ActionType.FOLLOW_VEHICLE else +1
                                for action_recipe in action_recipes])
        return margin_sign

    def get_distance_to_targets(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState)\
            -> np.ndarray:
        """ Should return the distance of the ego from the target object (e.g. cars) for the objects which the actions
        are relative to """
        longitudinal_differences = behavioral_state.calculate_longitudinal_differences(self.target_map_states)
        assert not np.isinf(longitudinal_differences).any()
        return longitudinal_differences

    def get_margin_to_keep_from_targets(self, action_recipes: List[DynamicActionRecipe],
                                        behavioral_state: BehavioralGridState) -> float:
        return LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT
