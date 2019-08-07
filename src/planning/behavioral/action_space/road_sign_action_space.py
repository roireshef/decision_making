from logging import Logger
from typing import List, Type

import numpy as np
from decision_making.src.global_constants import ZERO_SPEED, ROAD_SIGN_LENGTH, LONGITUDINAL_SPECIFY_MARGIN_FROM_STOP_BAR
from decision_making.src.planning.behavioral.action_space.distance_facing_action_space import DistanceFacingActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, \
    ActionType, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import FS_SX
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.utils.map_utils import MapUtils
from sklearn.utils.extmath import cartesian


class RoadSignActionSpace(DistanceFacingActionSpace):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, filtering: RecipeFiltering):
        super().__init__(logger,
                         predictor=predictor,
                         recipes=[DynamicActionRecipe.from_args_list(comb)
                                  for comb in cartesian([RelativeLane,  # [RelativeLane.SAME_LANE],  # TODO - do we want all options??
                                                         [RelativeLongitudinalPosition.FRONT],
                                                         [ActionType.FOLLOW_ROAD_SIGN],
                                                         AggressivenessLevel])
                                  ],
                         filtering=filtering)


    @property
    def recipe_classes(self) -> List[Type]:
        """a list of Recipe classes this action space can handle with"""
        return [DynamicActionRecipe]

    def perform_common(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState):
        """ do any calculation necessary for several abstract methods, to avoid duplication """
        pass

    def get_target_length(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        """ Should return the length of the target object (e.g. cars) for the objects which the actions are
        relative to """
        target_length = np.empty(len(action_recipes))
        target_length.fill(ROAD_SIGN_LENGTH)
        return target_length

    def get_target_velocities(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) \
            -> np.ndarray:
        """ Should return the velocities of the target object (e.g. cars) for the objects which the actions are
        relative to """
        v_T = np.empty(len(action_recipes))
        v_T.fill(ZERO_SPEED)  # TODO: will need modification for other road signs
        return v_T

    def get_end_target_relative_position(self, action_recipes: List[DynamicActionRecipe]) -> np.ndarray:
        """ Should return the relative longitudinal position of the target object (e.g. cars) relative to the ego at the
        end of the action, for the objects which the actions are relative to
        For example: -1 for FOLLOW_VEHICLE (behind target) and +1 for OVER_TAKE_VEHICLE (in front of target)  """
        margin_sign = np.empty(len(action_recipes))
        margin_sign.fill(-1)
        return margin_sign

    def get_distance_to_targets(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState)\
            -> np.ndarray:
        """ Should return the distance of the ego from the target object (e.g. cars) for the objects which the actions
        are relative to """
        longitudinal_differences = np.array([self._get_closest_stop_bar_distance(action_recipe, behavioral_state)
                                             for action_recipe in action_recipes])
        assert not np.isnan(longitudinal_differences).any()
        # TODO DEBUG REMOVE
        print('\x1b[6;30;41m', "distance to stop sign(", len(longitudinal_differences), ")",
              longitudinal_differences[0], '\x1b[0m')
        # TODO DEBUG REMOVE
        return longitudinal_differences

    def _get_closest_stop_bar_distance(self, action: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> \
            float:
        """
        Returns the s value of the closest StaticTrafficFlow.
        No existence checks necessary, as it was already tested by FilterActionsTowardsCellsWithoutRoadSigns
        :param action: the action recipe to be considered
        :param behavioral_state: BehavioralGridState in context
        :return: distance to closest stop bar
        """
        target_lane_frenet = behavioral_state.extended_lane_frames[action.relative_lane]  # the target GFF
        return MapUtils.get_static_traffic_flow_controls_s(target_lane_frenet)[0] - \
               behavioral_state.projected_ego_fstates[action.relative_lane][FS_SX]

    def get_margin_to_keep_from_targets(self, action_recipes: List[DynamicActionRecipe],
                                        behavioral_state: BehavioralGridState) -> float:
        return LONGITUDINAL_SPECIFY_MARGIN_FROM_STOP_BAR
