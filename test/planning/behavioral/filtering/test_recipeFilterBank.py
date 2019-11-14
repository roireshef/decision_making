from typing import List

import numpy as np
from rte.python.logger.AV_logger import AV_Logger

from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterActionsTowardsNonOccupiedCells
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor

from decision_making.test.planning.behavioral.behavioral_state_fixtures import \
    behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode, \
    behavioral_grid_state_with_objects_for_filtering_negative_sT, \
    behavioral_grid_state_with_objects_for_filtering_too_aggressive, \
    state_with_objects_for_filtering_almost_tracking_mode, state_with_objects_for_filtering_negative_sT, \
    state_with_objects_for_filtering_too_aggressive, follow_vehicle_recipes_towards_front_cells, follow_lane_recipes, route_plan_20_30
from decision_making.test.planning.custom_fixtures import turn_signal

def test_filter_recipesWithNonOccupiedCells_filterNonOccupiedCellsActionsOut(
        behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()

    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    filtering = RecipeFiltering(filters=[FilterActionsTowardsNonOccupiedCells()], logger=logger)

    # First three and last three are false because they're recipes of non-occupied cells
    # three middle results are true because tracking a vehicle
    expected_filter_results = np.array([False, False, False, True, True, True, False, False, False], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)
    filter_results = np.array(dynamic_action_space.filter_recipes(follow_vehicle_recipes_towards_front_cells,
                                                                  behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode))

    np.testing.assert_array_equal(filter_results, expected_filter_results)
