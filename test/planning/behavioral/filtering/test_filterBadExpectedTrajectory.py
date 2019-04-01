import numpy as np
from typing import List

from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, StaticActionRecipe
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterForKinematics, FilterIfNone
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterBadExpectedTrajectory
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from rte.python.logger.AV_logger import AV_Logger

from decision_making.test.messages.static_scene_fixture import scene_static
from decision_making.test.planning.behavioral.behavioral_state_fixtures import \
    behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode, \
    behavioral_grid_state_with_objects_for_filtering_negative_sT, \
    behavioral_grid_state_with_objects_for_filtering_too_aggressive, \
    state_with_objects_for_filtering_almost_tracking_mode, state_with_objects_for_filtering_negative_sT, \
    state_with_objects_for_filtering_too_aggressive, follow_vehicle_recipes_towards_front_cells, follow_lane_recipes


# TODO: remove or revisit
def test_filter_followVehicleTooAggressive_filterResultsMatchExpected(
        behavioral_grid_state_with_objects_for_filtering_too_aggressive: BehavioralGridState,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    filtering = RecipeFiltering(filters=[FilterBadExpectedTrajectory('predicates')], logger=logger)

    # State leads to a0=0,v0=10,sT=53.5,vT=30
    # First three and last three are false because they're recipes of non-occupied cells
    # three middle results are false because this following scenario is too aggressive (close a gap of 20[m/s] to a target vehicle ~50[m] from us)
    # calm and standard actions take too much time and aggressive violates velocity and acceleration
    # All ground truths checked with desmos - https://www.desmos.com/calculator/8kybpq4tta
    expected_filter_results = np.array([False, False, False, False, False, False, False, False, False], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)
    filter_results = np.array(dynamic_action_space.filter_recipes(follow_vehicle_recipes_towards_front_cells,
                                                                  behavioral_grid_state_with_objects_for_filtering_too_aggressive))

    np.testing.assert_array_almost_equal(filter_results, expected_filter_results)



