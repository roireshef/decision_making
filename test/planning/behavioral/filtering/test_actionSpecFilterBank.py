import numpy as np
from typing import List
from unittest.mock import patch

from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, StaticActionRecipe
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterForKinematics, FilterIfNone, \
    FilterForSafetyTowardsTargetVehicle
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterBadExpectedTrajectory
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from rte.python.logger.AV_logger import AV_Logger

from decision_making.test.messages.static_scene_fixture import scene_static
from decision_making.test.planning.behavioral.behavioral_state_fixtures import \
    behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode, \
    state_with_objects_for_filtering_exact_tracking_mode, \
    behavioral_grid_state_with_objects_for_filtering_too_aggressive, \
    state_with_objects_for_filtering_almost_tracking_mode, \
    behavioral_grid_state_with_objects_for_filtering_exact_tracking_mode, \
    state_with_objects_for_filtering_too_aggressive, follow_vehicle_recipes_towards_front_cells, follow_lane_recipes


@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT', 5)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.SAFETY_HEADWAY', 0.7)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LON_ACC_LIMITS', np.array([-5.5, 3.0]))
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LAT_ACC_LIMITS', np.array([-4.0, 4.0]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_ACTION_T_LIMITS', np.array([0, 15]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_JERK_S_JERK_D_TIME_WEIGHTS', np.array([
    [12, 0.15, 0.1],
    [2, 0.15, 0.1],
    [0.01, 0.15, 0.1]
]))
def test_filter_closeToTrackingMode_allActionsAreValid(
        behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    """ see velocities and accelerations at https://www.desmos.com/calculator/betept6wyx """

    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    filtering = RecipeFiltering(filters=[], logger=logger)

    # only look at the same lane, front cell actions
    actions_with_vehicle = follow_vehicle_recipes_towards_front_cells[3:6]

    # All ground truths checked with desmos - https://www.desmos.com/calculator/8kybpq4tta
    expected_filter_results = np.array([True, True, True], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)

    action_specs_with_vehicle = dynamic_action_space.specify_goals(actions_with_vehicle,
                                                                   behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode)

    action_spec_filter = ActionSpecFiltering(filters=[FilterIfNone(), FilterForKinematics()], logger=logger)

    filter_results = action_spec_filter.filter_action_specs(action_specs_with_vehicle,
                                                            behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode)

    np.testing.assert_array_equal(filter_results, expected_filter_results)


@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT', 5)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.SAFETY_HEADWAY', 0.7)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LON_ACC_LIMITS', np.array([-5.5, 3.0]))
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LAT_ACC_LIMITS', np.array([-4.0, 4.0]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_ACTION_T_LIMITS', np.array([0, 15]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_JERK_S_JERK_D_TIME_WEIGHTS', np.array([
    [12, 0.15, 0.1],
    [2, 0.15, 0.1],
    [0.01, 0.15, 0.1]
]))
def test_filter_trackingMode_allActionsAreValid(
        behavioral_grid_state_with_objects_for_filtering_exact_tracking_mode,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    """ see velocities and accelerations at https://www.desmos.com/calculator/betept6wyx """

    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    filtering = RecipeFiltering(filters=[], logger=logger)

    # only look at the same lane, front cell actions
    actions_with_vehicle = follow_vehicle_recipes_towards_front_cells[3:6]

    # All ground truths checked with desmos - https://www.desmos.com/calculator/8kybpq4tta
    expected_filter_results = np.array([True, True, True], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)

    action_specs_with_vehicle = dynamic_action_space.specify_goals(actions_with_vehicle,
                                                                   behavioral_grid_state_with_objects_for_filtering_exact_tracking_mode)

    action_spec_filter = ActionSpecFiltering(filters=[FilterIfNone(), FilterForKinematics()], logger=logger)

    filter_results = action_spec_filter.filter_action_specs(action_specs_with_vehicle,
                                                            behavioral_grid_state_with_objects_for_filtering_exact_tracking_mode)

    np.testing.assert_array_equal(filter_results, expected_filter_results)


@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT', 5)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.SAFETY_HEADWAY', 0.7)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LON_ACC_LIMITS', np.array([-5.5, 3.0]))
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LAT_ACC_LIMITS', np.array([-4.0, 4.0]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_ACTION_T_LIMITS', np.array([0, 15]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_JERK_S_JERK_D_TIME_WEIGHTS', np.array([
    [12, 0.15, 0.1],
    [2, 0.15, 0.1],
    [0.01, 0.15, 0.1]
]))
def test_filter_followLane_filterResultsMatchExpected(
        behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode,
        follow_lane_recipes: List[StaticActionRecipe]):

    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    filtering = RecipeFiltering(filters=[], logger=logger)

    # All ground truths checked with desmos - https://www.desmos.com/calculator/8kybpq4tta
    # TODO: figure out the correct values here!
    expected_filter_results = np.array([False, True, True, True, True, True, True, True, True,
                                        False, True, False, False, False, False, False, False, False], dtype=bool)
    static_action_space = StaticActionSpace(logger, filtering=filtering)

    action_specs = static_action_space.specify_goals(follow_lane_recipes,
                                                     behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode)

    action_spec_filter = ActionSpecFiltering(filters=[FilterIfNone(), FilterForSafetyTowardsTargetVehicle()], logger=logger)

    filter_results = action_spec_filter.filter_action_specs(action_specs,
                                                            behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode)

    np.testing.assert_array_equal(filter_results, expected_filter_results)