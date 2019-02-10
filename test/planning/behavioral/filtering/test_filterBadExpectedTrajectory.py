from logging import Logger
from typing import List

import time

from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, StaticActionRecipe
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.test.planning.behavioral.behavioral_state_fixtures import \
    behavioral_grid_state_with_objects_for_filtering_tracking_mode, \
    behavioral_grid_state_with_objects_for_filtering_negative_sT, \
    behavioral_grid_state_with_objects_for_filtering_too_aggressive, \
    state_with_objects_for_filtering_tracking_mode, state_with_objects_for_filtering_negative_sT, \
    state_with_objects_for_filtering_too_aggressive, follow_vehicle_recipes_towards_front_cells, follow_lane_recipes
from rte.python.logger.AV_logger import AV_Logger

from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterLimitsViolatingTrajectory

import numpy as np

from decision_making.test.messages.static_scene_fixture import scene_static

def test_filter_followVehicleTracking_filterResultsMatchExpected(
        behavioral_grid_state_with_objects_for_filtering_tracking_mode: BehavioralGridState,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()

    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    filtering = RecipeFiltering(filters=[FilterLimitsViolatingTrajectory('predicates')], logger=logger)

    # State leads to a0=0,v0=10,sT=15.5,vT=10.2
    # First three and last three are false because they're recipes of non-occupied cells
    # three middle results are true because tracking a vehicle whose velocity is very close to us can be done with
    # multiple horizons
    # All ground truths checked with desmos - https://www.desmos.com/calculator/8kybpq4tta
    expected_filter_results = np.array([False, False, False, True, True, True, False, False, False], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)
    filter_results = np.array(dynamic_action_space.filter_recipes(follow_vehicle_recipes_towards_front_cells,
                                                                  behavioral_grid_state_with_objects_for_filtering_tracking_mode))

    np.testing.assert_array_almost_equal(filter_results, expected_filter_results)


def test_filter_followVehicleSTNegative_filterResultsMatchExpected(
        behavioral_grid_state_with_objects_for_filtering_negative_sT: BehavioralGridState,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):

    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    filtering = RecipeFiltering(filters=[FilterLimitsViolatingTrajectory('predicates')], logger=logger)

    # State leads to a0=0,v0=10,sT=-0.7,vT=11
    # First three and last three are false because they're recipes of non-occupied cells;
    # three middle results are True because they meet the time, velocity and acceleration constraints.
    # All ground truths checked with desmos - https://www.desmos.com/calculator/8kybpq4tta
    expected_filter_results = np.array([False, False, False, True, True, True, False, False, False], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)
    filter_results = np.array(dynamic_action_space.filter_recipes(follow_vehicle_recipes_towards_front_cells,
                                                                  behavioral_grid_state_with_objects_for_filtering_negative_sT))

    np.testing.assert_array_almost_equal(filter_results, expected_filter_results)


def test_filter_followVehicleTooAggressive_filterResultsMatchExpected(
        behavioral_grid_state_with_objects_for_filtering_too_aggressive: BehavioralGridState,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    filtering = RecipeFiltering(filters=[FilterLimitsViolatingTrajectory('predicates')], logger=logger)

    # State leads to a0=0,v0=10,sT=53.5,vT=30
    # First three and last three are false because they're recipes of non-occupied cells
    # three middle results are false because this following scenario is too aggressive (close a gap of 20[m/s]
    # to a target vehicle ~50[m] from us)
    # calm and standard actions take too much time and aggressive violates velocity and acceleration
    # All ground truths checked with desmos - https://www.desmos.com/calculator/8kybpq4tta
    expected_filter_results = np.array([False, False, False, False, False, False, False, False, False], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)
    filter_results = np.array(dynamic_action_space.filter_recipes(follow_vehicle_recipes_towards_front_cells,
                                                                  behavioral_grid_state_with_objects_for_filtering_too_aggressive))

    np.testing.assert_array_almost_equal(filter_results, expected_filter_results)


def test_filter_followLane_filterResultsMatchExpected(
        behavioral_grid_state_with_objects_for_filtering_tracking_mode: BehavioralGridState,
        follow_lane_recipes: List[StaticActionRecipe]):
    logger = AV_Logger.get_logger()

    filtering = RecipeFiltering(filters=[FilterLimitsViolatingTrajectory('predicates')], logger=logger)

    # State leads to a0=0,v0=10,vT= 0:6:30
    # Each three consequent results are of the same velocity and three aggressiveness levels (calm, standard, aggressive)
    # First one is filtered because calming into a stop takes too much time, then all actions meet constraints and for
    # velocity of 18 [m/s] all actions are valid.
    # For acceleration to 24 m/s the calm action takes more than 20 sec and then is filtered.
    # For acceleration to 30 m/s only aggressive action takes less than 20 sec. The rest actions are filtered.
    # All ground truths checked with desmos - https://www.desmos.com/calculator/usk7djcttx

    expected_filter_results = np.array([False, True, True, True, True, True, True, True, True,
                                        True, True, True, False, True, True, False, False, True], dtype=bool)

    static_action_space = StaticActionSpace(logger, filtering=filtering)
    filter_results = np.array(static_action_space.filter_recipes(follow_lane_recipes,
                                                                 behavioral_grid_state_with_objects_for_filtering_tracking_mode))

    # for i, stat_rec in enumerate(follow_lane_recipes):
    #     print('result for %s number %d is %s' % (stat_rec.__dict__, i, filter_results[i]))

    np.testing.assert_array_almost_equal(filter_results, expected_filter_results)
