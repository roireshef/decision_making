from logging import Logger
from typing import List

import time

from decision_making.src.global_constants import SAFE_DIST_TIME_DELAY, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, \
    FILTER_V_0_GRID, FILTER_A_0_GRID, FILTER_S_T_GRID, FILTER_V_T_GRID
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, AggressivenessLevel, ActionType, \
    StaticActionRecipe
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import FS_SX, FS_SV
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import ObjectSize, EgoState, State
from decision_making.test.planning.behavioral.behavioral_state_fixtures import behavioral_grid_state_with_objects_for_filtering_tracking_mode, \
behavioral_grid_state_with_objects_for_filtering_negative_sT, behavioral_grid_state_with_objects_for_filtering_too_aggressive,\
state_with_objects_for_filtering_tracking_mode, state_with_objects_for_filtering_negative_sT,\
    state_with_objects_for_filtering_too_aggressive, follow_vehicle_recipes_towards_front_cells, follow_lane_recipes, pg_map_api
from decision_making.test.planning.utils.optimal_control.quartic_poly_formulas import QuarticMotionPredicatesCreator
from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import QuinticMotionPredicatesCreator
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger

from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterBadExpectedTrajectory

import numpy as np


def test_filterBadExpectedTrajectory_follow_vehicle_tracking(behavioral_grid_state_with_objects_for_filtering_tracking_mode: BehavioralGridState,
                                                    follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    filtering = RecipeFiltering(filters=[FilterBadExpectedTrajectory('predicates')], logger=logger)

    # State leads to a0=0,v0=10,sT=15.5,vT=10.2
    # First three and last three are false because they're recipes of non-occupied cells
    # three middle results are true because tracking a vehicle whose velocity is very close to us can be done with multiple horizons
    # All ground truths checked with desmos
    expected_filter_results = np.array([False, False, False,  True,  True,  True, False, False, False], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)
    filter_results = np.array(dynamic_action_space.filter_recipes(follow_vehicle_recipes_towards_front_cells, behavioral_grid_state_with_objects_for_filtering_tracking_mode))

    np.testing.assert_array_almost_equal(filter_results, expected_filter_results)


def test_filterBadExpectedTrajectory_follow_vehicle_sT_negative(behavioral_grid_state_with_objects_for_filtering_negative_sT: BehavioralGridState,
                                                    follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    filtering = RecipeFiltering(filters=[FilterBadExpectedTrajectory('predicates')], logger=logger)

    # State leads to a0=0,v0=10,sT=-0.7,vT=11
    # First three and last three are false because they're recipes of non-occupied cells
    # three middle results are [False,True,True] because calm action takes too much time(>20s) and gets filtered while the others don't
    # take this much time and meets the velocity and acceleration constraints
    # All ground truths checked with desmos
    expected_filter_results = np.array([False, False, False, False,  True,  True, False, False, False], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)
    filter_results = np.array(dynamic_action_space.filter_recipes(follow_vehicle_recipes_towards_front_cells, behavioral_grid_state_with_objects_for_filtering_negative_sT))

    np.testing.assert_array_almost_equal(filter_results, expected_filter_results)


def test_filterBadExpectedTrajectory_follow_vehicle_too_aggressive(behavioral_grid_state_with_objects_for_filtering_too_aggressive: BehavioralGridState,
                                                    follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    filtering = RecipeFiltering(filters=[FilterBadExpectedTrajectory('predicates')], logger=logger)

    # State leads to a0=0,v0=10,sT=53.5,vT=30
    # First three and last three are false because they're recipes of non-occupied cells
    # three middle results are false because this following scenario is too aggressive (close a gap of 20[m/s] to a target vehicle 100[m] from us)
    # calm and standard actions take too much time and aggressive violates velocity and acceleration
    # All ground truths checked with desmos
    expected_filter_results = np.array([False, False, False, False, False, False, False, False, False], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)
    filter_results = np.array(dynamic_action_space.filter_recipes(follow_vehicle_recipes_towards_front_cells, behavioral_grid_state_with_objects_for_filtering_too_aggressive))

    np.testing.assert_array_almost_equal(filter_results, expected_filter_results)


def test_filterBadExpectedTrajectory_follow_lane(behavioral_grid_state_with_objects_for_filtering_tracking_mode: BehavioralGridState,
                                                    follow_lane_recipes: List[StaticActionRecipe]):
    logger = AV_Logger.get_logger()

    filtering = RecipeFiltering(filters=[FilterBadExpectedTrajectory('predicates')], logger=logger)

    expected_filter_results = np.array([False,  True,  True,  True,  True,  True,  True,  True,  True,
       False,  True, False, False, False, False, False, False, False], dtype=bool)

    static_action_space = StaticActionSpace(logger, filtering=filtering)
    filter_results = np.array(static_action_space.filter_recipes(follow_lane_recipes, behavioral_grid_state_with_objects_for_filtering_tracking_mode))

    # for i, stat_rec in enumerate(follow_lane_recipes):
    #     print('result for %s number %d is %s' % (stat_rec.__dict__, i, filter_results[i]))

    np.testing.assert_array_almost_equal(filter_results, expected_filter_results)
