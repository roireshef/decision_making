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
from decision_making.test.planning.behavioral.behavioral_state_fixtures import behavioral_grid_state, \
    follow_vehicle_recipes_towards_front_cells, follow_lane_recipes, state_with_sorrounding_objects, pg_map_api
from decision_making.test.planning.utils.optimal_control.quartic_poly_formulas import QuarticMotionPredicatesCreator
from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import QuinticMotionPredicatesCreator
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger

from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterBadExpectedTrajectory

import numpy as np


def test_filterBadExpectedTrajectory_follow_vehicle(behavioral_grid_state: BehavioralGridState,
                                                    follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    quintic_predicates_creator = QuinticMotionPredicatesCreator(FILTER_V_0_GRID, FILTER_A_0_GRID, FILTER_S_T_GRID,
                                                                FILTER_V_T_GRID, T_m=SAFE_DIST_TIME_DELAY,
                                                                predicates_resources_target_directory='predicates')
    res = quintic_predicates_creator.generate_predicate_value(ActionType.FOLLOW_VEHICLE,
                                                              w_T=0.1, w_J=12, a_0=0, v_0=10, s_T=15, v_T=10, T_m=2)
    filtering = RecipeFiltering(filters=[FilterBadExpectedTrajectory('predicates')], logger=logger)

    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)
    filtered_recipes = dynamic_action_space.filter_recipes(follow_vehicle_recipes_towards_front_cells, behavioral_grid_state)

    targets = [behavioral_grid_state.road_occupancy_grid[(recipe.relative_lane, recipe.relative_lon)][0]
               for recipe in follow_vehicle_recipes_towards_front_cells]

    # np.testing.assert_array_almost_equal(filter_results, expected_filter_results)


def test_filterBadExpectedTrajectory_follow_lane(behavioral_grid_state: BehavioralGridState,
                                                    follow_lane_recipes: List[StaticActionRecipe]):
    logger = AV_Logger.get_logger()

    quartic_predicates_creator = QuarticMotionPredicatesCreator(FILTER_V_0_GRID, FILTER_A_0_GRID, FILTER_V_T_GRID,
                                                                predicates_resources_target_directory='predicates')
    res = quartic_predicates_creator.generate_predicate_value(w_T=0.1, w_J=12, a_0=0, v_0=10, v_T=10)
    filtering = RecipeFiltering(filters=[FilterBadExpectedTrajectory('predicates')], logger=logger)

    static_action_space = StaticActionSpace(logger, filtering=filtering)
    filtered_recipes = static_action_space.filter_recipes(follow_lane_recipes, behavioral_grid_state)

    # np.testing.assert_array_almost_equal(filter_results, expected_filter_results)
