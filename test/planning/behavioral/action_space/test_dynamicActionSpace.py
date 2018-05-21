from logging import Logger
from typing import List

import numpy as np

from decision_making.src.global_constants import SAFE_DIST_TIME_DELAY, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import FS_SX, FS_SV
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import ObjectSize, EgoState, State
from decision_making.test.planning.behavioral.behavioral_state_fixtures import behavioral_grid_state, \
    follow_recipes_towards_front_cells, state_with_sorrounding_objects, pg_map_api
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger

from decision_making.src.planning.behavioral.constants import DEFAULT_DYNAMIC_RECIPE_FILTERING

import numpy as np

def test_specifyGoals_stateWithSorroundingObjects_specifiesFollowTowardsFrontCellsWell(behavioral_grid_state: BehavioralGridState,
                                                                                       follow_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=DEFAULT_DYNAMIC_RECIPE_FILTERING)
    actions = dynamic_action_space.specify_goals(follow_recipes_towards_front_cells, behavioral_grid_state)

    targets = [behavioral_grid_state.road_occupancy_grid[(recipe.relative_lane, recipe.relative_lon)][0]
                          for recipe in follow_recipes_towards_front_cells]

    # terminal action-spec latitude equals the current latitude of target vehicle
    expected_latitudes = [1.8, 1.8, 1.8, 5.4, 5.4, 5.4, 9, 9, 9]
    latitudes = [action.d for action in actions]
    np.testing.assert_array_almost_equal(latitudes, expected_latitudes)

    # terminal action-spac longitude equals the terminal longitude of target vehicle
    # (according to prediction at the terminal time)
    # expected_longitudes = [predictor.predict_object_on_road(obj, [actions[i].t])[0].road_localization.road_lon -
    expected_longitudes = [target.fstate[FS_SX] + target.fstate[FS_SV] * actions[i].t -
                           actions[i].v * SAFE_DIST_TIME_DELAY -
                           LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT -
                           behavioral_grid_state.ego_state.size.length/2 - targets[i].dynamic_object.size.length/2
                           for i, target in enumerate(targets)]
    longitudes = [action.s for action in actions]
    np.testing.assert_array_almost_equal(longitudes, expected_longitudes)


# test specify for dynamic action from a slightly unsafe position:
# when the distance from the target is just 2 seconds * target velocity, without adding the cars' sizes
def test_specifyGoal_slightlyUnsafeState_shouldSucceed():
    logger = Logger("test_specifyDynamicAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = DynamicActionSpace(logger, predictor, filtering=DEFAULT_DYNAMIC_RECIPE_FILTERING)

    # verify the peak acceleration does not exceed the limit by calculating the average acceleration from 0 to 50 km/h
    ego_vel = 10
    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat - lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

    obj_vel = 10
    obj_lon = ego_lon + 20
    obj_cpoint, obj_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, obj_lon, road_mid_lat - lane_width)
    obj = EgoState(0, 0, obj_cpoint[0], obj_cpoint[1], obj_cpoint[2], obj_yaw, size, 0, obj_vel, 0, 0, 0, 0)

    state = State(None, [obj], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    action_recipes = action_space.recipes
    recipes_mask = action_space.filter_recipes(action_recipes, behavioral_state)

    front_recipies = [recipe for i, recipe in enumerate(action_space.recipes)
                      if recipe.relative_lane == RelativeLane.SAME_LANE and
                      recipe.relative_lon == RelativeLongitudinalPosition.FRONT and
                      recipes_mask[i]]

    # verify that there is at least one valid recipe for dynamic actions
    assert len(front_recipies) > 0
