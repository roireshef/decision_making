from logging import Logger
from typing import List

import numpy as np

from decision_making.src.global_constants import SAFE_DIST_TIME_DELAY, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.planning.types import FS_SX, FS_SV
from decision_making.src.planning.utils.map_utils import MapUtils
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from rte.python.logger.AV_logger import AV_Logger

from decision_making.test.planning.behavioral.behavioral_state_fixtures import behavioral_grid_state, \
    follow_vehicle_recipes_towards_front_cells, state_with_sorrounding_objects, pg_map_api


# specifies follow actions for front vehicles in 3 lanes. longitudinal and latitudinal coordinates
# of terminal states in action specification should be as expected
def test_specifyGoals_stateWithSorroundingObjects_specifiesFollowTowardsFrontCellsWell(
        behavioral_grid_state: BehavioralGridState,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=DEFAULT_DYNAMIC_RECIPE_FILTERING)
    actions = dynamic_action_space.specify_goals(follow_vehicle_recipes_towards_front_cells, behavioral_grid_state)

    targets = [behavioral_grid_state.road_occupancy_grid[(recipe.relative_lane, recipe.relative_lon)][0]
               for recipe in follow_vehicle_recipes_towards_front_cells]

    # terminal action-spec latitude equals the current latitude of target vehicle
    expected_latitudes = [1.8, 1.8, 1.8, 5.4, 5.4, 5.4, 9, 9, 9]
    latitudes = [action.d for action in actions]
    np.testing.assert_array_almost_equal(latitudes, expected_latitudes)

    # terminal action-spec longitude equals the terminal longitude of target vehicle
    # (according to prediction at the terminal time)
    expected_longitudes = [target.fstate[FS_SX] + target.fstate[FS_SV] * actions[i].t -
                           actions[i].v * SAFE_DIST_TIME_DELAY -
                           LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT -
                           behavioral_grid_state.ego_state.size.length / 2 - targets[i].dynamic_object.size.length / 2
                           for i, target in enumerate(targets)]
    longitudes = [action.s for action in actions]
    np.testing.assert_array_almost_equal(longitudes, expected_longitudes)





def test_specifyGoal_atLeastOneActionShouldBeFound():
    """
    Object and ego have the same velocity 14 m/s (~50 km/h), the initial distance 49 m.
    At least one dynamic action should be created.
    :return:
    """
    logger = Logger("test_specifyGoal_dynamic")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    length = 4
    size = ObjectSize(length, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)
    road_frenet = MapUtils.get_road_rhs_frenet_by_road_id(road_id)

    ego_vel = 14
    F_vel = 14
    dist = 45

    ego = MapUtils.create_canonic_ego(0, ego_lon, lane_width / 2, ego_vel, size, road_frenet)
    F_lon = ego_lon + dist
    F = MapUtils.create_canonic_object(1, 0, F_lon, lane_width / 2, F_vel, size, road_frenet)

    state = State(None, [F], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)
    recipes = action_space.recipes
    recipes_mask = action_space.filter_recipes(recipes, behavioral_state)

    # Action specification
    specs = np.full(recipes.__len__(), None)
    valid_action_recipes = [recipe for i, recipe in enumerate(recipes) if recipes_mask[i]]
    specs[recipes_mask] = action_space.specify_goals(valid_action_recipes, behavioral_state)

    specs_mask = action_spec_validator.filter_action_specs(specs, behavioral_state)
    valid_idxs = np.where(specs_mask)[0]
    assert len(valid_idxs) > 0
