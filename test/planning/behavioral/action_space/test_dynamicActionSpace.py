from logging import Logger
from typing import List

import numpy as np

from decision_making.src.global_constants import SPECIFICATION_MARGIN_TIME_DELAY, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.planning.types import FS_SX, FS_SV
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from mapping.src.service.map_service import MapService
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
    lane_latitudes = [(MapService().get_instance().get_lane_index(action.lane_id) + 0.5)*3.6 for action in actions]
    expected_latitudes = [1.8, 1.8, 1.8, 5.4, 5.4, 5.4, 9, 9, 9]
    latitudes = [action.d + lane_latitudes[i] for i, action in enumerate(actions)]
    np.testing.assert_array_almost_equal(latitudes, expected_latitudes)

    # terminal action-spec longitude equals the terminal longitude of target vehicle
    # (according to prediction at the terminal time)
    expected_longitudes = [target.dynamic_object.map_state.lane_fstate[FS_SX] +
                           target.dynamic_object.map_state.lane_fstate[FS_SV] * actions[i].t -
                           actions[i].v * SPECIFICATION_MARGIN_TIME_DELAY -
                           LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT -
                           behavioral_grid_state.ego_state.size.length / 2 - targets[i].dynamic_object.size.length / 2
                           for i, target in enumerate(targets)]
    longitudes = [action.s for action in actions]
    np.testing.assert_array_almost_equal(longitudes, expected_longitudes)
