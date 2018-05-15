from typing import List

import numpy as np

from decision_making.src.global_constants import SAFE_DIST_TIME_DELAY, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe
from decision_making.src.planning.types import FS_SX, FS_SV
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.test.planning.behavioral.behavioral_state_fixtures import behavioral_grid_state, \
    follow_recipes_towards_front_cells, state_with_sorrounding_objects, pg_map_api
from rte.python.logger.AV_logger import AV_Logger

import numpy as np

def test_specifyGoals_stateWithSorroundingObjects_specifiesFollowTowardsFrontCellsWell(behavioral_grid_state: BehavioralGridState,
                                                                                       follow_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    dynamic_action_space = DynamicActionSpace(logger, predictor)
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
