from typing import List

import numpy as np

from decision_making.src.global_constants import SPECIFICATION_HEADWAY, LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, RelativeLane
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.planning.types import FS_SX, FS_SV
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.utils.map_utils import MapUtils
from rte.python.logger.AV_logger import AV_Logger

from decision_making.test.planning.behavioral.behavioral_state_fixtures import behavioral_grid_state, \
    follow_vehicle_recipes_towards_front_cells, state_with_surrounding_objects, route_plan_20_30
from decision_making.test.planning.custom_fixtures import turn_signal


# Specifies follow actions for front vehicles in 3 lanes. longitudinal and lateral coordinates
# of terminal states in action specification should be as expected.
# Multi-segment map is used, such that the targets have different road segment than ego.
def test_specifyGoals_stateWithSorroundingObjects_specifiesFollowTowardsFrontCellsWell(
        behavioral_grid_state: BehavioralGridState,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)

    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=DEFAULT_DYNAMIC_RECIPE_FILTERING)
    actions = dynamic_action_space.specify_goals(follow_vehicle_recipes_towards_front_cells, behavioral_grid_state)

    targets = [behavioral_grid_state.road_occupancy_grid[(recipe.relative_lane, recipe.relative_lon)][0]
               for recipe in follow_vehicle_recipes_towards_front_cells]

    # terminal action-spec latitude equals the current latitude of target vehicle
    expected_latitudes = [0]*9
    latitudes = [action.d for i, action in enumerate(actions)]
    np.testing.assert_array_almost_equal(latitudes, expected_latitudes)

    # since the map is multi-segment, calculate objects longitudes relative to the corresponding GFF
    ego_ordinal = MapUtils.get_lane_ordinal(behavioral_grid_state.ego_state.map_state.lane_id)
    objects_longitudes = []
    for i, target in enumerate(targets):
        target_map_state = target.dynamic_object.map_state
        target_ordinal = MapUtils.get_lane_ordinal(target_map_state.lane_id)
        rel_lane = RelativeLane(target_ordinal - ego_ordinal)
        target_gff_fstate = behavioral_grid_state.extended_lane_frames[rel_lane].convert_from_segment_state(
            target_map_state.lane_fstate, target_map_state.lane_id)
        objects_longitudes.append(target_gff_fstate[FS_SX])

    # terminal action-spec longitude equals the terminal longitude of target vehicle
    # (according to prediction at the terminal time)
    expected_longitudes = [objects_longitudes[i] +
                           target.dynamic_object.map_state.lane_fstate[FS_SV] * actions[i].t -
                           actions[i].v * SPECIFICATION_HEADWAY -
                           LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT -
                           behavioral_grid_state.ego_state.size.length / 2 - targets[i].dynamic_object.size.length / 2
                           for i, target in enumerate(targets)]

    longitudes = [action.s for action in actions]  # also relative to the corresponding GFF
    np.testing.assert_array_almost_equal(longitudes, expected_longitudes)
