from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FS_SX
from decision_making.src.utils.map_utils import MapUtils
from rte.python.logger.AV_logger import AV_Logger

import numpy as np

from decision_making.test.planning.behavioral.behavioral_state_fixtures import behavioral_grid_state, \
    state_with_surrounding_objects, state_with_surrounding_objects_and_off_map_objects, route_plan_20_30
from decision_making.test.messages.scene_static_fixture import scene_static_short_testable

def test_createFromState_8objectsAroundEgo_correctGridSize(state_with_surrounding_objects, route_plan_20_30):
    """
    validate that 8 objects around ego create 8 grid cells in the behavioral state in multi-road map
    (a cell is created only if it contains at least one object)
    """
    logger = AV_Logger.get_logger()

    behavioral_state = BehavioralGridState.create_from_state(state_with_surrounding_objects, route_plan_20_30, logger)

    assert len(behavioral_state.road_occupancy_grid) == len(state_with_surrounding_objects.dynamic_objects)


def test_createFromState_eightObjectsAroundEgo_IgnoreThreeOffMapObjects(state_with_surrounding_objects_and_off_map_objects,
                                                                        route_plan_20_30):
    """
    Off map objects are located at ego's right lane.
    validate that 8 objects around ego create 5 grid cells in the behavioral state in multi-road map, while ignoring 3
    off map objects.
    (a cell is created only if it contains at least one on-map object, off map objects are marked with an off-map flag)
    """
    logger = AV_Logger.get_logger()
    behavioral_state = BehavioralGridState.create_from_state(
        state_with_surrounding_objects_and_off_map_objects, route_plan_20_30, logger)
    on_map_objects = [obj for obj in state_with_surrounding_objects_and_off_map_objects.dynamic_objects
                      if not obj.off_map]
    assert len(behavioral_state.road_occupancy_grid) == len(on_map_objects)

    for rel_lane, rel_lon in behavioral_state.road_occupancy_grid:
        assert rel_lane != RelativeLane.RIGHT_LANE
        dynamic_objects_on_grid = behavioral_state.road_occupancy_grid[(rel_lane, rel_lon)]
        assert np.all([not obj.dynamic_object.off_map for obj in dynamic_objects_on_grid])



def test_calculateLongitudinalDifferences_8objectsAroundEgo_accurate(state_with_surrounding_objects, behavioral_grid_state):
    """
    validate that 8 objects around ego have accurate longitudinal distances from ego in multi-road map
    """
    target_map_states = [obj.map_state for obj in state_with_surrounding_objects.dynamic_objects]

    longitudinal_distances = behavioral_grid_state.calculate_longitudinal_differences(target_map_states)

    for i, map_state in enumerate(target_map_states):
        ego_ordinal = MapUtils.get_lane_ordinal(behavioral_grid_state.ego_state.map_state.lane_id)
        target_ordinal = MapUtils.get_lane_ordinal(map_state.lane_id)
        rel_lane = RelativeLane(target_ordinal - ego_ordinal)
        target_gff_fstate = behavioral_grid_state.extended_lane_frames[rel_lane].convert_from_segment_state(
            map_state.lane_fstate, map_state.lane_id)
        assert longitudinal_distances[i] == target_gff_fstate[FS_SX] - behavioral_grid_state.projected_ego_fstates[rel_lane][FS_SX]
