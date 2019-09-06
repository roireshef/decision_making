from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FS_SX
from decision_making.src.utils.map_utils import MapUtils
from rte.python.logger.AV_logger import AV_Logger
import numpy as np
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.state.state import DynamicObject, MapState, ObjectSize
from decision_making.test.planning.behavioral.behavioral_state_fixtures import behavioral_grid_state, \
    state_with_sorrounding_objects, state_with_surrounding_objects_and_off_map_objects, route_plan_20_30
from decision_making.test.messages.scene_static_fixture import scene_static_oval_with_splits, \
    scene_static_short_testable


def test_createFromState_8objectsAroundEgo_correctGridSize(state_with_sorrounding_objects, route_plan_20_30):
    """
    validate that 8 objects around ego create 8 grid cells in the behavioral state in multi-road map
    (a cell is created only if it contains at least one object)
    """
    logger = AV_Logger.get_logger()

    behavioral_state = BehavioralGridState.create_from_state(state_with_sorrounding_objects, route_plan_20_30, logger)

    assert len(behavioral_state.road_occupancy_grid) == len(state_with_sorrounding_objects.dynamic_objects)


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


def test_calculateLongitudinalDifferences_8objectsAroundEgo_accurate(state_with_sorrounding_objects, behavioral_grid_state):
    """
    validate that 8 objects around ego have accurate longitudinal distances from ego in multi-road map
    """
    target_map_states = [obj.map_state for obj in state_with_sorrounding_objects.dynamic_objects]

    longitudinal_distances = behavioral_grid_state.calculate_longitudinal_differences(target_map_states)

    for i, map_state in enumerate(target_map_states):
        ego_ordinal = MapUtils.get_lane_ordinal(behavioral_grid_state.ego_state.map_state.lane_id)
        target_ordinal = MapUtils.get_lane_ordinal(map_state.lane_id)
        rel_lane = RelativeLane(target_ordinal - ego_ordinal)
        target_gff_fstate = behavioral_grid_state.extended_lane_frames[rel_lane].convert_from_segment_state(
            map_state.lane_fstate, map_state.lane_id)
        assert longitudinal_distances[i] == target_gff_fstate[FS_SX] - behavioral_grid_state.projected_ego_fstates[rel_lane][FS_SX]


def test_projectActorsInsideIntersection_laneSplit_carInOverlap(scene_static_oval_with_splits: SceneStatic):
    """
    Validate that projected object is correctly placed in overlapping lane
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_oval_with_splits)

    # Create other car in lane 21, which overlaps with lane 22
    dyn_obj = DynamicObject.create_from_map_state(obj_id=10, timestamp=5,
                                                  map_state=MapState(np.array([1,1,0,0,0,0]), 19670532),
                                                  size=ObjectSize(5, 2, 2), confidence=1, off_map=False)
    all_objects = BehavioralGridState._project_actors_inside_intersection([dyn_obj])
    projected_objects = [obj for obj in all_objects if obj.obj_id == -10]
    assert projected_objects[0].map_state.lane_id == 19670533


def test_projectActorsInsideIntersection_laneMerge_carInOverlap(scene_static_oval_with_splits: SceneStatic):
    """
    Validate that projected object is correctly placed in overlapping lane
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_oval_with_splits)

    overlaps = {}
    geo_lane_ids = [lane.e_i_lane_segment_id  for lane in scene_static_oval_with_splits.s_Data.s_SceneStaticGeometry.as_scene_lane_segments]
    for lane in scene_static_oval_with_splits.s_Data.s_SceneStaticBase.as_scene_lane_segments:
        if lane.e_Cnt_lane_overlap_count > 0 and lane.e_i_lane_segment_id in geo_lane_ids:
            for overlap in lane.as_lane_overlaps:
                overlaps[lane.e_i_lane_segment_id] = (
                overlap.e_i_other_lane_segment_id, overlap.e_e_lane_overlap_type)

    # Create other car in lane 21, which overlaps with lane 22
    dyn_obj = DynamicObject.create_from_map_state(obj_id=10, timestamp=5,
                                                  map_state=MapState(np.array([5, 1, 0, 0, 0, 0]), 58375684),
                                                  size=ObjectSize(5, 2, 2), confidence=1, off_map=False)
    all_objects = BehavioralGridState._project_actors_inside_intersection([dyn_obj])
    projected_objects = [obj for obj in all_objects if obj.obj_id == -10]
    assert projected_objects[0].map_state.lane_id == 58375685


def test_createMirrorObjects_laneSplit_carNotInOverlap(scene_static_short_testable: SceneStatic):
    """
    Validate that no mirror object is created if there is no overlap
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_short_testable)
    # Create other car in lane 21 which does NOT overlap with any other lane
    dyn_obj = DynamicObject.create_from_map_state(obj_id=10, timestamp=5, map_state=MapState(np.array([1,1,0,0,0,0]), 21),
                                                  size=ObjectSize(1,1,1), confidence=1, off_map=False)
    all_objects = BehavioralGridState._project_actors_inside_intersection([dyn_obj])
    assert len(all_objects) == 1 and all_objects[0] == dyn_obj

def test_isObjectInLane_carOnLaneLine(scene_static_short_testable: SceneStatic):
    """
    Tests the method is_object_in_lane. Places a car in lane 11, shifted 1.5 meters left from lane 11's nominal points.
    :param scene_static_short_testable:
    :return:
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_short_testable)

    # Create car in lane 11 which is offset 1.5 meters to the left
    dyn_obj = DynamicObject.create_from_map_state(obj_id=10, timestamp=5, map_state=MapState(np.array([1.0,1,0,1.5,0,0]), 11),
                                                  size=ObjectSize(4,1.5,1), confidence=1, off_map=False)

    assert BehavioralGridState.is_object_in_lane(dyn_obj, 11) == True
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 12) == True
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 10) == False

def test_isObjectInLane_carInSingleLane(scene_static_short_testable: SceneStatic):

    SceneStaticModel.get_instance().set_scene_static(scene_static_short_testable)

    # Create car in lane 11
    dyn_obj = DynamicObject.create_from_map_state(obj_id=10, timestamp=5, map_state=MapState(np.array([1.0,1,0,0,0,0]), 11),
                                                  size=ObjectSize(1,1,1), confidence=1, off_map=False)

    assert BehavioralGridState.is_object_in_lane(dyn_obj, 11) == True
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 12) == False
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 10) == False

def test_isObjectInLane_laneSplit_carInOverlap(scene_static_oval_with_splits: SceneStatic):
    """
    Validate that projected object is correctly placed in overlapping lane
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_oval_with_splits)

    # Create other car in lane 21, which overlaps with lane 22
    dyn_obj = DynamicObject.create_from_map_state(obj_id=10, timestamp=5,
                                                  map_state=MapState(np.array([1,1,0,0,0,0]), 19670532),
                                                  size=ObjectSize(5, 2, 2), confidence=1, off_map=False)
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 19670532) == True
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 19670533) == True


