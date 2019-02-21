from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
from decision_making.src.planning.types import FS_SX
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import DynamicObject
from decision_making.src.utils.map_utils import MapUtils
from itertools import product

from decision_making.test.planning.behavioral.behavioral_state_fixtures import behavioral_grid_state, \
    state_with_sorrounding_objects,intersection_state_with_sorrounding_objects,intersection_behavioral_grid_state,\
    NAVIGATION_PLAN
from rte.python.logger.AV_logger import AV_Logger
import numpy as np


def test_createFromState_8objectsAroundEgo_correctGridSize(state_with_sorrounding_objects):
    """
    validate that 8 objects around ego create 8 grid cells in the behavioral state in multi-road map
    (a cell is created only if it contains at least one object)
    """
    logger = AV_Logger.get_logger()

    behavioral_state = BehavioralGridState.create_from_state(state_with_sorrounding_objects, NAVIGATION_PLAN, logger)

    assert len(behavioral_state.road_occupancy_grid) == len(state_with_sorrounding_objects.dynamic_objects)


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


def test_createMirrorObjects_createsCorrectNumberOfPseudoObjectsNoProjections(state_with_sorrounding_objects):
    """
    Tests that all vehicles on intersections are being generated
    """
    overloaded_objects = BehavioralGridState._create_mirror_objects(dynamic_objects=state_with_sorrounding_objects.dynamic_objects)
    vehicles_on_intersection = []
    assert len(overloaded_objects) == len(state_with_sorrounding_objects.dynamic_objects) + len(vehicles_on_intersection)



def test_createMirrorObjects_createsCorrectPseudoObjects(intersection_state_with_sorrounding_objects):
    """
    Tests that all vehicles on intersections are being generated correctly
    """
    overloaded_objects = BehavioralGridState._create_mirror_objects(dynamic_objects=intersection_state_with_sorrounding_objects.dynamic_objects)
    object_on_210 = intersection_state_with_sorrounding_objects.dynamic_objects [2]
    object_on_211 = intersection_state_with_sorrounding_objects.dynamic_objects [4]

    vehicles_on_intersection = [DynamicObject(obj_id=object_on_210.obj_id, timestamp= object_on_210.timestamp,
                                              cartesian_state=object_on_210.cartesian_state, map_state=MapState(None, 211),
                                              map_state_on_host_lane=object_on_210.map_state,
                                              size=object_on_210.size, confidence=object_on_210.confidence),
                                DynamicObject(obj_id=object_on_211.obj_id, timestamp= object_on_211.timestamp,
                                              cartesian_state=object_on_210.cartesian_state, map_state=MapState(None, 210),
                                              map_state_on_host_lane=object_on_211.map_state,
                                              size=object_on_211.size, confidence=object_on_211.confidence)]

    for i in range(len(intersection_state_with_sorrounding_objects.dynamic_objects), len(overloaded_objects)):
        assert overloaded_objects[i].map_state.lane_fstate is None
        assert overloaded_objects[i].map_state.lane_id in [210, 211]


def test_lazySetMapStates_calculateCorrectFstate(intersection_state_with_sorrounding_objects):
    overloaded_objects = BehavioralGridState._create_mirror_objects(dynamic_objects=intersection_state_with_sorrounding_objects.dynamic_objects)
    object_on_210 = intersection_state_with_sorrounding_objects.dynamic_objects[2]
    object_on_211 = intersection_state_with_sorrounding_objects.dynamic_objects[4]

    for i in range(len(intersection_state_with_sorrounding_objects.dynamic_objects), len(overloaded_objects)):
        assert overloaded_objects[i].map_state.lane_fstate is None

    nav_plan = NavigationPlanMsg(np.array([20, 21, 22]))
    extended_lane_frames = BehavioralGridState._create_generalized_frenet_frames(intersection_state_with_sorrounding_objects,
                                                                                 nav_plan)
    objects_segment_ids = np.array([overloaded_object.map_state.lane_id for overloaded_object in overloaded_objects])
    # for objects on non-adjacent lane set relative_lanes[i] = None
    rel_lanes_per_obj = np.full(len(overloaded_objects), None)
    # calculate relative to ego lane (RIGHT, SAME, LEFT) for every object
    for rel_lane, extended_lane_frame in extended_lane_frames.items():
        # find all dynamic objects that belong to the current unified frame
        relevant_objects = extended_lane_frame.has_segment_ids(objects_segment_ids)
        rel_lanes_per_obj[relevant_objects] = rel_lane

    # RIGHT_LANE =  200, 210, 220, 230, 240
    # SAME_LANE  =  201, 211, 221, 231, 241
    # LEFT_LANE  =  202, 212, 222, 232, 242
    BehavioralGridState._lazy_set_map_states(overloaded_objects, extended_lane_frames, rel_lanes_per_obj)
    _, fstate210_on_211 = extended_lane_frames[RelativeLane.SAME_LANE].convert_to_segment_state(
        extended_lane_frames[RelativeLane.SAME_LANE].cstate_to_fstate(object_on_210.cartesian_state))
    _, fstate211_on_210 = extended_lane_frames[RelativeLane.RIGHT_LANE].convert_to_segment_state(
        extended_lane_frames[RelativeLane.RIGHT_LANE].cstate_to_fstate(object_on_211.cartesian_state))

    # TODO: Remove assumption of end of array
    for i in range(len(intersection_state_with_sorrounding_objects.dynamic_objects), len(overloaded_objects)):
        if overloaded_objects[i].map_state.lane_id == 210:
            assert np.array_equal(overloaded_objects[i].map_state.lane_fstate, fstate211_on_210)
        elif overloaded_objects[i].map_state.lane_id == 211:
            assert np.array_equal(overloaded_objects[i].map_state.lane_fstate, fstate210_on_211)


def test_addRoadSemantics_addedCorrectLongitudinalDistance(intersection_state_with_sorrounding_objects, intersection_behavioral_grid_state):
    """
    validate that 8 objects around ego have accurate longitudinal distances from ego in multi-road map
    """
    overloaded_objects = BehavioralGridState._create_mirror_objects(dynamic_objects=
                                                                       intersection_state_with_sorrounding_objects.dynamic_objects)
    objects_segment_ids = np.array([overloaded_object.map_state.lane_id for overloaded_object in overloaded_objects])
    # for objects on non-adjacent lane set relative_lanes[i] = None
    rel_lanes_per_obj = np.full(len(overloaded_objects), None)
    extended_lane_frames = intersection_behavioral_grid_state.extended_lane_frames
    # calculate relative to ego lane (RIGHT, SAME, LEFT) for every object
    for rel_lane, extended_lane_frame in extended_lane_frames.items():
        # find all dynamic objects that belong to the current unified frame
        relevant_objects = extended_lane_frame.has_segment_ids(objects_segment_ids)
        rel_lanes_per_obj[relevant_objects] = rel_lane

    # RIGHT_LANE =  200, 210, 220, 230, 240
    # SAME_LANE  =  201, 211, 221, 231, 241
    # LEFT_LANE  =  202, 212, 222, 232, 242
    BehavioralGridState._lazy_set_map_states(overloaded_objects, extended_lane_frames, rel_lanes_per_obj)
    target_map_states = [obj.map_state for obj in overloaded_objects]
    longitudinal_distances = intersection_behavioral_grid_state.calculate_longitudinal_differences(target_map_states)

    for i, map_state in enumerate(target_map_states):
        ego_ordinal = MapUtils.get_lane_ordinal(intersection_behavioral_grid_state.ego_state.map_state.lane_id)
        target_ordinal = MapUtils.get_lane_ordinal(map_state.lane_id)
        rel_lane = RelativeLane(target_ordinal - ego_ordinal)
        target_gff_fstate = intersection_behavioral_grid_state.extended_lane_frames[rel_lane].convert_from_segment_state(
            map_state.lane_fstate, map_state.lane_id)
        assert longitudinal_distances[i] == target_gff_fstate[FS_SX] - intersection_behavioral_grid_state.projected_ego_fstates[rel_lane][FS_SX]



def test_overloadDynamicObject_createsCorrectNumberOfPseudoObjects(intersection_state_with_sorrounding_objects):
    """
    Tests that all vehicles on intersections are being generated
    """
    overloaded_objects = BehavioralGridState._create_mirror_objects(dynamic_objects=intersection_state_with_sorrounding_objects.dynamic_objects)
    vehicles_on_intersection = ['210 projected on 211', '211 projected on 210' ]
    assert len(overloaded_objects) == len(intersection_state_with_sorrounding_objects.dynamic_objects) + len(vehicles_on_intersection)



def test_addRoadSemantics_addedCorrectRelativeLane(intersection_state_with_sorrounding_objects, intersection_behavioral_grid_state):
    """
    Tests if the correct road semantic has been added. Should pass whenever test_lazySetMapStates_accurateFstate passes
    :param intersection_state_with_sorrounding_objects:
    :param behavioral_grid_state:
    :return:
    """
    dynamic_objects = intersection_state_with_sorrounding_objects.dynamic_objects
    nav_plan = NavigationPlanMsg(np.array([20, 21, 22]))
    extended_lane_frames = BehavioralGridState._create_generalized_frenet_frames(intersection_state_with_sorrounding_objects,
                                                                                 nav_plan)

    projected_ego_fstates = intersection_behavioral_grid_state.projected_ego_fstates

    dynamic_objects_with_road_semantics = BehavioralGridState._add_road_semantics(dynamic_objects=dynamic_objects,
                                                                                  extended_lane_frames=extended_lane_frames,
                                                                                  projected_ego_fstates=projected_ego_fstates)
    expected_relative_lanes = [RelativeLane.RIGHT_LANE] * 3 + [RelativeLane.SAME_LANE] * 2 + [RelativeLane.LEFT_LANE] *3
    expected_relative_lanes += [RelativeLane.SAME_LANE, RelativeLane.RIGHT_LANE]
    assert len(expected_relative_lanes) == len(dynamic_objects_with_road_semantics)
    for i in range(len(dynamic_objects_with_road_semantics)):
        assert dynamic_objects_with_road_semantics[i].relative_lane == expected_relative_lanes[i]



def test_createFromState_correctProjectionOfIntersectionVehicles(intersection_state_with_sorrounding_objects):
    """
    Test if the correct cells are being assigned.
    Expected the following dynamic_object ids on the grid
    ###############################
    # (8)  #  (5,~3)   #  (3,~5)  #
    ###############################
    # (7)  #   EGO     #    (2)   #
    ###############################
    #  (6) #    (4)    #    (1)   #
    ###############################
    """
    nav_plan = NavigationPlanMsg(np.array([20, 21, 22]))
    behavioral_grid_state = BehavioralGridState.create_from_state(intersection_state_with_sorrounding_objects, nav_plan, None)

    expected_objects_on_grid = {(RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.PARALLEL): [2],
                                (RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.FRONT): [3, -5],
                                (RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.REAR): [1],
                                (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT): [5, -3],
                                (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.REAR): [4],
                                (RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.REAR): [6],
                                (RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.PARALLEL): [7],
                                (RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.FRONT): [8]
                                }

    for cell, expected_objects_on_cell in expected_objects_on_grid.items():
        assert set([obj.dynamic_object.obj_id for obj in behavioral_grid_state.road_occupancy_grid[cell]])\
               == set(expected_objects_on_cell)


