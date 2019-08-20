import pickle
from decision_making.paths import Paths
from decision_making.src.planning.types import FS_SA, C_A

from decision_making.src.messages.scene_static_message import StaticTrafficFlowControl, RoadObjectType
from decision_making.src.messages.route_plan_message import RoutePlan, DataRoutePlan, RoutePlanLaneSegment
from decision_making.src.messages.scene_common_messages import Header, Timestamp
from typing import List, Dict, Tuple
from decision_making.src.planning.types import RoadSegmentID, LaneOccupancyCost, LaneEndCost

import numpy as np
import pytest

from decision_making.src.global_constants import EPS, LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT, SPECIFICATION_HEADWAY
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, ActionType, AggressivenessLevel, \
    StaticActionRecipe
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import OccupancyState, State, ObjectSize, EgoState, DynamicObject
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.messages.scene_static_fixture import scene_static_pg_split, \
    scene_static_accel_towards_vehicle, scene_dynamic_accel_towards_vehicle, scene_static_left_lane_ends, scene_static_right_lane_ends, \
    right_lane_split_scene_static, left_lane_split_scene_static, left_right_lane_split_scene_static, \
    scene_static_lane_split_on_right_ends, scene_static_lane_split_on_left_ends, scene_static_lane_splits_on_left_and_right_end, \
    scene_static_lane_splits_on_left_and_right_offset
from decision_making.test.planning.route.scene_fixtures import default_route_plan_for_PG_split_file
from decision_making.test.planning.custom_fixtures import route_plan_1_2


EGO_LANE_LON = 120.  # ~2 meters behind end of a lane segment
NAVIGATION_PLAN = np.array(range(20, 30))

NAVIGATION_PLAN_OVAL_TRACK = np.array([3537, 76406, 3646, 46577, 46613, 87759, 8766, 76838, 228030,
                                       51360, 228028, 87622, 228007, 87660, 87744, 9893, 9894, 87740,
                                       77398, 87741, 25969, 10068, 87211, 10320, 10322, 228029, 87739,
                                       40953, 10073, 10066, 87732, 43516, 87770, 228034, 87996,
                                       228037, 10536, 88088, 228039, 88192, 10519, 10432, 3537])

@pytest.fixture(scope='function')
def route_plan_20_30():
    yield RoutePlan(s_Header=Header(e_Cnt_SeqNum=1, s_Timestamp=Timestamp(0, 0), e_Cnt_version=1),
                    s_Data=default_route_plan_for_PG_split_file())

@pytest.fixture(scope='function')
def route_plan_lane_split_on_right_ends(route_plan_20_30):
    # Delete all right lanes, except for lanes 220 and 230
    for index in [0, 1, 4, 5, 6, 7, 8, 9]:
        route_plan_20_30.s_Data.a_Cnt_num_lane_segments[index] -= 1
        del route_plan_20_30.s_Data.as_route_plan_lane_segments[index][0]

    yield route_plan_20_30

@pytest.fixture(scope='function')
def route_plan_lane_split_on_left_ends(route_plan_20_30):
    # Delete all left lanes, except for lanes 222 and 232
    for index in [0, 1, 4, 5, 6, 7, 8, 9]:
        route_plan_20_30.s_Data.a_Cnt_num_lane_segments[index] -= 1
        del route_plan_20_30.s_Data.as_route_plan_lane_segments[index][2]

    yield route_plan_20_30

@pytest.fixture(scope='function')
def route_plan_lane_splits_on_left_and_right_end(route_plan_20_30):
    # Delete all right and left lanes, except for lanes 220, 222, 230 and 232
    for index in [0, 1, 4, 5, 6, 7, 8, 9]:
        route_plan_20_30.s_Data.a_Cnt_num_lane_segments[index] -= 2
        del route_plan_20_30.s_Data.as_route_plan_lane_segments[index][2]
        del route_plan_20_30.s_Data.as_route_plan_lane_segments[index][0]

    yield route_plan_20_30

@pytest.fixture(scope='function')
def route_plan_lane_splits_offset(route_plan_20_30):
    # Delete all right and left lanes, except for lanes 220, 222, 230, 232, 240, 242
    for index in [0, 1]:
        route_plan_20_30.s_Data.a_Cnt_num_lane_segments[index] -= 2
        del route_plan_20_30.s_Data.as_route_plan_lane_segments[index][2]
        del route_plan_20_30.s_Data.as_route_plan_lane_segments[index][0]
    # delete  220, 242
    del route_plan_20_30.s_Data.as_route_plan_lane_segments[2][0]
    route_plan_20_30.s_Data.a_Cnt_num_lane_segments[2] -= 1
    del route_plan_20_30.s_Data.as_route_plan_lane_segments[4][2]
    route_plan_20_30.s_Data.a_Cnt_num_lane_segments[4] -= 1

    yield route_plan_20_30



@pytest.fixture(scope='function')
def route_plan_20():
    yield RoutePlan(s_Header=Header(e_Cnt_SeqNum=1, s_Timestamp=Timestamp(0, 0), e_Cnt_version=1),
                    s_Data=DataRoutePlan(e_b_is_valid=True,
                                         e_Cnt_num_road_segments=1,
                                         a_i_road_segment_ids=np.array([20]),
                                         a_Cnt_num_lane_segments=np.array([3]),
                                         as_route_plan_lane_segments=[[RoutePlanLaneSegment(e_i_lane_segment_id=lane_segment_id,
                                                                                            e_cst_lane_occupancy_cost=0.0,
                                                                                            e_cst_lane_end_cost=0.0)
                                                                       for lane_segment_id in [200, 201, 202]]]))


@pytest.fixture(scope='session')
def route_plan_for_oval_track_file():
    '''
    Currently, the pickle file that contains oval track data is missing road segments 10432, 3537, 76406, 3646, and 46577. This is the
    part of the oval track close to the entrance and exit. If the pickle file is updated to include those road segments, this fixture
    should be updated to account for them.
    '''
    nav_plan = [46613, 87759, 8766, 76838, 228030, 51360, 228028, 87622, 228007, 87660, 87744, 9893, 9894, 87740, 77398, 87741, 25969,
                10068, 87211, 10320, 10322, 228029, 87739, 40953, 10073, 10066, 87732, 43516, 87770, 228034, 87996, 228037, 10536, 88088,
                228039, 88192, 10519]

    # The lane segment index type is being defined and used here because this is a session fixture that will only be run once and this type
    # shouldn't be used anywhere else. It was created just for clarity in the below annotation.
    LaneSegmentIndex = int

    lane_cost_modifications: Dict[RoadSegmentID, List[Tuple[LaneSegmentIndex, LaneOccupancyCost, LaneEndCost]]] = \
        {87759: [(4, 1.0, 1.0)],
         76838: [(4, 1.0, 1.0)],
         228030: [(4, 1.0, 1.0)],
         228028: [(0, 1.0, 1.0)],
         87622: [(0, 1.0, 1.0)],
         228007: [(0, 1.0, 1.0)],
         40953: [(4, 0.0, 1.0)],
         228034: [(0, 1.0, 1.0)],
         87996: [(0, 1.0, 1.0)],
         228037: [(0, 1.0, 1.0)],
         88088: [(0, 1.0, 1.0)],
         228039: [(0, 1.0, 1.0)],
         88192: [(0, 1.0, 1.0)]}

    num_lane_segments = []
    route_plan_lane_segments = []

    SceneStaticModel.get_instance().set_scene_static(scene_static_accel_towards_vehicle())

    for road_segment_id in nav_plan:
        road_segment = MapUtils.get_road_segment(road_segment_id)
        num_lane_segments.append(road_segment.e_Cnt_lane_segment_id_count)

        # Set default values
        route_plan_road_segment = [RoutePlanLaneSegment(e_i_lane_segment_id=lane_segment_id,
                                                        e_cst_lane_occupancy_cost=0.0,
                                                        e_cst_lane_end_cost=0.0) for lane_segment_id in road_segment.a_i_lane_segment_ids]

        # Modify default values
        if road_segment_id in lane_cost_modifications:
            for lane_segment in lane_cost_modifications[road_segment_id]:
                route_plan_road_segment[lane_segment[0]].e_cst_lane_occupancy_cost = lane_segment[1]
                route_plan_road_segment[lane_segment[0]].e_cst_lane_end_cost = lane_segment[2]

        route_plan_lane_segments.append(route_plan_road_segment)

    yield RoutePlan(s_Header=Header(e_Cnt_SeqNum=1, s_Timestamp=Timestamp(0, 0), e_Cnt_version=1),
                    s_Data=DataRoutePlan(e_b_is_valid=True,
                                         e_Cnt_num_road_segments=len(nav_plan),
                                         a_i_road_segment_ids=np.array(nav_plan),
                                         a_Cnt_num_lane_segments=np.array(num_lane_segments),
                                         as_route_plan_lane_segments=route_plan_lane_segments))


# TODO: This should be a parametrized_fixture.
def create_route_plan_msg(road_segment_ids):
    return RoutePlan(s_Header=Header(e_Cnt_SeqNum=1, s_Timestamp=Timestamp(0, 0), e_Cnt_version=1),
                     s_Data=DataRoutePlan(e_b_is_valid=True,
                                          e_Cnt_num_road_segments=10,
                                          a_i_road_segment_ids=road_segment_ids,
                                          a_Cnt_num_lane_segments=0,
                                          as_route_plan_lane_segments=[]))


@pytest.fixture(scope='function')
def ego_state_for_takover_message_simple_scene():

    ego_lane_id = 101
    ego_lane_lon = 0  # station along the lane
    ego_vel = 10
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), ego_lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1,
                                               off_map=False)

    yield ego_state


@pytest.fixture(scope='function')
def ego_state_for_takover_message_default_scene():

    ego_lane_id = 201
    ego_lane_lon = 100  # station along the lane
    ego_vel = 10
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), ego_lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1,
                                               off_map=False)

    yield ego_state


@pytest.fixture(scope='function')
def state_with_surrounding_objects(route_plan_20_30: RoutePlan):

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split())

    road_segment_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = EGO_LANE_LON
    obj_vel = ego_vel = 10
    ego_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)[1]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), ego_lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1, off_map=False)

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1
    # Generate objects at the following locations:
    for rel_lane in RelativeLane:
        # calculate objects' lane_ids and longitudes: 20 m behind, parallel and 20 m ahead of ego on the relative lane
        parallel_lane_id = MapUtils.get_adjacent_lane_ids(ego_lane_id, rel_lane)[0] \
            if rel_lane != RelativeLane.SAME_LANE else ego_lane_id
        prev_lane_ids, back_lon = MapUtils._get_upstream_lanes_from_distance(parallel_lane_id, ego_lane_lon, 20)
        next_sub_segments, _, _ = MapUtils._advance_by_cost(parallel_lane_id, ego_lane_lon, 20, route_plan_20_30)[RelativeLane.SAME_LANE]
        obj_lane_lons = [back_lon, ego_lane_lon, next_sub_segments[-1].e_i_SEnd]
        obj_lane_ids = [prev_lane_ids[-1], parallel_lane_id, next_sub_segments[-1].e_i_SegmentID]

        for i, obj_lane_lon in enumerate(obj_lane_lons):

            if obj_lane_lon == ego_lane_lon and rel_lane == RelativeLane.SAME_LANE:
                # Don't create an object where the ego is
                continue

            map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_ids[i])
            dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                            size=car_size, confidence=1., off_map=False)
            dynamic_objects.append(dynamic_object)
            obj_id += 1

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_surrounding_objects_and_off_map_objects(route_plan_20_30: RoutePlan):

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split())

    road_segment_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = EGO_LANE_LON
    obj_vel = ego_vel = 10
    ego_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)[1]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), ego_lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1, off_map=False)

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1
    # Generate objects at the following locations:
    for rel_lane in RelativeLane:
        # calculate objects' lane_ids and longitudes: 20 m behind, parallel and 20 m ahead of ego on the relative lane
        parallel_lane_id = MapUtils.get_adjacent_lane_ids(ego_lane_id, rel_lane)[0] \
            if rel_lane != RelativeLane.SAME_LANE else ego_lane_id
        prev_lane_ids, back_lon = MapUtils._get_upstream_lanes_from_distance(parallel_lane_id, ego_lane_lon, 20)
        next_sub_segments, _, _ = MapUtils._advance_by_cost(parallel_lane_id, ego_lane_lon, 20,
                                                         route_plan=route_plan_20_30)[RelativeLane.SAME_LANE]
        obj_lane_lons = [back_lon, ego_lane_lon, next_sub_segments[-1].e_i_SEnd]
        obj_lane_ids = [prev_lane_ids[-1], parallel_lane_id, next_sub_segments[-1].e_i_SegmentID]
        for i, obj_lane_lon in enumerate(obj_lane_lons):

            if obj_lane_lon == ego_lane_lon and rel_lane == RelativeLane.SAME_LANE:
                # Don't create an object where the ego is
                continue

            map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_ids[i])
            dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                            size=car_size, confidence=1., off_map=False)
            # all the objects in the right lane are actually located off map
            dynamic_object.off_map = rel_lane == RelativeLane.RIGHT_LANE
            dynamic_objects.append(dynamic_object)
            obj_id += 1

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_objects_for_acceleration_towards_vehicle():
    # loads a scene dynamic where the vehicle is driving in its desired velocity towards another vehicle
    SceneStaticModel.get_instance().set_scene_static(scene_static_accel_towards_vehicle())
    scene_dynamic = scene_dynamic_accel_towards_vehicle()
    scene_dynamic.dynamic_objects[0].off_map = False
    # set a positive initial acceleration to create a scene where the vehicle is forced to exceed the desired velocity
    scene_dynamic.ego_state.cartesian_state[C_A] = 1
    yield scene_dynamic


@pytest.fixture(scope='function')
def state_with_objects_for_filtering_almost_tracking_mode(route_plan_20_30: RoutePlan):

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split())

    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = EGO_LANE_LON
    ego_vel = 10
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[1]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1, off_map=False)

    # Generate objects at the following locations:
    next_sub_segments, _, _ = MapUtils._advance_by_cost(lane_id, ego_lane_lon, 20, route_plan_20_30)[RelativeLane.SAME_LANE]
    obj_lane_lon = next_sub_segments[-1].e_i_SEnd
    obj_lane_id = next_sub_segments[-1].e_i_SegmentID
    obj_vel = 10.2

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1., off_map=False)

    dynamic_objects.append(dynamic_object)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_objects_for_filtering_exact_tracking_mode():

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split())

    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = 50
    ego_vel = 5
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[1]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1, off_map=False)

    # Generate objects at the following locations:
    obj_lane_lon = ego_lane_lon + car_size.length + LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT + ego_vel * SPECIFICATION_HEADWAY
    obj_vel = ego_vel

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1., off_map=False)

    dynamic_objects.append(dynamic_object)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)



@pytest.fixture(scope='function')
def state_with_objects_for_filtering_negative_sT(route_plan_20_30: RoutePlan):

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split())

    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = EGO_LANE_LON
    ego_vel = 10
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[1]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1, off_map=False)

    # Generate objects at the following locations:
    next_sub_segments, _, _ = MapUtils._advance_by_cost(lane_id, ego_lane_lon, 3.8, route_plan_20_30)[RelativeLane.SAME_LANE]
    obj_lane_lon = next_sub_segments[-1].e_i_SEnd
    obj_lane_id = next_sub_segments[-1].e_i_SegmentID
    obj_vel = 11

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1., off_map=False)

    dynamic_objects.append(dynamic_object)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_traffic_control(route_plan_20_30: RoutePlan):

    scene_static_with_traffic = scene_static_pg_split()
    SceneStaticModel.get_instance().set_scene_static(scene_static_with_traffic)

    stop_sign = StaticTrafficFlowControl(e_e_road_object_type=RoadObjectType.StopSign, e_l_station=20, e_Pct_confidence=1.0)
    scene_static_with_traffic.s_Data.s_SceneStaticBase.as_scene_lane_segments[0].as_static_traffic_flow_control.append(stop_sign)
    SceneStaticModel.get_instance().set_scene_static(scene_static_with_traffic)

    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = EGO_LANE_LON
    ego_vel = 10
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[1]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1, off_map=False)

    # Generate objects at the following locations:
    next_sub_segments, _, _ = MapUtils._advance_by_cost(lane_id, ego_lane_lon, 3.8, route_plan_20_30)[RelativeLane.SAME_LANE]
    obj_lane_lon = next_sub_segments[-1].e_i_SEnd
    obj_lane_id = next_sub_segments[-1].e_i_SegmentID
    obj_vel = 11

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1., off_map=False)

    dynamic_objects.append(dynamic_object)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_left_lane_ending():
    SceneStaticModel.get_instance().set_scene_static(scene_static_left_lane_ends())
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    ego_lane_lon = 700
    ego_vel = 10
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(1)[1]
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)
    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)


    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1,
                                               off_map=False)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=[], ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_right_lane_ending():
    SceneStaticModel.get_instance().set_scene_static(scene_static_right_lane_ends())
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    ego_lane_lon = 700
    ego_vel = 10
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(1)[1]
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)
    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)


    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1,
                                               off_map=False)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=[], ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_same_lane_ending_no_left_lane():
    SceneStaticModel.get_instance().set_scene_static(scene_static_left_lane_ends())
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    ego_lane_lon = 700
    ego_vel = 10
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(1)[2]
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)
    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)


    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1,
                                               off_map=False)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=[], ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_same_lane_ending_no_right_lane():
    SceneStaticModel.get_instance().set_scene_static(scene_static_right_lane_ends())
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    ego_lane_lon = 700
    ego_vel = 10
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(1)[0]
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)
    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)


    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1,
                                               off_map=False)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=[], ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_lane_split_on_right():
    SceneStaticModel.get_instance().set_scene_static(right_lane_split_scene_static())
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    ego_lane_lon = 700
    ego_vel = 10
    lane_id = 11
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)
    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)


    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1,
                                               off_map=False)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=[], ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_lane_split_on_left():
    SceneStaticModel.get_instance().set_scene_static(left_lane_split_scene_static())
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    ego_lane_lon = 700
    ego_vel = 10
    lane_id = 11
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)
    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)


    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1,
                                               off_map=False)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=[], ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_lane_split_on_left_and_right():
    SceneStaticModel.get_instance().set_scene_static(left_right_lane_split_scene_static())
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    ego_lane_lon = 700
    ego_vel = 10
    lane_id = 11
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)
    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)


    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1,
                                               off_map=False)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=[], ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_lane_split_on_left_and_right_offset():
    SceneStaticModel.get_instance().set_scene_static(scene_static_lane_splits_on_left_and_right_offset())
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    ego_lane_lon = 100
    ego_vel = 10
    lane_id = 211
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)
    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)


    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1,
                                               off_map=False)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=[], ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_lane_split_on_right_ending():
    SceneStaticModel.get_instance().set_scene_static(scene_static_lane_split_on_right_ends())
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    ego_lane_lon = 100
    ego_vel = 10
    lane_id = 211
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)
    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)


    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1,
                                               off_map=False)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=[], ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_lane_split_on_left_ending():
    SceneStaticModel.get_instance().set_scene_static(scene_static_lane_split_on_left_ends())
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    ego_lane_lon = 100
    ego_vel = 10
    lane_id = 211
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)
    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)


    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1,
                                               off_map=False)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=[], ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_lane_split_on_left_and_right_ending():
    SceneStaticModel.get_instance().set_scene_static(scene_static_lane_splits_on_left_and_right_end())
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    ego_lane_lon = 100
    ego_vel = 10
    lane_id = 211
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)
    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)


    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1,
                                               off_map=False)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=[], ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_objects_for_filtering_too_aggressive(route_plan_20_30: RoutePlan):

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split())

    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = EGO_LANE_LON
    ego_vel = 10
    right_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[0]
    mid_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[1]

    ego_map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), mid_lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=ego_map_state, size=car_size, confidence=1, off_map=False)

    # Generate objects at the following locations:
    mid_next_sub_segments, _, _ = MapUtils._advance_by_cost(mid_lane_id, ego_lane_lon, 58, route_plan_20_30)[RelativeLane.SAME_LANE]
    mid_obj_lane_lon = mid_next_sub_segments[-1].e_i_SEnd
    mid_obj_lane_id = mid_next_sub_segments[-1].e_i_SegmentID
    mid_obj_vel = 30

    right_next_sub_segments, _, _ = MapUtils._advance_by_cost(right_lane_id, ego_lane_lon, 58, route_plan_20_30)[RelativeLane.SAME_LANE]
    right_obj_lane_lon = right_next_sub_segments[-1].e_i_SEnd
    right_obj_lane_id = right_next_sub_segments[-1].e_i_SegmentID
    right_obj_vel = 20

    dynamic_objects: List[DynamicObject] = list()

    mid_obj_id = 1
    mid_map_state = MapState(np.array([mid_obj_lane_lon, mid_obj_vel, 0, 0, 0, 0]), mid_obj_lane_id)
    mid_dynamic_object = DynamicObject.create_from_map_state(obj_id=mid_obj_id, timestamp=0, map_state=mid_map_state,
                                                             size=car_size, confidence=1., off_map=False)
    right_obj_id = 2
    right_map_state = MapState(np.array([right_obj_lane_lon, right_obj_vel, 0, 0, 0, 0]), right_obj_lane_id)
    right_dynamic_object = DynamicObject.create_from_map_state(obj_id=right_obj_id, timestamp=0, map_state=right_map_state,
                                                               size=car_size, confidence=1., off_map=False)

    dynamic_objects.append(mid_dynamic_object)
    dynamic_objects.append(right_dynamic_object)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_for_testing_lanes_speed_limits_violations(route_plan_20_30: RoutePlan):

    scene_static_with_limits = scene_static_pg_split()
    SceneStaticModel.get_instance().set_scene_static(scene_static_with_limits)


    road_id = 20

    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = EGO_LANE_LON
    ego_vel = 10
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[0]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, -0.1, 0, 0, 0]), lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1, off_map=False)

    # Generate objects at the following locations:
    next_sub_segments, _, _ = MapUtils._advance_by_cost(lane_id, ego_lane_lon, 3.8, route_plan_20_30)[RelativeLane.SAME_LANE]
    obj_lane_lon = next_sub_segments[-1].e_i_SEnd
    obj_lane_id = next_sub_segments[-1].e_i_SegmentID
    obj_vel = 11

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1., off_map=False)

    dynamic_objects.append(dynamic_object)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def behavioral_grid_state(state_with_surrounding_objects: State, route_plan_20_30: RoutePlan):
    yield BehavioralGridState.create_from_state(state_with_surrounding_objects,
                                                route_plan_20_30, None)


@pytest.fixture(scope='function')
def behavioral_grid_state_with_objects_for_acceleration_towards_vehicle(
        state_with_objects_for_acceleration_towards_vehicle, route_plan_for_oval_track_file: RoutePlan):
    yield BehavioralGridState.create_from_state(state_with_objects_for_acceleration_towards_vehicle,
                                                route_plan_for_oval_track_file, None)


@pytest.fixture(scope='function')
def behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode(
        state_with_objects_for_filtering_almost_tracking_mode: State, route_plan_20_30: RoutePlan):
    yield BehavioralGridState.create_from_state(state_with_objects_for_filtering_almost_tracking_mode,
                                                route_plan_20_30, None)


@pytest.fixture(scope='function')
def behavioral_grid_state_with_objects_for_filtering_exact_tracking_mode(
        state_with_objects_for_filtering_exact_tracking_mode: State, route_plan_20_30: RoutePlan):
    yield BehavioralGridState.create_from_state(state_with_objects_for_filtering_exact_tracking_mode,
                                                route_plan_20_30, None)


@pytest.fixture(scope='function')
def behavioral_grid_state_with_objects_for_filtering_negative_sT(state_with_objects_for_filtering_negative_sT: State,
                                                                 route_plan_20_30: RoutePlan):
    yield BehavioralGridState.create_from_state(state_with_objects_for_filtering_negative_sT,
                                                route_plan_20_30, None)

@pytest.fixture(scope='function')
def behavioral_grid_state_with_objects_for_filtering_too_aggressive(
        state_with_objects_for_filtering_too_aggressive: State, route_plan_20_30: RoutePlan):
    yield BehavioralGridState.create_from_state(state_with_objects_for_filtering_too_aggressive,
                                                route_plan_20_30, None)

@pytest.fixture(scope='function')
def behavioral_grid_state_with_traffic_control(state_with_traffic_control: State, route_plan_20_30: RoutePlan):

    scene_static_with_traffic = scene_static_pg_split()
    stop_sign = StaticTrafficFlowControl(e_e_road_object_type=RoadObjectType.StopSign, e_l_station=20, e_Pct_confidence=1.0)
    scene_static_with_traffic.s_Data.s_SceneStaticBase.as_scene_lane_segments[0].as_static_traffic_flow_control.append(stop_sign)
    SceneStaticModel.get_instance().set_scene_static(scene_static_with_traffic)

    yield BehavioralGridState.create_from_state(state_with_traffic_control,
                                                route_plan_20_30, None)

@pytest.fixture(scope='function')
def behavioral_grid_state_with_segments_limits(state_for_testing_lanes_speed_limits_violations, route_plan_20_30: RoutePlan):
    yield BehavioralGridState.create_from_state(state_for_testing_lanes_speed_limits_violations,
                                                route_plan_20_30, None)

@pytest.fixture(scope='function')
def behavioral_grid_state_with_left_lane_ending(state_with_left_lane_ending, route_plan_1_2):
    yield BehavioralGridState.create_from_state(state_with_left_lane_ending, route_plan_1_2, None)


@pytest.fixture(scope='function')
def follow_vehicle_recipes_towards_front_cells():
    yield [DynamicActionRecipe(lane, RelativeLongitudinalPosition.FRONT, ActionType.FOLLOW_VEHICLE, agg)
           for lane in RelativeLane
           for agg in AggressivenessLevel]


@pytest.fixture(scope='function')
def follow_lane_recipes():
    velocity_grid = np.arange(0, 30 + EPS, 6)
    yield [StaticActionRecipe(RelativeLane.SAME_LANE, velocity, agg)
           for velocity in velocity_grid
           for agg in AggressivenessLevel]


