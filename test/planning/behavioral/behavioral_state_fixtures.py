import pickle
from decision_making.paths import Paths
from decision_making.src.planning.types import FS_SA, C_A

from decision_making.src.messages.scene_static_message import StaticTrafficFlowControl, RoadObjectType
from decision_making.src.messages.route_plan_message import RoutePlan, DataRoutePlan
from decision_making.src.messages.scene_common_messages import Header, Timestamp
from typing import List

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
    scene_static_accel_towards_vehicle, scene_dynamic_accel_towards_vehicle

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
                    s_Data=DataRoutePlan(e_b_is_valid=True,
                                         e_Cnt_num_road_segments=10,
                                         a_i_road_segment_ids=np.arange(20, 30, 1, dtype=np.int),
                                         a_Cnt_num_lane_segments=0,
                                         as_route_plan_lane_segments=[]))


@pytest.fixture(scope='function')
def route_plan_20():
    yield RoutePlan(s_Header=Header(e_Cnt_SeqNum=1, s_Timestamp=Timestamp(0, 0), e_Cnt_version=1),
                    s_Data=DataRoutePlan(e_b_is_valid=True,
                                         e_Cnt_num_road_segments=1,
                                         a_i_road_segment_ids=np.array([20]),
                                         a_Cnt_num_lane_segments=0,
                                         as_route_plan_lane_segments=[]))

@pytest.fixture(scope='function')
def route_plan_oval_track():
    yield RoutePlan(s_Header=Header(e_Cnt_SeqNum=1, s_Timestamp=Timestamp(0, 0), e_Cnt_version=1),
                    s_Data=DataRoutePlan(e_b_is_valid=True,
                                         e_Cnt_num_road_segments=1,
                                         a_i_road_segment_ids=NAVIGATION_PLAN_OVAL_TRACK,
                                         a_Cnt_num_lane_segments=0,
                                         as_route_plan_lane_segments=[]))


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
    ego_lane_lon = 0 # station along the lane
    ego_vel = 10
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), ego_lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    yield ego_state


@pytest.fixture(scope='function')
def ego_state_for_takover_message_default_scene():

    ego_lane_id = 201
    ego_lane_lon = 100 # station along the lane
    ego_vel = 10
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), ego_lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    yield ego_state


@pytest.fixture(scope='function')
def state_with_sorrounding_objects(route_plan_20_30: RoutePlan):

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
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1
    # Generate objects at the following locations:
    for rel_lane in RelativeLane:
        # calculate objects' lane_ids and longitudes: 20 m behind, parallel and 20 m ahead of ego on the relative lane
        parallel_lane_id = MapUtils.get_adjacent_lane_ids(ego_lane_id, rel_lane)[0] \
            if rel_lane != RelativeLane.SAME_LANE else ego_lane_id
        prev_lane_ids, back_lon = MapUtils._get_upstream_lanes_from_distance(parallel_lane_id, ego_lane_lon, 20)
        next_sub_segments = MapUtils._advance_on_plan(parallel_lane_id, ego_lane_lon, 20,
                                                      route_plan_road_ids=route_plan_20_30.s_Data.a_i_road_segment_ids)
        obj_lane_lons = [back_lon, ego_lane_lon, next_sub_segments[-1].e_i_SEnd]
        obj_lane_ids = [prev_lane_ids[-1], parallel_lane_id, next_sub_segments[-1].e_i_SegmentID]

        for i, obj_lane_lon in enumerate(obj_lane_lons):

            if obj_lane_lon == ego_lane_lon and rel_lane == RelativeLane.SAME_LANE:
                # Don't create an object where the ego is
                continue

            map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_ids[i])
            dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                            size=car_size, confidence=1.)
            dynamic_objects.append(dynamic_object)
            obj_id += 1

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_objects_for_acceleration_towards_vehicle():
    # loads a scene dynamic where the vehicle is driving in its desired velocity towards another vehicle
    SceneStaticModel.get_instance().set_scene_static(scene_static_accel_towards_vehicle())
    scene_dynamic = scene_dynamic_accel_towards_vehicle()
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
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    # Generate objects at the following locations:
    next_sub_segments = MapUtils._advance_on_plan(lane_id, ego_lane_lon, 20,
                                                  route_plan_road_ids=route_plan_20_30.s_Data.a_i_road_segment_ids)
    obj_lane_lon = next_sub_segments[-1].e_i_SEnd
    obj_lane_id = next_sub_segments[-1].e_i_SegmentID
    obj_vel = 10.2

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1.)

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
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    # Generate objects at the following locations:
    obj_lane_lon = ego_lane_lon + car_size.length + LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT + ego_vel * SPECIFICATION_HEADWAY
    obj_vel = ego_vel

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1.)

    dynamic_objects.append(dynamic_object)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)

@pytest.fixture(scope='function')
def state_with_segment_limits():

    scene_static_with_limits = scene_static_pg_split()
    SceneStaticModel.get_instance().set_scene_static(scene_static_with_limits)

    scene_static_with_limits.s_Data.s_SceneStaticBase.as_scene_lane_segments[0].e_v_nominal_speed = 10
    SceneStaticModel.get_instance().set_scene_static(scene_static_with_limits)

    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = EGO_LANE_LON
    ego_vel = 10
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[1]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    # Generate objects at the following locations:
    next_sub_segments = MapUtils._advance_on_plan(lane_id, ego_lane_lon, 3.8, NAVIGATION_PLAN)
    obj_lane_lon = next_sub_segments[-1].e_i_SEnd
    obj_lane_id = next_sub_segments[-1].e_i_SegmentID
    obj_vel = 11

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1.)

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
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    # Generate objects at the following locations:
    next_sub_segments = MapUtils._advance_on_plan(lane_id, ego_lane_lon, 3.8, route_plan_20_30.s_Data.a_i_road_segment_ids)
    obj_lane_lon = next_sub_segments[-1].e_i_SEnd
    obj_lane_id = next_sub_segments[-1].e_i_SegmentID
    obj_vel = 11

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1.)

    dynamic_objects.append(dynamic_object)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_traffic_control():

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
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    # Generate objects at the following locations:
    next_sub_segments = MapUtils._advance_on_plan(lane_id, ego_lane_lon, 3.8, NAVIGATION_PLAN)
    obj_lane_lon = next_sub_segments[-1].e_i_SEnd
    obj_lane_id = next_sub_segments[-1].e_i_SegmentID
    obj_vel = 11

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1.)

    dynamic_objects.append(dynamic_object)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


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
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[1]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    # Generate objects at the following locations:
    next_sub_segments = MapUtils._advance_on_plan(lane_id, ego_lane_lon, 58, route_plan_20_30.s_Data.a_i_road_segment_ids)
    obj_lane_lon = next_sub_segments[-1].e_i_SEnd
    obj_lane_id = next_sub_segments[-1].e_i_SegmentID
    obj_vel = 30

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1.)

    dynamic_objects.append(dynamic_object)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def behavioral_grid_state(state_with_sorrounding_objects: State, route_plan_20_30: RoutePlan):
    yield BehavioralGridState.create_from_state(state_with_sorrounding_objects,
                                                route_plan_20_30, None)


@pytest.fixture(scope='function')
def behavioral_grid_state_with_objects_for_acceleration_towards_vehicle(
        state_with_objects_for_acceleration_towards_vehicle, route_plan_oval_track: RoutePlan):
    yield BehavioralGridState.create_from_state(state_with_objects_for_acceleration_towards_vehicle,
                                                route_plan_oval_track, None)


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

