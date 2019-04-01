from typing import List

import numpy as np
import pytest

from decision_making.src.global_constants import EPS, FILTER_V_T_GRID
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, ActionType, AggressivenessLevel, \
    StaticActionRecipe
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import OccupancyState, State, ObjectSize, EgoState, DynamicObject
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.messages.static_scene_fixture import scene_static, scene_static_ovalmilford

NAVIGATION_PLAN = NavigationPlanMsg(np.array(range(20, 30)))
MILFORD_NAVIGATION_PLAN = NavigationPlanMsg(np.array([1]))
EGO_LANE_LON = 120.  # ~2 meters behind end of a lane segment


@pytest.fixture(scope='function')
def state_with_sorrounding_objects():

    SceneStaticModel.get_instance().set_scene_static(scene_static())

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
        next_sub_segments = MapUtils._advance_on_plan(parallel_lane_id, ego_lane_lon, 20, NAVIGATION_PLAN)
        obj_lane_lons = [back_lon, ego_lane_lon, next_sub_segments[-1].s_end]
        obj_lane_ids = [prev_lane_ids[-1], parallel_lane_id, next_sub_segments[-1].segment_id]

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
def state_with_objects_for_filtering_tracking_mode():

    SceneStaticModel.get_instance().set_scene_static(scene_static())

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
    next_sub_segments = MapUtils._advance_on_plan(lane_id, ego_lane_lon, 20, NAVIGATION_PLAN)
    obj_lane_lon = next_sub_segments[-1].s_end
    obj_lane_id = next_sub_segments[-1].segment_id
    obj_vel = 10.2

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1.)

    dynamic_objects.append(dynamic_object)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_objects_for_filtering_negative_sT():

    SceneStaticModel.get_instance().set_scene_static(scene_static())

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
    obj_lane_lon = next_sub_segments[-1].s_end
    obj_lane_id = next_sub_segments[-1].segment_id
    obj_vel = 11

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1.)

    dynamic_objects.append(dynamic_object)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_objects_for_filtering_too_aggressive():

    SceneStaticModel.get_instance().set_scene_static(scene_static())

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
    next_sub_segments = MapUtils._advance_on_plan(lane_id, ego_lane_lon, 58, NAVIGATION_PLAN)
    obj_lane_lon = next_sub_segments[-1].s_end
    obj_lane_id = next_sub_segments[-1].segment_id
    obj_vel = 30

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1.)

    dynamic_objects.append(dynamic_object)

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)

@pytest.fixture(scope='function')
def state_before_curvature_ovalmilford():

    SceneStaticModel.get_instance().set_scene_static(scene_static_ovalmilford())

    road_segment_ids = MapUtils.get_road_segment_ids()
    road_segment_id = road_segment_ids[0]

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = 140# take a point before curvature
    ego_vel = 30 # high speed
    ego_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)[0]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), ego_lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    dynamic_objects: List[DynamicObject] = list()

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_without_objects_ovalmilford():

    SceneStaticModel.get_instance().set_scene_static(scene_static_ovalmilford())

    road_segment_ids = MapUtils.get_road_segment_ids()
    road_segment_id = road_segment_ids[0]

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = 100
    ego_vel = 20
    ego_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)[1]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), ego_lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    dynamic_objects: List[DynamicObject] = list()

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)



@pytest.fixture(scope='function')
def behavioral_grid_state():
    yield BehavioralGridState.create_from_state(next(state_with_sorrounding_objects()),
                                                NAVIGATION_PLAN, None)


@pytest.fixture(scope='function')
def behavioral_grid_state_with_objects_for_filtering_tracking_mode(
        state_with_objects_for_filtering_tracking_mode: State):
    yield BehavioralGridState.create_from_state(state_with_objects_for_filtering_tracking_mode,
                                                NAVIGATION_PLAN, None)


@pytest.fixture(scope='function')
def behavioral_grid_state_with_objects_for_filtering_negative_sT(state_with_objects_for_filtering_negative_sT: State):
    yield BehavioralGridState.create_from_state(state_with_objects_for_filtering_negative_sT,
                                                NAVIGATION_PLAN, None)


@pytest.fixture(scope='function')
def behavioral_grid_state_with_objects_for_filtering_too_aggressive(
        state_with_objects_for_filtering_too_aggressive: State):
    yield BehavioralGridState.create_from_state(state_with_objects_for_filtering_too_aggressive,
                                                NAVIGATION_PLAN, None)


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


@pytest.fixture(scope='function')
def all_follow_lane_recipes():
    velocity_grid = np.arange(0, FILTER_V_T_GRID.end + EPS, 2)
    yield [StaticActionRecipe(RelativeLane.SAME_LANE, velocity, agg)
           for velocity in velocity_grid
           for agg in AggressivenessLevel]


@pytest.fixture(scope='function')
def follow_vehicle_recipes_towards_front_same_lane():
    yield [DynamicActionRecipe(RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT, ActionType.FOLLOW_VEHICLE, agg)
           for agg in AggressivenessLevel]
