import numpy as np

from decision_making.src.map.map_api import MapAPI
from decision_making.src.map.naive_cache_map import NaiveCacheMap
from decision_making.src.planning.utils.acda import AcdaApi
from decision_making.src.planning.utils.acda_constants import *
from decision_making.src.state.state import EgoState, ObjectSize, DynamicObject, RoadLocalization, \
    RelativeRoadLocalization


class MapMock(MapAPI):
    def __init__(self):
        pass


def test_calc_road_turn_radius_TurnOnCircle_successful():
    # simulate curved road structure
    road_turn_radians = np.pi
    road_points_num = 30
    turn_amp = 5.0
    road_points = np.zeros(shape=[2, road_points_num])
    road_points[0, :] = -1 * turn_amp * np.sin(np.linspace(0, road_turn_radians, road_points_num))
    road_points[1, :] = turn_amp * np.cos(np.linspace(0, road_turn_radians, road_points_num))
    turn_radius = AcdaApi.calc_road_turn_radius(road_points)
    assert np.abs(turn_radius - turn_amp) < 0.001

def test_calc_safe_speed_critical_speed_CheckSpeed_successful():
    # test calc_safe_speed_critical_speed
    assert AcdaApi.calc_safe_speed_critical_speed(curve_radius=5.0) > 0

def test_calc_safe_speed_following_distance_CheckSpeed_successful():
    # test calc_safe_speed_following_distance
    assert AcdaApi.calc_safe_speed_following_distance(following_distance=10.0) > 0

def test_calc_safe_speed_forward_line_of_sight_CheckSpeed_successful():
    # test calc_safe_speed_forward_line_of_sight
    assert AcdaApi.calc_safe_speed_forward_line_of_sight(forward_sight_distance=10.0) > 0

def test_AcdaFeaturesInComplexScenraio_successful():
    # Prepare scenario for test

    # Prepare cached map
    map_api = MapMock()

    # ego state at (0,0,0)
    road_localization = RoadLocalization(road_id=1, lane_num=0, full_lat=0.0, intra_lane_lat=0.0, road_lon=0.0,
                                         intra_lane_yaw=0.0)
    relative_road_localization = RelativeRoadLocalization(rel_lat=0.0, rel_lon=0.0, rel_yaw=0.0)
    ego_state = EgoState(obj_id=0, timestamp=0, x=0.0, y=0.0, z=0.0, yaw=0.0,
                         size=ObjectSize(length=2.5, width=1.5, height=1.0),
                         confidence=1.0, v_x=0.0, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, yaw_deriv=0.0, map_api=map_api, road_localization=road_localization,
                         rel_road_localization=relative_road_localization)

    # obstacle at (10,1.5,0)
    road_localization = RoadLocalization(road_id=1, lane_num=0, full_lat=1.5, intra_lane_lat=1.5, road_lon=10.0,
                                         intra_lane_yaw=0.0)
    relative_road_localization = RelativeRoadLocalization(rel_lat=1.5, rel_lon=10.0, rel_yaw=0.0)
    near_static_object = DynamicObject(obj_id=1, timestamp=0, x=10.0, y=1.0, z=0.0, yaw=0.0,
                                       size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                       confidence=1.0, v_x=0.0, v_y=0.0,
                                       acceleration_lon=0.0, yaw_deriv=0.0, ego_state=ego_state, map_api=map_api,
                                       road_localization=road_localization,
                                       rel_road_localization=relative_road_localization)

    # obstacle at (15,2.5,0)
    road_localization = RoadLocalization(road_id=1, lane_num=0, full_lat=2.5, intra_lane_lat=2.5, road_lon=15.0,
                                         intra_lane_yaw=0.0)
    relative_road_localization = RelativeRoadLocalization(rel_lat=2.5, rel_lon=15.0, rel_yaw=0.0)
    far_static_object = DynamicObject(obj_id=1, timestamp=0, x=15.0, y=2.5, z=0.0, yaw=0.0,
                                      size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                      confidence=1.0, v_x=0.0, v_y=0.0,
                                      acceleration_lon=0.0, yaw_deriv=0.0, ego_state=ego_state, map_api=map_api,
                                      road_localization=road_localization,
                                      rel_road_localization=relative_road_localization)

    objects_on_road = list()
    objects_on_road.append(near_static_object)
    objects_on_road.append(far_static_object)

    # Test ACDA functions

    # test calc_forward_sight_distance
    forward_distance = 10.0
    assert np.abs(AcdaApi.calc_forward_sight_distance(static_objects=objects_on_road,
                                                      ego_state=ego_state) - (
                  forward_distance - SENSOR_OFFSET_FROM_FRONT)) < 0.001
    # test calc_horizontal_sight_distance
    horizontal_dist = 2.5 - 0.5*(ego_state.size.width + far_static_object.size.width)
    assert np.abs(AcdaApi.calc_horizontal_sight_distance(static_objects=objects_on_road, ego_state=ego_state,
                                                         set_safety_lookahead_dist_by_ego_vel=True) - horizontal_dist) < 0.001

    # test compute_acda
    lookahead_path = np.zeros(shape=[2, 20])
    lookahead_path[0, :] = np.linspace(0, 10, 20)
    AcdaApi.compute_acda(objects_on_road=objects_on_road, ego_state=ego_state, lookahead_path=lookahead_path)

