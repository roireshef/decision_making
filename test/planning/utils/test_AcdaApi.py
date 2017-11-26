import numpy as np
import pytest

from unittest.mock import patch

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.utils.acda import AcdaApi
from decision_making.src.planning.utils.acda_constants import SENSOR_OFFSET_FROM_FRONT
from decision_making.src.state.state import EgoState, ObjectSize, DynamicObject
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from mapping.src.model.localization import RelativeRoadLocalization
from mapping.src.model.map_api import MapAPI
from mapping.test.model.map_model_utils import TestMapModelUtils
from rte.python.logger.AV_logger import AV_Logger

MAP_INFLATION_FACTOR = 300.0
NUM_LANES = 3
LANE_WIDTH = 3.0


class MapMock(MapAPI):
    pass


@pytest.fixture()
def testable_navigation_plan():
    yield NavigationPlanMsg(road_ids=np.array([1, 2]))


@pytest.fixture()
def map_api_mock():
    # Create a rectangle test map
    road_coordinates_1 = np.array([[0., 0.],
                                   [1., 0.],
                                   [2., 0.],
                                   [2., 0.5],
                                   [2., 1.],
                                   [1., 1.]]) * MAP_INFLATION_FACTOR
    road_coordinates_2 = np.array([[1., 1.],
                                   [0., 1.],
                                   [0., 0.5],
                                   [0., 0.1]]) * MAP_INFLATION_FACTOR
    road_coordinates = list()
    road_coordinates.append(road_coordinates_1)
    road_coordinates.append(road_coordinates_2)
    test_map_model = TestMapModelUtils.create_road_map_from_coordinates(points_of_roads=road_coordinates, road_id=1,
                                                                        road_name='def',
                                                                        lanes_num=NUM_LANES, lane_width=LANE_WIDTH)

    return MapMock(map_model=test_map_model, logger=AV_Logger.get_logger('test_map_acda'))


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


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_AcdaFeaturesInComplexScenraio_successful(testable_navigation_plan):
    logger = AV_Logger.get_logger('acda_test')

    # Prepare scenario for test

    # Prepare cached map
    navigation_plan = testable_navigation_plan

    # ego state at (0,0,0)
    # road_localization = RoadLocalization(road_id=1, lane_num=0, full_lat=0.0, intra_lane_lat=0.0, road_lon=0.0,
    #                                      intra_lane_yaw=0.0)
    relative_road_localization = RelativeRoadLocalization(rel_lat=0.0, rel_lon=0.0, rel_yaw=0.0)
    ego_state = EgoState(obj_id=0, timestamp=0, x=0.0, y=0.0, z=0.0, yaw=0.0,
                         size=ObjectSize(length=2.5, width=1.5, height=1.0),
                         confidence=1.0, v_x=0.0, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0)

    # obstacle at (10,1.5,0)
    # road_localization = RoadLocalization(road_id=1, lane_num=0, full_lat=1.5, intra_lane_lat=1.5, road_lon=10.0,
    #                                      intra_lane_yaw=0.0)
    # relative_road_localization should now be computed on-the-fly using DynamicObject.get_relative_road_localization
    # relative_road_localization = RelativeRoadLocalization(rel_lat=1.5, rel_lon=10.0, rel_yaw=0.0)
    near_static_object = DynamicObject(obj_id=1, timestamp=0, x=10.0, y=1.0, z=0.0, yaw=0.0,
                                       size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                       confidence=1.0, v_x=0.0, v_y=0.0,
                                       acceleration_lon=0.0, omega_yaw=0.0)

    # obstacle at (15,2.5,0)
    # road_localization = RoadLocalization(road_id=1, lane_num=0, full_lat=2.5, intra_lane_lat=2.5, road_lon=15.0,
    #                                      intra_lane_yaw=0.0)
    # relative_road_localization should now be computed on-the-fly using DynamicObject.get_relative_road_localization
    # relative_road_localization = RelativeRoadLocalization(rel_lat=2.5, rel_lon=15.0, rel_yaw=0.0)
    far_static_object = DynamicObject(obj_id=1, timestamp=0, x=15.0, y=2.5, z=0.0, yaw=0.0,
                                      size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                      confidence=1.0, v_x=0.0, v_y=0.0,
                                      acceleration_lon=0.0, omega_yaw=0.0)

    objects_on_road = list()
    objects_on_road.append(near_static_object)
    objects_on_road.append(far_static_object)

    # Test ACDA functions

    # test calc_forward_sight_distance
    forward_distance = 10.0
    assert np.abs(AcdaApi.calc_forward_sight_distance(static_objects=objects_on_road,
                                                      ego_state=ego_state, navigation_plan=navigation_plan) -
                  (forward_distance - SENSOR_OFFSET_FROM_FRONT)) < 0.001
    # test calc_horizontal_sight_distance
    horizontal_dist = 2.5 - 0.5 * (ego_state.size.width + far_static_object.size.width)
    assert np.abs(AcdaApi.calc_horizontal_sight_distance(static_objects=objects_on_road, ego_state=ego_state,
                                                         navigation_plan=navigation_plan,
                                                         set_safety_lookahead_dist_by_ego_vel=True) - horizontal_dist) < 0.001

    # test compute_acda
    lookahead_path = np.zeros(shape=[2, 20])
    lookahead_path[0, :] = np.linspace(0, 10, 20)
    safe_speed = AcdaApi.compute_acda(objects_on_road=objects_on_road, ego_state=ego_state,
                                      navigation_plan=navigation_plan, lookahead_path=lookahead_path)

    assert safe_speed > 0.0
