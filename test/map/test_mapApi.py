import numpy as np
import pytest

from decision_making.src.map.map_api import MapAPI
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.test.map.map_model_utils import TestMapModelUtils
from rte.python.logger.AV_logger import AV_Logger

NUM_LANES = 3
LANE_WIDTH = 3.0
ROAD_WIDTH = LANE_WIDTH * NUM_LANES
MAP_INFLATION_FACTOR = 300.0


class TestableMapApi(MapAPI):
    def call_shift_road_vector_in_latitude(self, points: np.ndarray, lat_shift: float) -> np.ndarray:
        return self._shift_road_points_in_latitude(points, lat_shift)


@pytest.fixture()
def testable_map_api():
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

    yield TestableMapApi(map_model=test_map_model, logger=AV_Logger.get_logger('tes_map'))


def test_shiftRoadVector_simpleRoadShift1MRight_accurateShift(testable_map_api):
    points = np.array([[0, 0], [1, -1], [1, -2]])
    shift = -1
    shifted_points = testable_map_api.call_shift_road_vector_in_latitude(points, shift)
    expected_shifted_points = np.array([[-1 / np.sqrt(2), -1 / np.sqrt(2)], [0, -1], [0, -2]])

    np.testing.assert_array_almost_equal(shifted_points, expected_shifted_points)


def test_convertRoadToGlobalCoordinates_accurateConversion(testable_map_api):
    road = testable_map_api._cached_map_model.get_road_data(1)

    lon_of_first_segment = 10.2
    point_on_first_segment = np.array([10.2, -ROAD_WIDTH / 2])
    point_lateral_shift = 9.
    right_edge_position, world_yaw = testable_map_api._convert_road_to_global_coordinates(road_id=1, lat=0.0,
                                                                                          lon=lon_of_first_segment)
    shifted_world_position, world_yaw = testable_map_api._convert_road_to_global_coordinates(road_id=1,
                                                                                             lat=point_lateral_shift,
                                                                                             lon=lon_of_first_segment)

    np.testing.assert_array_almost_equal(right_edge_position[0:2], point_on_first_segment)
    np.testing.assert_array_almost_equal(shifted_world_position[0:2],
                                         point_on_first_segment + np.array([0., point_lateral_shift]))

    lon_of_second_segment = MAP_INFLATION_FACTOR * 2 + 10.2
    point_on_second_segment = np.array([MAP_INFLATION_FACTOR * 2 + ROAD_WIDTH / 2, 10.2])
    point_lateral_shift = 9.
    right_edge_position, world_yaw = testable_map_api._convert_road_to_global_coordinates(road_id=1, lat=0.0,
                                                                                          lon=lon_of_second_segment)
    shifted_world_position, world_yaw = testable_map_api._convert_road_to_global_coordinates(road_id=1,
                                                                                             lat=point_lateral_shift,
                                                                                             lon=lon_of_second_segment)

    np.testing.assert_array_almost_equal(right_edge_position[0:2], point_on_second_segment)
    np.testing.assert_array_almost_equal(shifted_world_position[0:2],
                                         point_on_second_segment + np.array([-point_lateral_shift, 0.]))

    # Uncomment below to see the road structure
    # plt.plot(road_points[:, 0], road_points[:, 1], '-b')
    # plt.plot(right_edge_position[0], right_edge_position[1], '*c')
    # plt.plot(shifted_world_position[0], shifted_world_position[1], '*r')
    # plt.show()


def test_advanceToEndOfPlan_accurate(testable_map_api):
    navigation_plan = NavigationPlanMsg(road_ids=[1, 2])
    start_lon = 10.0
    first_road_length = testable_map_api._cached_map_model.get_road_data(1).longitudes[-1]
    second_road_length = testable_map_api._cached_map_model.get_road_data(2).longitudes[-1]
    roads_id, roads_len, roads_dist_to_end = testable_map_api.advance_to_end_of_plan(1, start_lon, navigation_plan)

    # Check that we got to the end of the plan
    assert roads_id == 2
    assert roads_len == second_road_length
    assert roads_dist_to_end == first_road_length + second_road_length - start_lon


def test_advanceOnPlan_accurate(testable_map_api):
    navigation_plan = NavigationPlanMsg(road_ids=[1, 2])
    path_total_len = MAP_INFLATION_FACTOR * (6 - 0.1)
    start_lon = 10.0
    first_road_length = testable_map_api._cached_map_model.get_road_data(1).longitudes[-1]
    advance_in_lon = path_total_len * 0.9

    road_id, lon = testable_map_api.advance_on_plan(1, start_lon, advance_in_lon, navigation_plan)

    # Check that the longitude on the second road equals to the advantage in lon, minus the first road's length
    assert road_id == 2
    assert lon == (start_lon + advance_in_lon - first_road_length)


def test_findClosestRoad_accurate(testable_map_api):
    point_close_to_road_1 = np.array([0.9, -0.1]) * MAP_INFLATION_FACTOR
    closest_lat, closest_lon, closest_id = testable_map_api._find_closest_road(point_close_to_road_1[0],
                                                                               point_close_to_road_1[1], [1, 2])

    # Check that closest road is 1 and (lat, lon) location is correct
    assert closest_id == 1
    assert np.math.isclose(closest_lon, point_close_to_road_1[0])
    assert np.math.isclose(closest_lat,
                           ROAD_WIDTH / 2 + point_close_to_road_1[1])  # center of lane + dist from center of lane

    point_close_to_road_2 = np.array([0.7, 1.1]) * MAP_INFLATION_FACTOR
    closest_lat, closest_lon, closest_id = testable_map_api._find_closest_road(point_close_to_road_2[0],
                                                                               point_close_to_road_2[1], [1, 2])

    # Check that closest road is 2 and (lat, lon) location is correct
    assert closest_id == 2
    assert np.math.isclose(closest_lon, MAP_INFLATION_FACTOR - point_close_to_road_2[0])
    assert np.math.isclose(closest_lat, (MAP_INFLATION_FACTOR + ROAD_WIDTH / 2)
                           - point_close_to_road_2[1])  # center of lane + dist from center of lane
