from decision_making.src.mapping.service.map_service import MapService
from decision_making.test.mapping.model.testable_map_fixtures import *
import numpy as np

from decision_making.test.mapping.model.testable_map_fixtures import MAP_INFLATION_FACTOR


def test_shiftRoadVectorInLatitude_simpleRoadShift1MRight_accurateShift(testable_map_api):
    points = np.array([[0, 0], [1, -1], [1, -2]])
    shift = -1
    shifted_points = testable_map_api._shift_road_points(points, shift)
    expected_shifted_points = np.array([[-1 / np.sqrt(2), -1 / np.sqrt(2)], [0, -1], [0, -2]])

    np.testing.assert_array_almost_equal(shifted_points, expected_shifted_points)


def test_advanceToEndOfPlan_twoRoadsStartFromMiddleOfFirst_rightRoadAndDistances(testable_map_api, navigation_fixture):
    navigation_plan = navigation_fixture
    start_lon = 10.0
    first_road_length = testable_map_api._cached_map_model.get_road_data(1).length
    second_road_length = testable_map_api._cached_map_model.get_road_data(2).length
    roads_id, roads_len, roads_dist_to_end = testable_map_api.advance_to_end_of_plan(1, start_lon, navigation_plan)

    # Check that we got to the end of the plan
    assert roads_id == 2
    assert roads_len == second_road_length
    assert roads_dist_to_end == first_road_length + second_road_length - start_lon


# def test_advanceOnPlan_twoRoadsStartFromMiddleOfFirst_rightRoadAndDistances(testable_map_api, navigation_fixture):
#     navigation_plan = navigation_fixture
#     path_total_len = MAP_INFLATION_FACTOR * (6 - 0.1)
#     start_lon = 10.0
#     first_road_length = testable_map_api._cached_map_model.get_road_data(1).length
#     advance_in_lon = path_total_len * 0.9
#
#     road_id, lon = testable_map_api.advance_on_plan(1, start_lon, advance_in_lon, navigation_plan)
#
#     # Check that the longitude on the second road equals to the advantage in lon, minus the first road's length
#     assert road_id == 2
#     assert lon == (start_lon + advance_in_lon - first_road_length)


def test_findClosestRoad_twoRoads_firstRoadIsClosest(testable_map_api):
    point_close_to_road_1 = np.array([0.9, -0.1]) * MAP_INFLATION_FACTOR
    closest_id = testable_map_api._find_closest_road(point_close_to_road_1[0], point_close_to_road_1[1], [1, 2])

    # Check that closest road is 1 and (lat, lon) location is correct
    assert closest_id == 1
    # assert np.math.isclose(closest_lon, point_close_to_road_1[0])
    # assert np.math.isclose(closest_lat,
    #                        ROAD_WIDTH / 2 + point_close_to_road_1[1])  # center of lane + dist from center of lane


# def test_findClosestRoad_twoRoads_secondRoadIsClosest(testable_map_api):
#     point_close_to_road_2 = np.array([0.7, 1.1]) * MAP_INFLATION_FACTOR
#     closest_id = testable_map_api._find_closest_road(point_close_to_road_2[0], point_close_to_road_2[1], [1, 2])
#
#     # Check that closest road is 2 and (lat, lon) location is correct
#     assert closest_id == 2
#     # assert np.math.isclose(closest_lon, MAP_INFLATION_FACTOR - point_close_to_road_2[0])
#     # assert np.math.isclose(closest_lat, (MAP_INFLATION_FACTOR + ROAD_WIDTH / 2)
#     #                        - point_close_to_road_2[1])  # center of lane + dist from center of lane

# def test_convertGlobalToRoadCoordinates_funnelPoints_CorrectLat():
#     """
#     Funnel points are a special case of global point which lie in between two normals of adjacent segments.
#     Therefore the projection of the point onto one of the segments should be handled differently
#     """
#     demo_map = MapService.get_instance()
#     road_id = 20
#     yaw = 0
#
#     #Funnel points
#     x = 499.8281122
#     y = -186.0180083
#     lon, lat, yaw = demo_map._convert_global_to_road_coordinates(x, y, yaw, road_id)
#     assert(np.isclose(lat, 2.51888865033))
#
#     #Non-funnel points
#     x = 1103.863087
#     y = -2.047997531
#     lon, lat, yaw = demo_map._convert_global_to_road_coordinates(x, y, yaw, road_id)
#     assert(np.isclose(lat, 3.58905716369))
