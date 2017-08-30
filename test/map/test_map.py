import matplotlib.pyplot as plt
from decision_making.src.global_constants import DM_MANAGER_NAME_FOR_LOGGING
from decision_making.src.map.map_api import *
from decision_making.src.map.naive_cache_map import NaiveCacheMap
from rte.python.logger.AV_logger import AV_Logger


def test_map():
    # TODO: add concrete assert statements
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    map = NaiveCacheMap("../../resources/maps/testingGroundMap.bin", logger)
    road_id = 20
    lat = 2.0
    navigation_plan = NavigationPlanMsg([20])

    # Get road info
    road_info = map._cached_map_model.roads_data[20]
    road_points = road_info.points
    rightmost_edge_of_road_points = map._shift_road_points_in_latitude(road_points[0:2, :], -road_info.width / 2)
    leftmost_edge_of_road_points = map._shift_road_points_in_latitude(road_points[0:2, :], +road_info.width / 2)
    plt.plot(rightmost_edge_of_road_points[0, :], rightmost_edge_of_road_points[1, :], '-b')
    plt.plot(leftmost_edge_of_road_points[0, :], leftmost_edge_of_road_points[1, :], '-c')
    plt.plot(road_points[0, 0], road_points[1, 0], '*b')

    # Short longitudinal lookahead
    lon = 4.0
    road_id, relative_lon, residual_lon = map.advance_on_plan(initial_road_id=road_id, initial_lon=0.0,
                                                              desired_lon=lon,
                                                              navigation_plan=navigation_plan)
    world_pnt, actual_lon_lookahead = map.convert_road_to_global_coordinates(road_id, lat, lon, navigation_plan)

    # long longitudinal lookahead
    lon = 1000.0
    road_id, road_lon, residual_lon = map.advance_on_plan(initial_road_id=road_id, initial_lon=0.0,
                                                          desired_lon=lon,
                                                          navigation_plan=navigation_plan)
    world_pnt, actual_lon_lookahead = map.convert_road_to_global_coordinates(road_id, lat, lon, navigation_plan)



    # Find closest point on road to an arbitrary point in the world
    point_in_world = [100.0, 200.0, 0.0]
    lat_dist, lon1 = map._convert_global_to_road_coordinates(point_in_world[0], point_in_world[1], road_id)

    closest_lat, closest_lon, closest_id = map._find_closest_road(point_in_world[0], point_in_world[1], [20])
    closest_world_point, actual_lon_lookahead = map.convert_road_to_global_coordinates(closest_id, 0.0, closest_lon,
                                                                                       navigation_plan)
    plt.plot([closest_world_point[0], point_in_world[0]], [closest_world_point[1], point_in_world[1]], 'g')
    plt.plot(100.0, 200.0, '*r')
    plt.xlim([0.0, 1000.0])
    plt.ylim([0.0, 1000.0])
    plt.show()

