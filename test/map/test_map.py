import matplotlib.pyplot as plt
from decision_making.src.global_constants import DM_MANAGER_NAME_FOR_LOGGING
from decision_making.src.map.map_api import *
from decision_making.src.map.naive_cache_map import NaiveCacheMap
from rte.python.logger.AV_logger import AV_Logger


def test_map():
    # TODO: add concrete assert statements

    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    map = NaiveCacheMap("../../resources/maps/testingGroundMap.bin", logger)
    points = np.array([[0, 51], [2, 51]]).transpose()
    shifted_points = map._shift_road_vector_in_lat(points, 1)
    road_id = 20
    lon = 40.0
    lat = 2.0
    navigation_plan = NavigationPlanMsg([20])

    # Short longitudinal lookahead
    lon = 4.0


    road_id, relative_lon, residual_lon = map._advance_road_coordinates_in_lon(road_id=road_id, start_lon=0.0,
                                                                               lon_step=lon,
                                                                               navigation_plan=navigation_plan)
    world_pnt, actual_lon_lookahead = map._convert_lat_lon_to_world(road_id, lat, lon, navigation_plan)

    # long longitudinal lookahead
    lon = 1000.0
    road_id, road_lon, residual_lon = map._advance_road_coordinates_in_lon(road_id=road_id, start_lon=0.0,
                                                                               lon_step=lon,
                                                                               navigation_plan=navigation_plan)
    world_pnt, actual_lon_lookahead = map._convert_lat_lon_to_world(road_id, lat, lon, navigation_plan)


    lat_dist, lon1 = map._convert_world_to_lat_lon_for_given_road(25.0, 48.0, road_id)

    road_info = map._cached_map_model.roads_data[20]
    road_points = road_info.points
    closest_lat, closest_lon, closest_id = map._find_closest_road(100., 200., [20])
    map.
    plt.plot(road_points[0,:], road_points[1,:])
    plt.plot(road_points[0, 0], road_points[1, 0], '*b')
    plt.plot(100.0, 200.0, '*r')
    plt.show()

    a = 2

test_map()
