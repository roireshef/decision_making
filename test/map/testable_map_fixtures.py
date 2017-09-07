import pytest
import numpy as np

from decision_making.src.map.naive_cache_map import NaiveCacheMap
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.test.map.map_model_utils import TestMapModelUtils
from rte.python.logger.AV_logger import AV_Logger

NUM_LANES = 3
LANE_WIDTH = 3.0
ROAD_WIDTH = LANE_WIDTH * NUM_LANES
MAP_INFLATION_FACTOR = 300.0

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

    yield NaiveCacheMap(map_model=test_map_model, logger=AV_Logger.get_logger("testable_map_api"))

@pytest.fixture()
def navigation_fixture():
    yield NavigationPlanMsg(road_ids=np.array([1, 2]))