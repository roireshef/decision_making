from scipy import interpolate
import numpy as np
import pytest

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.mapping.model.map_api import MapAPI
from decision_making.src.mapping.transformations.geometry_utils import CartesianFrame
from decision_making.test.mapping.model.map_model_utils import TestMapModelUtils
from rte.python.logger.AV_logger import AV_Logger

NUM_LANES = 3
LANE_WIDTH = 3.0
ROAD_WIDTH = LANE_WIDTH * NUM_LANES
MAP_INFLATION_FACTOR = 300.0
MAP_RESOLUTION = 1.0


@pytest.fixture()
def simple_testable_map_api():
    yield simple_map_api_mock()


def simple_map_api_mock():
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
    frame_origin = [32, 34]
    test_map_model = TestMapModelUtils.create_road_map_from_coordinates(points_of_roads=road_coordinates,
                                                                        road_id=[1, 2],
                                                                        road_name=['def1', 'def2'],
                                                                        lanes_num=[NUM_LANES, NUM_LANES],
                                                                        lane_width=[LANE_WIDTH, LANE_WIDTH],
                                                                        frame_origin=frame_origin)

    return MapAPI(map_model=test_map_model, logger=AV_Logger.get_logger("map_api_mock"))


@pytest.fixture()
def testable_map_api():
    yield map_api_mock()


def map_api_mock():
    # # Create a rectangle test map
    # road_coordinates_1 = np.array([[0., 0.],
    #                                [1., 0.],
    #                                [2., 0.],
    #                                [2., 0.5],
    #                                [2., 1.],
    #                                [1., 1.]]) * MAP_INFLATION_FACTOR
    #
    # road_coordinates_2 = np.array([[1., 1.],
    #                                [0., 1.],
    #                                [0., 0.5],
    #                                [0., 0.1]]) * MAP_INFLATION_FACTOR
    #
    # # Resample road_coordinates_1 & road_coordinates_2
    # road_coordinates_1_s = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(road_coordinates_1, axis=0), axis=1)])
    # road_coordinates_1_arclen = road_coordinates_1_s[-1]
    # road_coordinates_1_s_new = np.linspace(0.0, road_coordinates_1_arclen,
    #                                        int(road_coordinates_1_arclen / MAP_RESOLUTION) + 1)
    # road_coordinates_1_x = interpolate.interp1d(road_coordinates_1_s, road_coordinates_1[:, 0])
    # road_coordinates_1_y = interpolate.interp1d(road_coordinates_1_s, road_coordinates_1[:, 1])
    # road_coordinates_1_resampled = np.c_[
    #     road_coordinates_1_x(road_coordinates_1_s_new), road_coordinates_1_y(road_coordinates_1_s_new)]
    #
    # road_coordinates_2_s = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(road_coordinates_2, axis=0), axis=1)])
    # road_coordinates_2_arclen = road_coordinates_2_s[-1]
    # road_coordinates_2_s_new = np.linspace(0.0, road_coordinates_2_arclen,
    #                                        int(road_coordinates_2_arclen / MAP_RESOLUTION) + 1)
    # road_coordinates_2_x = interpolate.interp1d(road_coordinates_2_s, road_coordinates_2[:, 0])
    # road_coordinates_2_y = interpolate.interp1d(road_coordinates_2_s, road_coordinates_2[:, 1])
    # road_coordinates_2_resampled = np.c_[
    #     road_coordinates_2_x(road_coordinates_2_s_new), road_coordinates_2_y(road_coordinates_2_s_new)]
    #
    # road_coordinates = list()
    # road_coordinates.append(road_coordinates_1_resampled)
    # road_coordinates.append(road_coordinates_2_resampled)

    road_coordinates = list()
    road_coordinates.append(np.array([[x, 0] for x in np.linspace(0.0, 6.0, 150)]) * MAP_INFLATION_FACTOR)
    road_coordinates.append(np.array([[x, 0] for x in np.linspace(6.0, 10.0, 100)]) * MAP_INFLATION_FACTOR)

    frame_origin = [32, 34]
    test_map_model = TestMapModelUtils.create_road_map_from_coordinates(points_of_roads=road_coordinates,
                                                                        road_id=[1, 2],
                                                                        road_name=['def1', 'def2'],
                                                                        lanes_num=[NUM_LANES, NUM_LANES],
                                                                        lane_width=[LANE_WIDTH, LANE_WIDTH],
                                                                        frame_origin=frame_origin)

    return MapAPI(map_model=test_map_model, logger=AV_Logger.get_logger("map_api_mock"))


@pytest.fixture()
def short_testable_map_api():
    yield short_map_api_mock()


def short_map_api_mock():
    """
    This map was created for SceneModel that is limited by length of 1000 m
    """
    road_coordinates = list()
    road_coordinates.append(np.array([[x, 0] for x in np.linspace(0.0, 3.0, 75)]) * MAP_INFLATION_FACTOR)
    road_coordinates.append(np.array([[x, 0] for x in np.linspace(3.0, 5.0, 50)]) * MAP_INFLATION_FACTOR)

    frame_origin = [32, 34]
    test_map_model = TestMapModelUtils.create_road_map_from_coordinates(points_of_roads=road_coordinates,
                                                                        road_id=[1, 2],
                                                                        road_name=['def1', 'def2'],
                                                                        lanes_num=[NUM_LANES, NUM_LANES],
                                                                        lane_width=[LANE_WIDTH, LANE_WIDTH],
                                                                        frame_origin=frame_origin)

    return MapAPI(map_model=test_map_model, logger=AV_Logger.get_logger("short_map_api_mock"))


@pytest.fixture()
def navigation_fixture():
    yield NavigationPlanMsg(road_ids=np.array([1, 2]))


@pytest.fixture()
def navigation_fixture_for_proving_ground():
    yield NavigationPlanMsg(road_ids=np.array([20]))
