import pytest
import numpy as np
import pickle
from decision_making.src.global_constants import PG_SPLIT_PICKLE_FILE_NAME, PG_PICKLE_FILE_NAME, \
    ACCEL_TOWARDS_VEHICLE_SCENE_STATIC_PICKLE_FILE_NAME, ACCEL_TOWARDS_VEHICLE_SCENE_DYNAMIC_PICKLE_FILE_NAME
from decision_making.paths import Paths

from decision_making.test.utils.scene_static_utils import SceneStaticUtils

NUM_LANES = 3
LANE_WIDTH = 3.0
ROAD_WIDTH = LANE_WIDTH * NUM_LANES
MAP_INFLATION_FACTOR = 300.0
MAP_RESOLUTION = 1.0


@pytest.fixture
def scene_static_pg_no_split():
    return pickle.load(open(Paths.get_scene_static_absolute_path_filename(PG_PICKLE_FILE_NAME), 'rb'))


@pytest.fixture
def scene_static_pg_split():
    return pickle.load(open(Paths.get_scene_static_absolute_path_filename(PG_SPLIT_PICKLE_FILE_NAME), 'rb'))


@pytest.fixture
def scene_static_accel_towards_vehicle():
    return pickle.load(open(Paths.get_scene_static_absolute_path_filename(
                        ACCEL_TOWARDS_VEHICLE_SCENE_STATIC_PICKLE_FILE_NAME), 'rb'))


@pytest.fixture
def scene_dynamic_accel_towards_vehicle():
    return pickle.load(
        open(Paths.get_scene_dynamic_absolute_path_filename(
            ACCEL_TOWARDS_VEHICLE_SCENE_DYNAMIC_PICKLE_FILE_NAME), 'rb'))


@pytest.fixture()
def scene_static_testable():
    yield testable_scene_static_mock()


def testable_scene_static_mock():
    road_coordinates = list()
    road_coordinates.append(np.array([[x, 0] for x in np.linspace(0.0, 6.0, 150)]) * MAP_INFLATION_FACTOR)
    road_coordinates.append(np.array([[x, 0] for x in np.linspace(6.0, 10.0, 100)]) * MAP_INFLATION_FACTOR)

    # frame_origin = [32, 34]
    return SceneStaticUtils.create_scene_static_from_points(road_segment_ids=[1, 2],
                                                            num_lanes=NUM_LANES,
                                                            lane_width=LANE_WIDTH,
                                                            points_of_roads=road_coordinates)


@pytest.fixture()
def scene_static_short_testable():
    yield short_testable_scene_static_mock()


def short_testable_scene_static_mock():
    """
    This map was created for SceneModel that is limited by length of 1000 m
    """
    road_coordinates = list()
    road_coordinates.append(np.array([[x, 0] for x in np.linspace(0.0, 3.0, 75)]) * MAP_INFLATION_FACTOR)
    road_coordinates.append(np.array([[x, 0] for x in np.linspace(3.0, 5.0, 50)]) * MAP_INFLATION_FACTOR)

    return SceneStaticUtils.create_scene_static_from_points(road_segment_ids=[1, 2],
                                                            num_lanes=NUM_LANES,
                                                            lane_width=LANE_WIDTH,
                                                            points_of_roads=road_coordinates)
