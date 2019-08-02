import pytest
import numpy as np
import pickle
from decision_making.src.global_constants import PG_SPLIT_PICKLE_FILE_NAME, PG_PICKLE_FILE_NAME, \
    ACCEL_TOWARDS_VEHICLE_SCENE_STATIC_PICKLE_FILE_NAME, ACCEL_TOWARDS_VEHICLE_SCENE_DYNAMIC_PICKLE_FILE_NAME, \
    OVAL_WITH_SPLITS_PICKLE_FILE_NAME
from decision_making.paths import Paths
from decision_making.src.messages.scene_static_message import MapRoadSegmentType, LaneSegmentConnectivity, ManeuverType
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
    return pickle.load(open(Paths.get_scene_static_absolute_path_filename(ACCEL_TOWARDS_VEHICLE_SCENE_STATIC_PICKLE_FILE_NAME), 'rb'))

@pytest.fixture
def scene_static_oval_with_splits():
    return pickle.load(open(Paths.get_scene_static_absolute_path_filename(OVAL_WITH_SPLITS_PICKLE_FILE_NAME), 'rb'))


@pytest.fixture
def scene_dynamic_accel_towards_vehicle():
    return pickle.load(open(Paths.get_scene_dynamic_absolute_path_filename(ACCEL_TOWARDS_VEHICLE_SCENE_DYNAMIC_PICKLE_FILE_NAME), 'rb'))


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


@pytest.fixture()
def right_lane_split_scene_static():
    scene = short_testable_scene_static_mock()
    # disconnect right lane in 1st road segment
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[0].as_left_adjacent_lanes = []
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[0].e_Cnt_left_adjacent_lane_count = 0

    # add connection from 11 to 20
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].e_Cnt_downstream_lane_count = 2
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].as_downstream_lanes.append(LaneSegmentConnectivity(20,
                                                                                                                ManeuverType.RIGHT_SPLIT))
    # add upstream connection from 20 to 11
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[3].as_upstream_lanes = LaneSegmentConnectivity(11, ManeuverType.RIGHT_SPLIT)

    # change type of 2nd road segment
    scene.s_Data.s_SceneStaticBase.as_scene_road_segment[1].e_e_road_segment_type = MapRoadSegmentType.Intersection

    # delete right lane in 1st road segment
    del scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[0]
    scene.s_Data.s_SceneStaticBase.e_Cnt_num_lane_segments -= 1

    scene.s_Data.s_SceneStaticBase.as_scene_road_segment[0].a_i_lane_segment_ids = np.delete(
        scene.s_Data.s_SceneStaticBase.as_scene_road_segment[0].a_i_lane_segment_ids, 0)
    scene.s_Data.s_SceneStaticBase.as_scene_road_segment[0].e_Cnt_lane_segment_id_count -= 1

    del scene.s_Data.s_SceneStaticGeometry.as_scene_lane_segments[0]
    scene.s_Data.s_SceneStaticGeometry.e_Cnt_num_lane_segments -= 1

    return scene

@pytest.fixture()
def left_right_lane_split_scene_static():
    """
    Creates the following scenario:
    11 -> [22, 21, 20]
    :return:
    """

    scene = short_testable_scene_static_mock()
    # disconnect right lane in 1st road segment
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[0].as_left_adjacent_lanes = []
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[0].e_Cnt_left_adjacent_lane_count = 0
    # disconnect left lane in 1st road segment
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[2].as_right_adjacent_lanes = []
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[2].e_Cnt_right_adjacent_lane_count = 0
    # add connection from 11 to 20
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].e_Cnt_downstream_lane_count = 2
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].as_downstream_lanes.append(LaneSegmentConnectivity(20, ManeuverType.RIGHT_SPLIT))
    # add upstream connection from 20 to 11
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[3].as_upstream_lanes = LaneSegmentConnectivity(11, ManeuverType.RIGHT_SPLIT)
    # add connection from 11 to 22
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].e_Cnt_downstream_lane_count = 2
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].as_downstream_lanes.append(LaneSegmentConnectivity(22, ManeuverType.LEFT_SPLIT))
    # add upstream connection from 22 to 11
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[5].as_upstream_lanes = LaneSegmentConnectivity(11, ManeuverType.LEFT_SPLIT)
    # change type of 2nd road segment
    scene.s_Data.s_SceneStaticBase.as_scene_road_segment[1].e_e_road_segment_type = MapRoadSegmentType.Intersection
    # delete right lane in 1st road segment
    del scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[0]
    scene.s_Data.s_SceneStaticBase.e_Cnt_num_lane_segments -= 1
    scene.s_Data.s_SceneStaticBase.as_scene_road_segment[0].a_i_lane_segment_ids = np.delete(
        scene.s_Data.s_SceneStaticBase.as_scene_road_segment[0].a_i_lane_segment_ids, 0)
    scene.s_Data.s_SceneStaticBase.as_scene_road_segment[0].e_Cnt_lane_segment_id_count -= 1
    del scene.s_Data.s_SceneStaticGeometry.as_scene_lane_segments[0]
    scene.s_Data.s_SceneStaticGeometry.e_Cnt_num_lane_segments -= 1
    # delete left lane in 1st road segment
    del scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1]
    scene.s_Data.s_SceneStaticBase.e_Cnt_num_lane_segments -= 1
    scene.s_Data.s_SceneStaticBase.as_scene_road_segment[0].a_i_lane_segment_ids = np.delete(
        scene.s_Data.s_SceneStaticBase.as_scene_road_segment[0].a_i_lane_segment_ids, 1)
    scene.s_Data.s_SceneStaticBase.as_scene_road_segment[0].e_Cnt_lane_segment_id_count -= 1
    del scene.s_Data.s_SceneStaticGeometry.as_scene_lane_segments[1]
    scene.s_Data.s_SceneStaticGeometry.e_Cnt_num_lane_segments -= 1
    return scene
