import pytest
import numpy as np
import pickle
from decision_making.src.global_constants import PG_SPLIT_PICKLE_FILE_NAME, PG_PICKLE_FILE_NAME, \
    ACCEL_TOWARDS_VEHICLE_SCENE_STATIC_PICKLE_FILE_NAME, ACCEL_TOWARDS_VEHICLE_SCENE_DYNAMIC_PICKLE_FILE_NAME, \
    OVAL_WITH_SPLITS_PICKLE_FILE_NAME
from decision_making.paths import Paths
from decision_making.src.messages.scene_static_message import MapRoadSegmentType, LaneSegmentConnectivity, ManeuverType, SceneRoadSegment
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
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].as_right_adjacent_lanes = []
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].e_Cnt_right_adjacent_lane_count = 0

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
def left_lane_split_scene_static():
    scene = short_testable_scene_static_mock()
    # disconnect left lane in 1st road segment
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].as_left_adjacent_lanes = []
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].e_Cnt_left_adjacent_lane_count = 0

    # add connection from 11 to 22
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].e_Cnt_downstream_lane_count = 2
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].as_downstream_lanes.append(LaneSegmentConnectivity(22,
                                                                                                                ManeuverType.LEFT_SPLIT))
    # add upstream connection from 22 to 11
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[5].as_upstream_lanes = LaneSegmentConnectivity(11, ManeuverType.LEFT_SPLIT)

    # change type of 2nd road segment
    scene.s_Data.s_SceneStaticBase.as_scene_road_segment[1].e_e_road_segment_type = MapRoadSegmentType.Intersection

    # delete left lane in 1st road segment
    del scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[2]
    scene.s_Data.s_SceneStaticBase.e_Cnt_num_lane_segments -= 1

    scene.s_Data.s_SceneStaticBase.as_scene_road_segment[0].a_i_lane_segment_ids = np.delete(
        scene.s_Data.s_SceneStaticBase.as_scene_road_segment[0].a_i_lane_segment_ids, 2)
    scene.s_Data.s_SceneStaticBase.as_scene_road_segment[0].e_Cnt_lane_segment_id_count -= 1

    del scene.s_Data.s_SceneStaticGeometry.as_scene_lane_segments[2]
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
    # disconnect left lane in 1st road segment
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].as_left_adjacent_lanes = []
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].e_Cnt_left_adjacent_lane_count = 0
    # disconnect right lane in 1st road segment
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].as_right_adjacent_lanes = []
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].e_Cnt_right_adjacent_lane_count = 0
    # add connection from 11 to 20
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].e_Cnt_downstream_lane_count = 2
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].as_downstream_lanes.append(LaneSegmentConnectivity(20,
                                                                                                                ManeuverType.RIGHT_SPLIT))
    # add upstream connection from 20 to 11
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[3].as_upstream_lanes = LaneSegmentConnectivity(11, ManeuverType.RIGHT_SPLIT)
    # add connection from 11 to 22
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].e_Cnt_downstream_lane_count = 2
    scene.s_Data.s_SceneStaticBase.as_scene_lane_segments[1].as_downstream_lanes.append(LaneSegmentConnectivity(22,
                                                                                                                ManeuverType.LEFT_SPLIT))
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

@pytest.fixture()
def scene_static_left_fork():
    scene = short_testable_scene_static_mock()
    ssb = scene.s_Data.s_SceneStaticBase

    # add a 3rd road segment
    ssb.as_scene_road_segment.append(SceneRoadSegment(3, 0, 1, np.array([30]), MapRoadSegmentType.Normal,
                                                                                 1, np.array([2]), 0, np.array([])))
    ssb.e_Cnt_num_road_segments += 1

    # make lane segment 22 become lane segment 30
    lane30 = ssb.as_scene_lane_segments[5]
    lane30.as_right_adjacent_lanes = []
    lane30.e_Cnt_right_adjacent_lane_count = 0
    lane30.e_i_lane_segment_id = 30
    lane30.e_i_road_segment_id = 3

    return scene

@pytest.fixture()
def scene_static_left_lane_ends():
    """
    Creates map where left lane suddenly ends
    12 -> _
    11 -> 21
    10 -> 20
    :return:
    """
    scene = short_testable_scene_static_mock()
    ssb = scene.s_Data.s_SceneStaticBase

    # delete lane 22
    del ssb.as_scene_lane_segments[5]
    #delete downstreams of lane 12
    ssb.as_scene_lane_segments[2].as_downstream_lanes = np.array([])
    ssb.as_scene_lane_segments[2].e_Cnt_downstream_lane_count = 0
    #delete lane 22 in road 2
    ssb.as_scene_road_segment[1].a_i_lane_segment_ids = np.array([20, 21])
    ssb.as_scene_road_segment[1].e_Cnt_lane_segment_id_count -= 1
    #delete left adjacent of lane 21
    ssb.as_scene_lane_segments[4].as_left_adjacent_lanes = []
    ssb.as_scene_lane_segments[4].e_Cnt_left_adjacent_lane_count = 0

    ssb.e_Cnt_num_lane_segments -= 1

    return scene

@pytest.fixture()
def scene_static_right_lane_ends():
    """
    Creates map where right lane suddenly ends
    12 -> 22
    11 -> 21
    10 -> _
    :return:
    """
    scene = short_testable_scene_static_mock()
    ssb = scene.s_Data.s_SceneStaticBase

    # delete lane 20
    del ssb.as_scene_lane_segments[3]
    #delete downstreams of lane 10
    ssb.as_scene_lane_segments[0].as_downstream_lanes = np.array([])
    ssb.as_scene_lane_segments[0].e_Cnt_downstream_lane_count = 0
    #delete lane 20 in road 2
    ssb.as_scene_road_segment[1].a_i_lane_segment_ids = np.array([21, 22])
    ssb.as_scene_road_segment[1].e_Cnt_lane_segment_id_count -= 1
    #delete right adjacent of lane 21
    ssb.as_scene_lane_segments[4].as_right_adjacent_lanes = []
    ssb.as_scene_lane_segments[4].e_Cnt_right_adjacent_lane_count = 0

    ssb.e_Cnt_num_lane_segments -= 1

    return scene

@pytest.fixture(scope='function')
def scene_static_lane_split_on_right_ends():
    """
    Creates map where lane splits to right and then ends shortly after
    202 -> 212 -> 222 -> 232 -> 242 -> 252 -> 262 -> 272 -> 282 -> 292
    201 -> 211 -> 221 -> 231 -> 241 -> 251 -> 261 -> 271 -> 281 -> 291
              `-> 220 -> 230
    """
    scene_static = scene_static_pg_split()
    scene_static_base = scene_static.s_Data.s_SceneStaticBase
    scene_static_geometry = scene_static.s_Data.s_SceneStaticGeometry

    # Disconnect right lane from center lane, except for lanes 221 and 231
    for lane_segment in scene_static_base.as_scene_lane_segments[1::3]:
        if lane_segment.e_i_lane_segment_id not in [221, 231]:
            lane_segment.as_right_adjacent_lanes = []
            lane_segment.e_Cnt_right_adjacent_lane_count = 0

    # Connect 211 to 220 and vice versa
    scene_static_base.as_scene_lane_segments[4].as_downstream_lanes.append(LaneSegmentConnectivity(220, ManeuverType.RIGHT_SPLIT))
    scene_static_base.as_scene_lane_segments[4].e_Cnt_downstream_lane_count += 1

    scene_static_base.as_scene_lane_segments[6].as_upstream_lanes = LaneSegmentConnectivity(211, ManeuverType.RIGHT_SPLIT)

    # Disconnect 240 from 230
    scene_static_base.as_scene_lane_segments[9].as_downstream_lanes = []
    scene_static_base.as_scene_lane_segments[9].e_Cnt_downstream_lane_count = 0

    # Change third road segment type
    scene_static_base.as_scene_road_segment[2].e_e_road_segment_type = MapRoadSegmentType.Intersection

    # Delete all right lanes, except for lanes 220 and 230
    for lane_segment_index, road_segment_index in zip([27, 24, 21, 18, 15, 12, 3, 0], [9, 8, 7, 6, 5, 4, 1, 0]):
        # Remove lane segment from road segment info
        lane_segment_id_mask = scene_static_base.as_scene_road_segment[road_segment_index].a_i_lane_segment_ids != \
                               scene_static_base.as_scene_lane_segments[lane_segment_index].e_i_lane_segment_id

        scene_static_base.as_scene_road_segment[road_segment_index].a_i_lane_segment_ids = \
            scene_static_base.as_scene_road_segment[road_segment_index].a_i_lane_segment_ids[lane_segment_id_mask]

        scene_static_base.as_scene_road_segment[road_segment_index].e_Cnt_lane_segment_id_count -= 1

        # Delete lane segment base data
        del scene_static_base.as_scene_lane_segments[lane_segment_index]
        scene_static_base.e_Cnt_num_lane_segments -= 1

        # Delete lane segment geometry data
        del scene_static_geometry.as_scene_lane_segments[lane_segment_index]
        scene_static_geometry.e_Cnt_num_lane_segments -= 1

    return scene_static

@pytest.fixture(scope='function')
def scene_static_lane_split_on_left_ends():
    """
    Creates map where lane splits to left and then ends shortly after
              ,-> 222 -> 232
    201 -> 211 -> 221 -> 231 -> 241 -> 251 -> 261 -> 271 -> 281 -> 291
    200 -> 210 -> 220 -> 230 -> 240 -> 250 -> 260 -> 270 -> 280 -> 290
    """
    scene_static = scene_static_pg_split()
    scene_static_base = scene_static.s_Data.s_SceneStaticBase
    scene_static_geometry = scene_static.s_Data.s_SceneStaticGeometry

    # Disconnect left lane from center lane, except for lanes 221 and 231
    for lane_segment in scene_static_base.as_scene_lane_segments[1::3]:
        if lane_segment.e_i_lane_segment_id not in [221, 231]:
            lane_segment.as_left_adjacent_lanes = []
            lane_segment.e_Cnt_left_adjacent_lane_count = 0

    # Connect 211 to 222 and vice versa
    scene_static_base.as_scene_lane_segments[4].as_downstream_lanes.append(LaneSegmentConnectivity(222, ManeuverType.LEFT_SPLIT))
    scene_static_base.as_scene_lane_segments[4].e_Cnt_downstream_lane_count += 1

    scene_static_base.as_scene_lane_segments[8].as_upstream_lanes = LaneSegmentConnectivity(211, ManeuverType.LEFT_SPLIT)

    # Disconnect 242 from 232
    scene_static_base.as_scene_lane_segments[11].as_downstream_lanes = []
    scene_static_base.as_scene_lane_segments[11].e_Cnt_downstream_lane_count = 0

    # Change third road segment type
    scene_static_base.as_scene_road_segment[2].e_e_road_segment_type = MapRoadSegmentType.Intersection

    # Delete all left lanes, except for lanes 220 and 230
    for lane_segment_index, road_segment_index in zip([29, 26, 23, 20, 17, 14, 5, 2], [9, 8, 7, 6, 5, 4, 1, 0]):
        # Remove lane segment from road segment info
        lane_segment_id_mask = scene_static_base.as_scene_road_segment[road_segment_index].a_i_lane_segment_ids != \
                               scene_static_base.as_scene_lane_segments[lane_segment_index].e_i_lane_segment_id

        scene_static_base.as_scene_road_segment[road_segment_index].a_i_lane_segment_ids = \
            scene_static_base.as_scene_road_segment[road_segment_index].a_i_lane_segment_ids[lane_segment_id_mask]

        scene_static_base.as_scene_road_segment[road_segment_index].e_Cnt_lane_segment_id_count -= 1

        # Delete lane segment base data
        del scene_static_base.as_scene_lane_segments[lane_segment_index]
        scene_static_base.e_Cnt_num_lane_segments -= 1

        # Delete lane segment geometry data
        del scene_static_geometry.as_scene_lane_segments[lane_segment_index]
        scene_static_geometry.e_Cnt_num_lane_segments -= 1

    return scene_static

@pytest.fixture(scope='function')
def scene_static_lane_splits_on_left_and_right_end():
    """
    Creates map where lane splits to left and right but then ends shortly after
              ,-> 222 -> 232
    201 -> 211 -> 221 -> 231 -> 241 -> 251 -> 261 -> 271 -> 281 -> 291
              `-> 220 -> 230
    """
    scene_static = scene_static_pg_split()
    scene_static_base = scene_static.s_Data.s_SceneStaticBase
    scene_static_geometry = scene_static.s_Data.s_SceneStaticGeometry

    # Disconnect right and left lanes from center lane, except for lanes 221 and 231
    for lane_segment in scene_static_base.as_scene_lane_segments[1::3]:
        if lane_segment.e_i_lane_segment_id not in [221, 231]:
            lane_segment.as_right_adjacent_lanes = []
            lane_segment.e_Cnt_right_adjacent_lane_count = 0

            lane_segment.as_left_adjacent_lanes = []
            lane_segment.e_Cnt_left_adjacent_lane_count = 0

    # Connect 211 to 220 and 222 and vice versa
    scene_static_base.as_scene_lane_segments[4].as_downstream_lanes.append(LaneSegmentConnectivity(220, ManeuverType.RIGHT_SPLIT))
    scene_static_base.as_scene_lane_segments[4].as_downstream_lanes.append(LaneSegmentConnectivity(222, ManeuverType.LEFT_SPLIT))
    scene_static_base.as_scene_lane_segments[4].e_Cnt_downstream_lane_count += 2

    scene_static_base.as_scene_lane_segments[6].as_upstream_lanes = LaneSegmentConnectivity(211, ManeuverType.RIGHT_SPLIT)
    scene_static_base.as_scene_lane_segments[8].as_upstream_lanes = LaneSegmentConnectivity(211, ManeuverType.LEFT_SPLIT)

    # Disconnect 240 from 230
    scene_static_base.as_scene_lane_segments[9].as_downstream_lanes = []
    scene_static_base.as_scene_lane_segments[9].e_Cnt_downstream_lane_count = 0

    # Disconnect 242 from 232
    scene_static_base.as_scene_lane_segments[11].as_downstream_lanes = []
    scene_static_base.as_scene_lane_segments[11].e_Cnt_downstream_lane_count = 0

    # Change third road segment type
    scene_static_base.as_scene_road_segment[2].e_e_road_segment_type = MapRoadSegmentType.Intersection

    # Delete all right and left lanes, except for lanes 220, 222, 230, and 232
    for lane_segment_index, road_segment_index in zip([29, 27, 26, 24, 23, 21, 20, 18, 17, 15, 14, 12, 5, 3, 2, 0],
                                                      [9, 8, 7, 6, 5, 4, 1, 0]):
        # Remove lane segment from road segment info
        lane_segment_id_mask = scene_static_base.as_scene_road_segment[road_segment_index].a_i_lane_segment_ids != \
                               scene_static_base.as_scene_lane_segments[lane_segment_index].e_i_lane_segment_id

        scene_static_base.as_scene_road_segment[road_segment_index].a_i_lane_segment_ids = \
            scene_static_base.as_scene_road_segment[road_segment_index].a_i_lane_segment_ids[lane_segment_id_mask]

        scene_static_base.as_scene_road_segment[road_segment_index].e_Cnt_lane_segment_id_count -= 1

        # Delete lane segment base data
        del scene_static_base.as_scene_lane_segments[lane_segment_index]
        scene_static_base.e_Cnt_num_lane_segments -= 1

        # Delete lane segment geometry data
        del scene_static_geometry.as_scene_lane_segments[lane_segment_index]
        scene_static_geometry.e_Cnt_num_lane_segments -= 1

    return scene_static
