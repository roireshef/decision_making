import timeit
from typing import List
import pytest

import numpy as np
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_SceneLaneSegmentGeometry import \
    TsSYSSceneLaneSegmentGeometry
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_SceneStaticGeometry import TsSYSSceneStaticGeometry
from decision_making.src.messages.scene_static_message import SceneStaticGeometry

MAX_NOMINAL_PATH_POINTS = 2000
MAX_MAP_NUM_LANES = 16
MAX_NOMINAL_PATH_POINT_FIELDS = 10
TOTAL_POINTS = MAX_NOMINAL_PATH_POINTS * MAX_MAP_NUM_LANES


class TsSYSSceneLaneSegmentGeometryMock(TsSYSSceneLaneSegmentGeometry):

    def __init__(self,
                 e_Cnt_nominal_path_point_count: int,
                 e_i_nominal_path_point_start_index: int,
                 e_i_lane_segment_id: int=0,
                 e_i_road_segment_id: int=0,
                 e_Cnt_left_boundary_points_count: int=0,
                 as_left_boundary_points: np.ndarray = None,
                 e_Cnt_right_boundary_points_count: int=0,
                 as_right_boundary_points: np.ndarray = None,
                 ):
        super(TsSYSSceneLaneSegmentGeometryMock, self).__init__()
        if as_left_boundary_points is None:
            as_left_boundary_points = []
        if as_right_boundary_points is None:
            as_right_boundary_points = []

        self.e_i_lane_segment_id = e_i_lane_segment_id
        self.e_i_road_segment_id = e_i_road_segment_id
        self.e_Cnt_nominal_path_point_count = e_Cnt_nominal_path_point_count
        self._dic["e_i_nominal_path_point_start_index"] = e_i_nominal_path_point_start_index
        self.e_Cnt_left_boundary_points_count = e_Cnt_left_boundary_points_count
        self.as_left_boundary_points = as_left_boundary_points
        self.e_Cnt_right_boundary_points_count = e_Cnt_right_boundary_points_count
        self.as_right_boundary_points = as_right_boundary_points

        if self.e_Cnt_right_boundary_points_count > 0:
            self.e_Cnt_right_boundary_points_count = np.zeros((
                e_Cnt_right_boundary_points_count, 2
            ))

        if self.e_Cnt_left_boundary_points_count > 0:
            self.e_Cnt_left_boundary_points_count = np.zeros((
                e_Cnt_left_boundary_points_count, 2
            ))


class TsSYSSceneStaticGeometryMock(TsSYSSceneStaticGeometry):

    def __init__(self,
                 e_Cnt_num_lane_segments: int,
                 as_scene_lane_segments: List[TsSYSSceneLaneSegmentGeometry],
                 a_nominal_path_points: np.ndarray=None
                 ):
        super(TsSYSSceneStaticGeometryMock, self).__init__()
        self.e_Cnt_num_lane_segments = e_Cnt_num_lane_segments
        self.as_scene_lane_segments = as_scene_lane_segments
        if a_nominal_path_points is None:
            a_nominal_path_points = np.zeros((
                MAX_NOMINAL_PATH_POINTS*MAX_MAP_NUM_LANES,
                MAX_NOMINAL_PATH_POINT_FIELDS,
            ))
        self.a_nominal_path_points = a_nominal_path_points


def create_scene_static_configurations(num_lanes, random_distribution=False):

    if random_distribution:
        rand_nums = np.random.rand(num_lanes)
        fracs = rand_nums / sum(rand_nums)
    else:
        fracs = np.ones(num_lanes) / num_lanes
    points_per_lane = np.floor(TOTAL_POINTS * fracs)

    all_lanes = []
    index = 0
    for i in range(num_lanes):
        curr_lane = TsSYSSceneLaneSegmentGeometryMock(int(points_per_lane[i]), int(index))
        index += points_per_lane[i]
        all_lanes.append(curr_lane)
    scene_static_mock_msg = TsSYSSceneStaticGeometryMock(num_lanes, all_lanes)
    return scene_static_mock_msg


@pytest.mark.skip(reason="This can be run optionally, instead of creating a MAT file every time all unit tests are run.")
def test_timings():
    """
    This function runs multiple combinations
    :return:
    """

    import scipy.io as sio

    global scene_static_mock_msg

    def callback_measured():
        global scene_static_mock_msg
        SceneStaticGeometry.deserialize(scene_static_mock_msg)

    def callback_create():
        global scene_static_mock_msg
        scene_static_mock_msg = create_scene_static_configurations(k, False)

    def callback_create_random():
        global scene_static_mock_msg
        scene_static_mock_msg = create_scene_static_configurations(k, True)

    max_lanes = 64
    num_repetitions = 10
    uniform_results = []
    random_results = []
    for k in np.arange(1, max_lanes, 16):

        times = timeit.repeat(setup=callback_create, stmt=callback_measured, repeat=num_repetitions, number=1)
        mean, std, max_time, min_time = np.mean(times), np.std(times), max(times), min(times)
        print("mean = {:.6f} ( std = {:.6f}, max = {:.6f} )".format(mean, std, max_time, min_time))
        uniform_results.append([k, mean, std, max_time, min_time])

        times = timeit.repeat(setup=callback_create_random, stmt=callback_measured, repeat=100, number=1)
        mean, std, max_time, min_time = np.mean(times), np.std(times), max(times), min(times)
        print("mean = {:.6f} ( std = {:.6f}, max = {:.6f} )".format(mean, std, max_time, min_time))
        random_results.append([k, mean, std, max_time, min_time])

    sio.savemat('results.mat', {'random_results': random_results, 'uniform_results': uniform_results})


def test_valid_conversion():

    num_lanes = 10

    scene_static_mock_msg = create_scene_static_configurations(num_lanes, True)

    scene_static_mock_msg.a_nominal_path_points = np.repeat(
        np.arange(TOTAL_POINTS).reshape(TOTAL_POINTS, 1),
        MAX_NOMINAL_PATH_POINT_FIELDS,
        axis=1)

    serialized_obj = SceneStaticGeometry.deserialize(scene_static_mock_msg)

    # Check sizes
    for lane in serialized_obj.as_scene_lane_segments:
        assert(lane.e_Cnt_nominal_path_point_count == lane.a_nominal_path_points.shape[0])

    # Check values
    for i in range(num_lanes):
        assert(scene_static_mock_msg.as_scene_lane_segments[i]._dic["e_i_nominal_path_point_start_index"]
               == serialized_obj.as_scene_lane_segments[i].a_nominal_path_points[0][0])


if __name__ == "__main__":
    test_valid_conversion()
    test_timings()
