import numpy as np
import pytest
from typing import List

from decision_making.src.mapping.scene_model import SceneModel
from decision_making.src.messages.scene_static_message import SceneStatic, NominalPathPoint
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.utils.map_utils import MapUtils

from decision_making.test.messages.static_scene_sample import scene_static_sample, NUM_ROAD_SEGS, NUM_LANE_SEGS, \
                                                              NUM_NOM_PATH_PTS, HORIZON_PERCEP_DIST, HALF_LANE_WIDTH, \
                                                              ROAD_SEG_ID, LANE_SEG_IDS, START_EAST_X_VAL, \
                                                              START_NORTH_Y_VAL, NAVIGATION_PLAN

@pytest.mark.skip('Remove for lane-based')
def test_get_road_rhs_frenet(scene_static: SceneStatic):
    # test_answer = MapUtils.get_road_rhs_frenet(road_id)
    # assert (test_answer == expected_answer)
    pass

@pytest.mark.skip('Remove, not completed')
def test_get_lookahead_frenet_frame(scene_static_sample: SceneStatic):
    # test_answer = MapUtils.get_lookahead_frenet_frame(lane_id, starting_lon, lookahead_dist, navigation_plan)
    # assert (test_answer == expected_answer)
    pass

def test_get_road_segment_id_from_lane_id(scene_static_sample: SceneStatic):
    SceneModel.get_instance().add_scene_static(scene_static_sample)
    test_answer = MapUtils.get_road_segment_id_from_lane_id(LANE_SEG_IDS[0])
    assert (test_answer == ROAD_SEG_ID)

def test_get_lane_ordinal(scene_static_sample: SceneStatic):
    SceneModel.get_instance().add_scene_static(scene_static_sample)
    test_answer = MapUtils.get_lane_ordinal(LANE_SEG_IDS[NUM_LANE_SEGS - 1])
    assert (test_answer == 0)

def test_get_lane_length(scene_static_sample: SceneStatic):
    SceneModel.get_instance().add_scene_static(scene_static_sample) 
    first_lane_nom_path_pts = scene_static_sample.s_Data.as_scene_lane_segment[0].a_nominal_path_points
    first_lane_last_nom_pt = first_lane_nom_path_pts[scene_static_sample.s_Data.as_scene_lane_segment[0].e_Cnt_nominal_path_point_count - 1]
    first_lane_length = first_lane_last_nom_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]
    test_answer = MapUtils.get_lane_length(LANE_SEG_IDS[0])
    assert (test_answer == first_lane_length)

@pytest.mark.skip('Remove, not completed')
def test_get_lane_frenet_frame(scene_static_sample: SceneStatic):
    # SceneModel.get_instance().add_scene_static(scene_static_sample)
    # test_answer = MapUtils.get_lane_frenet_frame(lane_id)
    # assert (test_answer == expected_answer)
    pass

@pytest.mark.skip('Remove, not completed')
def test_get_adjacent_lanes(scene_static_sample: SceneStatic):
    # SceneModel.get_instance().add_scene_static(scene_static_sample)
    # test_answer = MapUtils.get_adjacent_lanes(lane_id, relative_lane)
    # assert (test_answer == expected_answer)
    pass

def test_get_dist_to_lane_borders(scene_static_sample: SceneStatic):
    SceneModel.get_instance().add_scene_static(scene_static_sample)
    test_answer = MapUtils.get_dist_to_lane_borders(LANE_SEG_IDS[0], 0)
    assert (test_answer == (HALF_LANE_WIDTH, HALF_LANE_WIDTH))

def test_get_dist_to_road_borders(scene_static_sample: SceneStatic):
    SceneModel.get_instance().add_scene_static(scene_static_sample)
    test_answer = MapUtils.get_dist_to_road_borders(LANE_SEG_IDS[0], 0)
    assert (test_answer == (HALF_LANE_WIDTH, HALF_LANE_WIDTH))

def test_get_lane_width(scene_static_sample: SceneStatic):
    SceneModel.get_instance().add_scene_static(scene_static_sample)
    test_answer = MapUtils.get_lane_width(LANE_SEG_IDS[0], 0)
    assert (test_answer == (HALF_LANE_WIDTH * 2))

def test_get_upstream_lanes(scene_static_sample: SceneStatic):
    SceneModel.get_instance().add_scene_static(scene_static_sample)
    test_answer = MapUtils.get_upstream_lanes(LANE_SEG_IDS[NUM_LANE_SEGS - 1])
    assert (test_answer == [LANE_SEG_IDS[2]])

def test_get_downstream_lanes(scene_static_sample: SceneStatic):
    SceneModel.get_instance().add_scene_static(scene_static_sample)
    test_answer = MapUtils.get_downstream_lanes(LANE_SEG_IDS[0])
    assert (test_answer == [LANE_SEG_IDS[1]])

def test_get_lanes_ids_from_road_segment_id(scene_static_sample: SceneStatic):
    SceneModel.get_instance().add_scene_static(scene_static_sample)
    test_answer = MapUtils.get_lanes_ids_from_road_segment_id(ROAD_SEG_ID)
    assert (test_answer == LANE_SEG_IDS)
