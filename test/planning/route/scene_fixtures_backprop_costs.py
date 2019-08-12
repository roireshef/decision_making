import numpy as np
import pickle
import pytest
from typing import List, Dict, Tuple

from decision_making.paths import Paths
from decision_making.src.global_constants import PG_SPLIT_PICKLE_FILE_NAME, ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR

from decision_making.src.messages.route_plan_message import DataRoutePlan, RoutePlanLaneSegment
from decision_making.src.messages.scene_static_enums import LaneConstructionType,\
    RoutePlanLaneSegmentAttr, LaneMappingStatusType, GMAuthorityType, MapLaneDirection,\
    ManeuverType
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import EgoState, ObjectSize
from decision_making.test.planning.route.scene_static_publisher import SceneStaticPublisher
from decision_making.src.planning.types import LaneSegmentID
from decision_making.src.utils.map_utils import MapUtils


class RoutePlanTestData:
    def __init__(self, scene_static: SceneStatic, expected_output: DataRoutePlan):
        self.scene_static = scene_static
        self.expected_output = expected_output


class TakeOverTestData:
    def __init__(self, scene_static: SceneStatic, route_plan_data: DataRoutePlan, ego_state: EgoState, expected_takeover: bool):
        self.scene_static = scene_static
        self.route_plan_data = route_plan_data
        self.ego_state = ego_state
        self.expected_takeover = expected_takeover


def default_route_plan_for_PG_split_file() -> DataRoutePlan:
    return DataRoutePlan(e_b_is_valid=True,
                         e_Cnt_num_road_segments = 10,
                         a_i_road_segment_ids = np.arange(20, 30),
                         a_Cnt_num_lane_segments = np.full(10, 3),
                         as_route_plan_lane_segments = [[RoutePlanLaneSegment(e_i_lane_segment_id=lane_segment_id_base + lane_number,
                                                                              e_cst_lane_occupancy_cost=0.0,
                                                                              e_cst_lane_end_cost=0.0) for lane_number in [0, 1, 2]]
                                                        for lane_segment_id_base in np.arange(200, 300, 10)])


IsLaneAttributeActive = bool
LaneAttribute = int  # actually, LaneMappingStatusType, MapLaneDirection, GMAuthorityType, or LaneConstructionType
LaneAttributeConfidence = float
LaneAttributeModification = Tuple[IsLaneAttributeActive, RoutePlanLaneSegmentAttr, LaneAttribute, LaneAttributeConfidence]
LaneAttributeModifications = Dict[LaneSegmentID, List[LaneAttributeModification]]


def modify_default_lane_attributes(lane_attribute_modifications: LaneAttributeModifications = None) -> SceneStatic:
    if lane_attribute_modifications is None:
        lane_attribute_modifications = {}

    # Load saved scene static message
    scene_static = pickle.load(open(Paths.get_scene_static_absolute_path_filename(PG_SPLIT_PICKLE_FILE_NAME), 'rb'))

    for lane_segment in scene_static.s_Data.s_SceneStaticBase.as_scene_lane_segments:
        # Check for lane attribute modifications
        if lane_segment.e_i_lane_segment_id in lane_attribute_modifications:
            # Default lane attribute values
            num_active_lane_attributes = 4
            active_lane_attribute_indices = np.array([0, 1, 2, 3])
            lane_attributes = np.array([LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap.value,
                                        GMAuthorityType.CeSYS_e_GMAuthorityType_None.value,
                                        LaneConstructionType.CeSYS_e_LaneConstructionType_Normal.value,
                                        MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle.value])
            lane_attribute_confidences = np.ones(4)

            for lane_attribute_modification in lane_attribute_modifications[lane_segment.e_i_lane_segment_id]:
                if lane_attribute_modification[0] is True:
                    lane_attributes[lane_attribute_modification[1]] = lane_attribute_modification[2]
                    lane_attribute_confidences[lane_attribute_modification[1]] = lane_attribute_modification[3]
                else:
                    active_lane_attribute_indices = np.delete(active_lane_attribute_indices, lane_attribute_modification[1])
                    num_active_lane_attributes -= 1

            lane_segment.e_Cnt_num_active_lane_attributes = num_active_lane_attributes
            lane_segment.a_i_active_lane_attribute_indices = active_lane_attribute_indices
            lane_segment.a_cmp_lane_attributes = lane_attributes
            lane_segment.a_cmp_lane_attribute_confidences = lane_attribute_confidences

    return scene_static


@pytest.fixture(scope='function', params=["scene_one",
                                          "scene_two",
                                          "scene_three"])
def construction_scene_and_expected_output(request):
    # Set Default Expected Output
    expected_output = default_route_plan_for_PG_split_file()

    scene_static = pickle.load(open(Paths.get_scene_static_absolute_path_filename(PG_SPLIT_PICKLE_FILE_NAME), 'rb'))

    # Define Lane Modifications and Modify Expected Outputs
    if request.param is "scene_one":
        lane_modifications = {211: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneConstructionType.CeSYS_e_LaneConstructionType_Blocked.value,
                                     1.0)],
                              212: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneConstructionType.CeSYS_e_LaneConstructionType_Blocked.value,
                                     1.0)]}

        # Road Segment 20
        expected_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost = 1.0

        # Road Segment 21
        expected_output.as_route_plan_lane_segments[1][1].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0

        expected_output.as_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0

    elif request.param is "scene_two":
        lane_modifications = {291: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneConstructionType.CeSYS_e_LaneConstructionType_Blocked.value,
                                     1.0)],
                              292: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneConstructionType.CeSYS_e_LaneConstructionType_Blocked.value,
                                     1.0)]}

        # Road Segment 28
        expected_output.as_route_plan_lane_segments[8][1].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[8][2].e_cst_lane_end_cost = 1.0

        # Road Segment 29
        expected_output.as_route_plan_lane_segments[9][1].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[9][2].e_cst_lane_occupancy_cost = 1.0

        expected_output.as_route_plan_lane_segments[9][1].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[9][2].e_cst_lane_end_cost = 1.0

        for i in reversed(range(8)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i+1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR**MapUtils.get_lane(expected_output.as_route_plan_lane_segments[i+1][j].e_i_lane_segment_id).e_l_length

    elif request.param is "scene_three":
        lane_modifications = {251: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneConstructionType.CeSYS_e_LaneConstructionType_Blocked.value,
                                     1.0)],
                              252: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneConstructionType.CeSYS_e_LaneConstructionType_Blocked.value,
                                     1.0)]}

        # Road Segment 24
        expected_output.as_route_plan_lane_segments[4][1].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[4][2].e_cst_lane_end_cost = 1.0

        # Road Segment 25
        expected_output.as_route_plan_lane_segments[5][1].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[5][2].e_cst_lane_occupancy_cost = 1.0

        expected_output.as_route_plan_lane_segments[5][1].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[5][2].e_cst_lane_end_cost = 1.0

        for i in reversed(range(4)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

    else:
        lane_modifications = {}

    # for road_segment in expected_output.as_route_plan_lane_segments:
    #     for lane_segment in road_segment:
    #         print("lane_segment_id     = ", lane_segment.e_i_lane_segment_id)
    #         print("lane_occupancy_cost = ", lane_segment.e_cst_lane_occupancy_cost)
    #         print("lane_end_cost       = ", lane_segment.e_cst_lane_end_cost, "\n")

    #     print("==========================\n")

    return RoutePlanTestData(scene_static=modify_default_lane_attributes(lane_modifications),
                             expected_output=expected_output)


@pytest.fixture(scope='function', params=["scene_one",
                                          "scene_two"])
def map_scene_and_expected_output(request):
    # Set Default Expected Output
    expected_output = default_route_plan_for_PG_split_file()
    scene_static = pickle.load(open(Paths.get_scene_static_absolute_path_filename(PG_SPLIT_PICKLE_FILE_NAME), 'rb'))

    # Define Lane Modifications and Modify Expected Outputs
    if request.param is "scene_one":
        lane_modifications = {212: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus.value,
                                     LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_NotMapped.value,
                                     1.0)]}

        expected_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0

    elif request.param is "scene_two":
        lane_modifications = {290: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus.value,
                                     LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_NotMapped.value,
                                     1.0)],
                              291: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus.value,
                                     LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_NotMapped.value,
                                     1.0)],
                              292: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus.value,
                                     LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_NotMapped.value,
                                     1.0)]}

        # Road Segment 28
        expected_output.as_route_plan_lane_segments[8][0].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[8][1].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[8][2].e_cst_lane_end_cost = 1.0

        # Road Segment 29
        expected_output.as_route_plan_lane_segments[9][0].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[9][1].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[9][2].e_cst_lane_occupancy_cost = 1.0

        expected_output.as_route_plan_lane_segments[9][0].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[9][1].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[9][2].e_cst_lane_end_cost = 1.0

        for i in reversed(range(8)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

    else:
        lane_modifications = {}

    return RoutePlanTestData(scene_static=modify_default_lane_attributes(lane_modifications),
                             expected_output=expected_output)


@pytest.fixture(scope='function', params=["scene_one",
                                          "scene_two",
                                          "scene_three",
                                          "scene_four",
                                          "scene_five",
                                          "scene_six",
                                          "scene_seven",
                                          "scene_eight",
                                          "scene_nine",
                                          "scene_ten"])
def gmfa_scene_and_expected_output(request):
    # Set Default Expected Output
    expected_output = default_route_plan_for_PG_split_file()

    # Define Lane Modifications and Modify Expected Outputs
    if request.param is "scene_one":
        lane_modifications = {212: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_RoadConstruction.value,
                                     1.0)]}

        expected_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_two":
        lane_modifications = {222: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_BadRoadCondition.value,
                                     1.0)]}

        expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[2][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[2][2].e_cst_lane_end_cost = 1.0
        for i in reversed(range(1)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

    elif request.param is "scene_three":
        lane_modifications = {232: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_ComplexRoad.value,
                                     1.0)]}

        expected_output.as_route_plan_lane_segments[2][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[3][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[3][2].e_cst_lane_end_cost = 1.0
        for i in reversed(range(2)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

    elif request.param is "scene_four":
        lane_modifications = {242: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_MovableBarriers.value,
                                     1.0)]}

        expected_output.as_route_plan_lane_segments[3][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[4][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[4][2].e_cst_lane_end_cost = 1.0
        for i in reversed(range(3)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

    elif request.param is "scene_five":
        lane_modifications = {252: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_BidirectionalFreew.value,
                                     1.0)]}

        expected_output.as_route_plan_lane_segments[4][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[5][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[5][2].e_cst_lane_end_cost = 1.0
        for i in reversed(range(4)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

    elif request.param is "scene_six":
        lane_modifications = {262: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_HighCrossTrackSlope.value,
                                     1.0)]}

        expected_output.as_route_plan_lane_segments[5][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[6][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[6][2].e_cst_lane_end_cost = 1.0
        for i in reversed(range(5)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

    elif request.param is "scene_seven":
        lane_modifications = {272: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_HighAlongTrackSlope.value,
                                     1.0)]}

        expected_output.as_route_plan_lane_segments[6][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[7][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[7][2].e_cst_lane_end_cost = 1.0
        for i in reversed(range(6)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

    elif request.param is "scene_eight":
        lane_modifications = {282: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_HighVerticalCurvature.value,
                                     1.0)]}

        expected_output.as_route_plan_lane_segments[7][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[8][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[8][2].e_cst_lane_end_cost = 1.0
        for i in reversed(range(7)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

    elif request.param is "scene_nine":
        lane_modifications = {292: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_HighHorizontalCurvat.value,
                                     1.0)]}

        expected_output.as_route_plan_lane_segments[8][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[9][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[9][2].e_cst_lane_end_cost = 1.0
        for i in reversed(range(8)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

    elif request.param is "scene_ten":
        lane_modifications = {292: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_Unknown.value,
                                     1.0)]}

        expected_output.as_route_plan_lane_segments[8][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[9][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[9][2].e_cst_lane_end_cost = 1.0
        for i in reversed(range(8)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

    else:
        lane_modifications = {}

    return RoutePlanTestData(scene_static=modify_default_lane_attributes(lane_modifications),
                             expected_output=expected_output)


@pytest.fixture(scope='function', params=["scene_one"])
def lane_direction_scene_and_expected_output(request):
    # Set Default Expected Output
    expected_output = default_route_plan_for_PG_split_file()

    # Define Lane Modifications and Modify Expected Outputs
    if request.param is "scene_one":
        lane_modifications = {212: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Direction.value,
                                     MapLaneDirection.CeSYS_e_MapLaneDirection_OppositeTo_HostVehicle.value,
                                     1.0)]}

        expected_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0

    else:
        lane_modifications = {}

    return RoutePlanTestData(scene_static=modify_default_lane_attributes(lane_modifications),
                             expected_output=expected_output)


@pytest.fixture(scope='function', params=["scene_one"]) #,"scene_two"
def combined_scene_and_expected_output(request):
    # Set Default Expected Output
    expected_output = default_route_plan_for_PG_split_file()

    # Define Lane Modifications and Modify Expected Outputs
    if request.param is "scene_one":
        lane_modifications = {212: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus.value,
                                     LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_NotMapped.value,
                                     1.0)],
                              221: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_RoadConstruction.value,
                                     1.0)],
                              222: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus.value,
                                     LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_NotMapped.value,
                                     1.0),
                                    (True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_RoadConstruction.value,
                                     1.0)],
                              280: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneConstructionType.CeSYS_e_LaneConstructionType_Blocked.value,
                                     1.0)]}

        # Road Segment 27
        expected_output.as_route_plan_lane_segments[7][0].e_cst_lane_end_cost = 1.0

        # Road Segment 28
        expected_output.as_route_plan_lane_segments[8][0].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[8][0].e_cst_lane_end_cost = 1.0

        for i in reversed(range(7)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

        # Road Segment 22
        expected_output.as_route_plan_lane_segments[2][1].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[2][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[2][1].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[2][2].e_cst_lane_end_cost = 1.0

        for i in reversed(range(2)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

        # Road Segment 21
        expected_output.as_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0

        for i in reversed(range(1)):
            for j in range(expected_output.a_Cnt_num_lane_segments[i]):
                expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
                    expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
                    ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
                        expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length

        # Road Segment 20
        expected_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost = 1.0

    # elif request.param is "scene_two":
    #     # Scenario in Slides
    #     road_segment_ids = [1, 2, 3, 4, 5, 6]
    #
    #     lane_segment_ids = [[1, 2],
    #                         [7, 3, 4],
    #                         [5, 6],
    #                         [8, 9],
    #                         [10],
    #                         [11]]
    #
    #     navigation_plan = [1, 2, 4, 5]
    #
    #     downstream_road_segment_ids = {1: [2],
    #                                    2: [3, 4],
    #                                    3: [],
    #                                    4: [5, 6],
    #                                    5: [],
    #                                    6: []}
    #
    #     downstream_lane_connectivity = {1: [(7, ManeuverType.RIGHT_EXIT_CONNECTION),
    #                                         (3, ManeuverType.STRAIGHT_CONNECTION)],
    #                                     2: [(4, ManeuverType.STRAIGHT_CONNECTION)],
    #                                     3: [(5, ManeuverType.STRAIGHT_CONNECTION)],
    #                                     4: [(6, ManeuverType.STRAIGHT_CONNECTION)],
    #                                     7: [(8, ManeuverType.RIGHT_FORK_CONNECTION),
    #                                         (9, ManeuverType.LEFT_FORK_CONNECTION)],
    #                                     8: [(11, ManeuverType.STRAIGHT_CONNECTION)],
    #                                     9: [(10, ManeuverType.STRAIGHT_CONNECTION)]}
    #
    #     lane_modifications = {2: [(True,
    #                                RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus.value,
    #                                LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_NotMapped.value,
    #                                1.0)],
    #                           4: [(True,
    #                                RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
    #                                GMAuthorityType.CeSYS_e_GMAuthorityType_RoadConstruction.value,
    #                                1.0)]}
    #
    #     scene_static_publisher = SceneStaticPublisher(road_segment_ids=road_segment_ids,
    #                                                   lane_segment_ids=lane_segment_ids,
    #                                                   navigation_plan=navigation_plan,
    #                                                   downstream_road_segment_ids=downstream_road_segment_ids,
    #                                                   downstream_lane_connectivity=downstream_lane_connectivity,
    #                                                   lane_attribute_modifications=lane_modifications)
    #
    #     # Expected Output
    #     expected_output = DataRoutePlan(e_b_is_valid=True,
    #                                     e_Cnt_num_road_segments = len(navigation_plan),
    #                                     a_i_road_segment_ids = np.array(navigation_plan),
    #                                     a_Cnt_num_lane_segments = np.array([2, 3, 2, 1]),
    #                                     as_route_plan_lane_segments = [[RoutePlanLaneSegment(e_i_lane_segment_id=lane_segment_id,
    #                                                                                          e_cst_lane_occupancy_cost=0.0,
    #                                                                                          e_cst_lane_end_cost=0.0)
    #                                                                     for lane_segment_id in lane_segment_ids[road_segment_id - 1]]
    #                                                                    for road_segment_id in navigation_plan])
    #
    #     for i in reversed(range(7)):
    #         for j in range(expected_output.a_Cnt_num_lane_segments[i]):
    #             expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost = \
    #                 expected_output.as_route_plan_lane_segments[i + 1][j].e_cst_lane_end_cost * \
    #                 ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** MapUtils.get_lane(
    #                     expected_output.as_route_plan_lane_segments[i + 1][j].e_i_lane_segment_id).e_l_length
    #     # Road Segment 1
    #     expected_output.as_route_plan_lane_segments[0][1].e_cst_lane_occupancy_cost = 1.0
    #     expected_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost = 1.0
    #
    #     # Road Segment 2
    #     expected_output.as_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1.0
    #     expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0
    #     expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0
    #
    #     # Road Segment 4
    #     expected_output.as_route_plan_lane_segments[2][0].e_cst_lane_end_cost = 1.0
    #
    #     return RoutePlanTestData(scene_static=scene_static_publisher.generate_data(),
    #                              expected_output=expected_output)
    else:
        lane_modifications = {}

    return RoutePlanTestData(scene_static=modify_default_lane_attributes(lane_modifications),
                             expected_output=expected_output)
