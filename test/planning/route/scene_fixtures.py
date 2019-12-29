import numpy as np
import pickle
import pytest
from typing import List, Dict, Tuple

from decision_making.paths import Paths
from decision_making.src.global_constants import PG_SPLIT_PICKLE_FILE_NAME
from decision_making.src.messages.route_plan_message import DataRoutePlan, RoutePlanLaneSegment
from decision_making.src.messages.scene_static_enums import LaneConstructionType,\
    RoutePlanLaneSegmentAttr, LaneMappingStatusType, GMAuthorityType, MapLaneDirection,\
    ManeuverType
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import EgoState, ObjectSize
from decision_making.test.planning.route.scene_static_publisher import SceneStaticPublisher
from decision_making.src.planning.types import LaneSegmentID


class RoutePlanTestData:
    def __init__(self, scene_static: SceneStatic, expected_binary_output: DataRoutePlan, scene_name: str):
        self.scene_static = scene_static
        self.expected_binary_output = expected_binary_output
        self.scene_name = scene_name


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
            lane_attributes = [LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap,
                               GMAuthorityType.CeSYS_e_GMAuthorityType_None,
                               LaneConstructionType.CeSYS_e_LaneConstructionType_Normal,
                               MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle]
            lane_attribute_confidences = np.ones(4)

            for lane_attribute_modification in lane_attribute_modifications[lane_segment.e_i_lane_segment_id]:
                if lane_attribute_modification[0] is True:
                    lane_attribute_type = type(lane_attributes[lane_attribute_modification[1]])
                    lane_attributes[lane_attribute_modification[1]] = lane_attribute_type(lane_attribute_modification[2])
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
    expected_binary_output = default_route_plan_for_PG_split_file()

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
        expected_binary_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost = 1.0

        # Road Segment 21
        expected_binary_output.as_route_plan_lane_segments[1][1].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0

        expected_binary_output.as_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0

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
        expected_binary_output.as_route_plan_lane_segments[8][1].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[8][2].e_cst_lane_end_cost = 1.0

        # Road Segment 29
        expected_binary_output.as_route_plan_lane_segments[9][1].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[9][2].e_cst_lane_occupancy_cost = 1.0

        expected_binary_output.as_route_plan_lane_segments[9][1].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[9][2].e_cst_lane_end_cost = 1.0

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
        expected_binary_output.as_route_plan_lane_segments[4][1].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[4][2].e_cst_lane_end_cost = 1.0

        # Road Segment 25
        expected_binary_output.as_route_plan_lane_segments[5][1].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[5][2].e_cst_lane_occupancy_cost = 1.0

        expected_binary_output.as_route_plan_lane_segments[5][1].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[5][2].e_cst_lane_end_cost = 1.0
    else:
        lane_modifications = {}

    return RoutePlanTestData(scene_static=modify_default_lane_attributes(lane_modifications),
                             expected_binary_output=expected_binary_output,
                             scene_name=request.param)


@pytest.fixture(scope='function', params=["scene_one",
                                          "scene_two"])
def map_scene_and_expected_output(request):
    # Set Default Expected Output
    expected_binary_output = default_route_plan_for_PG_split_file()

    # Define Lane Modifications and Modify Expected Outputs
    if request.param is "scene_one":
        lane_modifications = {212: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus.value,
                                     LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_NotMapped.value,
                                     1.0)]}

        expected_binary_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0

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
        expected_binary_output.as_route_plan_lane_segments[8][0].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[8][1].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[8][2].e_cst_lane_end_cost = 1.0

        # Road Segment 29
        expected_binary_output.as_route_plan_lane_segments[9][0].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[9][1].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[9][2].e_cst_lane_occupancy_cost = 1.0

        expected_binary_output.as_route_plan_lane_segments[9][0].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[9][1].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[9][2].e_cst_lane_end_cost = 1.0

    else:
        lane_modifications = {}

    return RoutePlanTestData(scene_static=modify_default_lane_attributes(lane_modifications),
                             expected_binary_output=expected_binary_output,
                             scene_name=request.param)


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
    expected_binary_output = default_route_plan_for_PG_split_file()

    # Define Lane Modifications and Modify Expected Outputs
    if request.param is "scene_one":
        lane_modifications = {212: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_RoadConstruction.value,
                                     1.0)]}

        expected_binary_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_two":
        lane_modifications = {222: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_BadRoadCondition.value,
                                     1.0)]}

        expected_binary_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[2][2].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[2][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_three":
        lane_modifications = {232: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_ComplexRoad.value,
                                     1.0)]}

        expected_binary_output.as_route_plan_lane_segments[2][2].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[3][2].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[3][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_four":
        lane_modifications = {242: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_MovableBarriers.value,
                                     1.0)]}

        expected_binary_output.as_route_plan_lane_segments[3][2].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[4][2].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[4][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_five":
        lane_modifications = {252: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_BidirectionalFreew.value,
                                     1.0)]}

        expected_binary_output.as_route_plan_lane_segments[4][2].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[5][2].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[5][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_six":
        lane_modifications = {262: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_HighCrossTrackSlope.value,
                                     1.0)]}

        expected_binary_output.as_route_plan_lane_segments[5][2].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[6][2].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[6][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_seven":
        lane_modifications = {272: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_HighAlongTrackSlope.value,
                                     1.0)]}

        expected_binary_output.as_route_plan_lane_segments[6][2].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[7][2].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[7][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_eight":
        lane_modifications = {282: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_HighVerticalCurvature.value,
                                     1.0)]}

        expected_binary_output.as_route_plan_lane_segments[7][2].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[8][2].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[8][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_nine":
        lane_modifications = {292: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_HighHorizontalCurvat.value,
                                     1.0)]}

        expected_binary_output.as_route_plan_lane_segments[8][2].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[9][2].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[9][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_ten":
        lane_modifications = {292: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_Unknown.value,
                                     1.0)]}

        expected_binary_output.as_route_plan_lane_segments[8][2].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[9][2].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[9][2].e_cst_lane_end_cost = 1.0
    else:
        lane_modifications = {}

    return RoutePlanTestData(scene_static=modify_default_lane_attributes(lane_modifications),
                             expected_binary_output=expected_binary_output,
                             scene_name=request.param)


@pytest.fixture(scope='function', params=["scene_one"])
def lane_direction_scene_and_expected_output(request):
    # Set Default Expected Output
    expected_binary_output = default_route_plan_for_PG_split_file()

    # Define Lane Modifications and Modify Expected Outputs
    if request.param is "scene_one":
        lane_modifications = {212: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Direction.value,
                                     MapLaneDirection.CeSYS_e_MapLaneDirection_OppositeTo_HostVehicle.value,
                                     1.0)]}

        expected_binary_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0

    else:
        lane_modifications = {}

    return RoutePlanTestData(scene_static=modify_default_lane_attributes(lane_modifications),
                             expected_binary_output=expected_binary_output,
                             scene_name=request.param)


@pytest.fixture(scope='function', params=["scene_one",
                                          "scene_two"])
def combined_scene_and_expected_output(request):
    # Set Default Expected Output
    expected_binary_output = default_route_plan_for_PG_split_file()

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

        # Road Segment 20
        expected_binary_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost = 1.0

        # Road Segment 21
        expected_binary_output.as_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0

        # Road Segment 22
        expected_binary_output.as_route_plan_lane_segments[2][1].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[2][2].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[2][1].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[2][2].e_cst_lane_end_cost = 1.0

        # Road Segment 27
        expected_binary_output.as_route_plan_lane_segments[7][0].e_cst_lane_end_cost = 1.0

        # Road Segment 28
        expected_binary_output.as_route_plan_lane_segments[8][0].e_cst_lane_occupancy_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[8][0].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_two":
        """
        The other scenes in this file are on a straight road so visualization is easier. This geometry is a little more
        complicated so the diagram below was added for clarity. In this scene, no lanes will be blocked, but the goal
        is to reach lane 71.

            12 -> 22 -> 33 -> 42
            11 -> 21 -> 32 -> 41
                    `-> 31   ,-> 52 -> 71
                          `-<
                             `-> 51 -> 61
        """
        road_segment_ids = [1, 2, 3, 4, 5, 6, 7]

        lane_segment_ids = [[11, 12],
                            [21, 22],
                            [31, 32, 33],
                            [41, 42],
                            [51, 52],
                            [61],
                            [71]]

        navigation_plan = [1, 2, 3, 5, 7]

        downstream_road_segment_ids = {1: [2],
                                       2: [3],
                                       3: [4, 5],
                                       4: [],
                                       5: [6, 7],
                                       6: [],
                                       7: []}

        downstream_lane_connectivity = {11: [(21, ManeuverType.STRAIGHT_CONNECTION)],
                                        12: [(22, ManeuverType.STRAIGHT_CONNECTION)],
                                        21: [(31, ManeuverType.RIGHT_EXIT_CONNECTION),
                                             (32, ManeuverType.STRAIGHT_CONNECTION)],
                                        22: [(33, ManeuverType.STRAIGHT_CONNECTION)],
                                        32: [(41, ManeuverType.STRAIGHT_CONNECTION)],
                                        33: [(42, ManeuverType.STRAIGHT_CONNECTION)],
                                        31: [(51, ManeuverType.RIGHT_FORK_CONNECTION),
                                             (52, ManeuverType.LEFT_FORK_CONNECTION)],
                                        51: [(61, ManeuverType.STRAIGHT_CONNECTION)],
                                        52: [(71, ManeuverType.STRAIGHT_CONNECTION)]}

        scene_static_publisher = SceneStaticPublisher(road_segment_ids=road_segment_ids,
                                                      lane_segment_ids=lane_segment_ids,
                                                      navigation_plan=navigation_plan,
                                                      downstream_road_segment_ids=downstream_road_segment_ids,
                                                      downstream_lane_connectivity=downstream_lane_connectivity)

        # Expected Output
        expected_binary_output = DataRoutePlan(e_b_is_valid=True,
                                               e_Cnt_num_road_segments = len(navigation_plan),
                                               a_i_road_segment_ids = np.array(navigation_plan),
                                               a_Cnt_num_lane_segments = np.array([2, 2, 3, 2, 1]),
                                               as_route_plan_lane_segments = [[RoutePlanLaneSegment(e_i_lane_segment_id=lane_segment_id,
                                                                                                    e_cst_lane_occupancy_cost=0.0,
                                                                                                    e_cst_lane_end_cost=0.0)
                                                                               for lane_segment_id in lane_segment_ids[road_segment_id - 1]]
                                                                              for road_segment_id in navigation_plan])

        # Road Segment 2
        expected_binary_output.as_route_plan_lane_segments[2][1].e_cst_lane_end_cost = 1.0
        expected_binary_output.as_route_plan_lane_segments[2][2].e_cst_lane_end_cost = 1.0

        # Road Segment 4
        expected_binary_output.as_route_plan_lane_segments[3][0].e_cst_lane_end_cost = 1.0

        return RoutePlanTestData(scene_static=scene_static_publisher.generate_data(),
                                 expected_binary_output=expected_binary_output,
                                 scene_name=request.param)
    else:
        lane_modifications = {}

    return RoutePlanTestData(scene_static=modify_default_lane_attributes(lane_modifications),
                             expected_binary_output=expected_binary_output,
                             scene_name=request.param)


def generate_ego_state(ego_lane_id: int, ego_lane_station: float) -> EgoState:
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)
    map_state = MapState(np.array([ego_lane_station, 10, 0, 0, 0, 0]), ego_lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1, off_map=False)
    return ego_state


@pytest.fixture(scope='function', params=["scene_one",
                                          "scene_two",
                                          "scene_three"])
def construction_scene_for_takeover_test(request):
    # Set Default Expected Output
    route_plan_data = default_route_plan_for_PG_split_file()
    expected_takeover = False
    ego_state = generate_ego_state(ego_lane_id = 200, ego_lane_station = 0)

    # Define Lane Modifications and Modify Expected Outputs
    if request.param is "scene_one":
        # Two lanes are blocked ahead, and the vehicle is close to crossing into one of them. The takeover flag should be False
        # because a lane change is preferred over handing back to the driver.
        lane_modifications = {211: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneConstructionType.CeSYS_e_LaneConstructionType_Blocked.value,
                                     1.0)],
                              212: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneConstructionType.CeSYS_e_LaneConstructionType_Blocked.value,
                                     1.0)]}

        # Road Segment 20
        route_plan_data.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost = 1.0

        # Road Segment 21
        route_plan_data.as_route_plan_lane_segments[1][1].e_cst_lane_occupancy_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0

        ego_state = generate_ego_state(ego_lane_id = 201 , ego_lane_station = 75)

    elif request.param is "scene_two":
        # All lanes are blocked, and the vehicle is close to crossing into one of them. The takeover flag should be True.
        lane_modifications = {290: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneConstructionType.CeSYS_e_LaneConstructionType_Blocked.value,
                                     1.0)],
                              291: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneConstructionType.CeSYS_e_LaneConstructionType_Blocked.value,
                                     1.0)],
                              292: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneConstructionType.CeSYS_e_LaneConstructionType_Blocked.value,
                                     1.0)]}

        # Road Segment 28
        route_plan_data.as_route_plan_lane_segments[8][0].e_cst_lane_end_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[8][1].e_cst_lane_end_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[8][2].e_cst_lane_end_cost = 1.0

        # Road Segment 29
        route_plan_data.as_route_plan_lane_segments[9][0].e_cst_lane_occupancy_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[9][1].e_cst_lane_occupancy_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[9][2].e_cst_lane_occupancy_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[9][0].e_cst_lane_end_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[9][1].e_cst_lane_end_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[9][2].e_cst_lane_end_cost = 1.0

        ego_state = generate_ego_state(ego_lane_id = 282, ego_lane_station = 80)

        expected_takeover = True

    elif request.param is "scene_three":
        # TODO: this is now not true since thersholds were increased. Need to change this fixture
        # All lanes are blocked, but the vehicle is not close to crossing into one of them. The takeover flag should be False.
        lane_modifications = {220: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_RoadConstruction.value,
                                     1.0)],
                              221: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_RoadConstruction.value,
                                     1.0)],
                              222: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_RoadConstruction.value,
                                     1.0)]}

        # Road Segment 21
        route_plan_data.as_route_plan_lane_segments[1][0].e_cst_lane_end_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0

        # Road Segment 22
        route_plan_data.as_route_plan_lane_segments[2][0].e_cst_lane_occupancy_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[2][1].e_cst_lane_occupancy_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[2][2].e_cst_lane_occupancy_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[2][0].e_cst_lane_end_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[2][1].e_cst_lane_end_cost = 1.0
        route_plan_data.as_route_plan_lane_segments[2][2].e_cst_lane_end_cost = 1.0

        ego_state = generate_ego_state(ego_lane_id=211, ego_lane_station=30)

        expected_takeover = True
    else:
        lane_modifications = {}

    return TakeOverTestData(scene_static=modify_default_lane_attributes(lane_modifications),
                            route_plan_data=route_plan_data, ego_state=ego_state, expected_takeover=expected_takeover)
