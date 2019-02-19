import pytest
import numpy as np
from typing import List

from decision_making.src.messages.route_plan_message import DataRoutePlan, RoutePlanLaneSegment
from decision_making.src.messages.scene_static_enums import (
    LaneConstructionType,
    RoutePlanLaneSegmentAttr,
    LaneMappingStatusType,
    GMAuthorityType,
    MapLaneDirection)
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.test.messages.static_scene_fixture import create_scene_static_from_map_api

from mapping.src.service.map_service import MapService

class RoutePlanTestData:
    def __init__(self, scene_static: SceneStatic, expected_output: DataRoutePlan):
        self.scene_static = scene_static
        self.expected_output = expected_output

def default_route_plan() -> DataRoutePlan:
    return DataRoutePlan(e_b_is_valid=True,
                         e_Cnt_num_road_segments = 10,
                         a_i_road_segment_ids = np.arange(20, 30),
                         a_Cnt_num_lane_segments = np.full(10, 3),
                         as_route_plan_lane_segments = [[RoutePlanLaneSegment(e_i_lane_segment_id=lane_segment_id_base+lane_number,
                                                                              e_cst_lane_occupancy_cost=0.0,
                                                                              e_cst_lane_end_cost=0.0) for lane_number in [0, 1, 2]]
                                                        for lane_segment_id_base in np.arange(200, 300, 10)])

@pytest.fixture(scope='function', params=[
                                            "scene_one"
#                                          ,"scene_two"
#                                          ,"scene_three"
                                          ])
def construction_scene_and_expected_output(request):
    # Set Default Expected Output
    expected_output = default_route_plan()

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
    else:
        lane_modifications = {}
    
    # for road_segment in expected_output.as_route_plan_lane_segments:
    #     for lane_segment in road_segment:
    #         print("lane_segment_id     = ", lane_segment.e_i_lane_segment_id)
    #         print("lane_occupancy_cost = ", lane_segment.e_cst_lane_occupancy_cost)
    #         print("lane_end_cost       = ", lane_segment.e_cst_lane_end_cost, "\n")
        
    #     print("==========================\n")
        
    # Initialize Map
    MapService.initialize('PG_split.bin')
    
    return RoutePlanTestData(scene_static=create_scene_static_from_map_api(MapService.get_instance(), lane_modifications),
                             expected_output=expected_output)

@pytest.fixture(scope='function', params=["scene_one",
                                          "scene_two"])
def map_scene_and_expected_output(request):
    # Set Default Expected Output
    expected_output = default_route_plan()

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

    else:
        lane_modifications = {}
        
    # Initialize Map
    MapService.initialize('PG_split.bin')
    
    return RoutePlanTestData(scene_static=create_scene_static_from_map_api(MapService.get_instance(), lane_modifications),
                             expected_output=expected_output)

@pytest.fixture(scope='function', params=["scene_one",
                                          "scene_two",
                                          "scene_three",
                                          "scene_four",
                                          "scene_five",
                                          "scene_six",
                                          "scene_seven",
                                          "scene_eight",
                                          "scene_nine"])
def gmfa_scene_and_expected_output(request):
    # Set Default Expected Output
    expected_output = default_route_plan()

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
    elif request.param is "scene_three":
        lane_modifications = {232: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_ComplexRoad.value,
                                     1.0)]}
        
        expected_output.as_route_plan_lane_segments[2][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[3][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[3][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_four":
        lane_modifications = {242: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_MovableBarriers.value,
                                     1.0)]}
        
        expected_output.as_route_plan_lane_segments[3][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[4][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[4][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_five":
        lane_modifications = {252: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_BidirectionalFreew.value,
                                     1.0)]}
        
        expected_output.as_route_plan_lane_segments[4][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[5][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[5][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_six":
        lane_modifications = {262: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_HighCrossTrackSlope.value,
                                     1.0)]}
        
        expected_output.as_route_plan_lane_segments[5][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[6][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[6][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_seven":
        lane_modifications = {272: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_HighAlongTrackSlope.value,
                                     1.0)]}
        
        expected_output.as_route_plan_lane_segments[6][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[7][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[7][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_eight":
        lane_modifications = {282: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_HighVerticalCurvature.value,
                                     1.0)]}
        
        expected_output.as_route_plan_lane_segments[7][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[8][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[8][2].e_cst_lane_end_cost = 1.0
    elif request.param is "scene_nine":
        lane_modifications = {292: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_HighHorizontalCurvat.value,
                                     1.0)]}
        
        expected_output.as_route_plan_lane_segments[8][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[9][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[9][2].e_cst_lane_end_cost = 1.0
    else:
        lane_modifications = {}
        
    # Initialize Map
    MapService.initialize('PG_split.bin')
    
    return RoutePlanTestData(scene_static=create_scene_static_from_map_api(MapService.get_instance(), lane_modifications),
                             expected_output=expected_output)

@pytest.fixture(scope='function', params=["scene_one"])
def lane_direction_scene_and_expected_output(request):
    # Set Default Expected Output
    expected_output = default_route_plan()

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
        
    # Initialize Map
    MapService.initialize('PG_split.bin')
    
    return RoutePlanTestData(scene_static=create_scene_static_from_map_api(MapService.get_instance(), lane_modifications),
                             expected_output=expected_output)

@pytest.fixture(scope='function', params=["scene_one"])
def combined_scene_and_expected_output(request):
    # Set Default Expected Output
    expected_output = default_route_plan()

    # Define Lane Modifications and Modify Expected Outputs
    if request.param is "scene_one":
        lane_modifications = {212: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_NotMapped.value,
                                     1.0)],
                              221: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_RoadConstruction.value,
                                     1.0)],
                              222: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_NotMapped.value,
                                     1.0),
                                     (True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     GMAuthorityType.CeSYS_e_GMAuthorityType_RoadConstruction.value,
                                     1.0)],
                              280: [(True,
                                     RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction.value,
                                     LaneConstructionType.CeSYS_e_LaneConstructionType_Blocked.value,
                                     1.0)]}
        
        # Road Segment 20
        expected_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost = 1.0

        # Road Segment 21
        expected_output.as_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[1][2].e_cst_lane_occupancy_cost = 1.0

        # Road Segment 22
        expected_output.as_route_plan_lane_segments[2][1].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[2][2].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[2][1].e_cst_lane_end_cost = 1.0
        expected_output.as_route_plan_lane_segments[2][2].e_cst_lane_end_cost = 1.0

        # Road Segment 27
        expected_output.as_route_plan_lane_segments[7][0].e_cst_lane_end_cost = 1.0

        # Road Segment 28
        expected_output.as_route_plan_lane_segments[8][0].e_cst_lane_occupancy_cost = 1.0
        expected_output.as_route_plan_lane_segments[8][0].e_cst_lane_end_cost = 1.0
    else:
        lane_modifications = {}
        
    # Initialize Map
    MapService.initialize('PG_split.bin')
    
    return RoutePlanTestData(scene_static=create_scene_static_from_map_api(MapService.get_instance(), lane_modifications),
                             expected_output=expected_output)
