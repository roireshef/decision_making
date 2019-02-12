import copy
import pytest
import numpy as np

from decision_making.src.messages.scene_common_messages import Header, MapOrigin, Timestamp
from decision_making.src.messages.route_plan_message import RoutePlan, DataRoutePlan, RoutePlanLaneSegment
from decision_making.src.messages.scene_static_message import SceneStatic, DataSceneStaticBase, DataSceneStaticGeometry, \
    DataNavigationPlan, SceneRoadSegment, MapRoadSegmentType, SceneLaneSegmentGeometry, SceneLaneSegmentBase, MapLaneType, \
    LaneSegmentConnectivity, ManeuverType, NominalPathPoint, MapLaneMarkerType, BoundaryPoint, AdjacentLane, MovingDirection

'''
This pytest fixture will create scene_static for two roads:
(1) One straight road, ROAD_ID, with NUM_STRAIGHT_ROAD_SEGS_LONG road segments 
    in the s direction and NUM_STRAIGHT_ROAD_SEGS_CROSS in the d direction. 
(2) One lane split road, ROAD_ID_SPLIT, connected ROAD_ID at road_seg_split_idx (param),
    referring to the index of as_scene_road_segment in the scene_static. This lane split 
    will be one lane segment by NUM_SPLIT_LANE_SEGS_LONG

Road IDs = single digits, road segment IDs = 10*road ID, lane IDs = 10*road segment ID + lane ordinal
Connectivity (upstream, downstream, adjacent lanes and roads) information is valid.
NominalPathPoints for both roads are set to count = NUM_NOM_PATH_POINTS, setting only:
- CeSYS_NominalPathPoint_e_l_EastX
- CeSYS_NominalPathPoint_e_l_NorthY
- CeSYS_NominalPathPoint_e_l_s
- CeSYS_NominalPathPoint_e_l_right_offset/left_offset
'''

@pytest.fixture
def route_plan_with_split():
    header = Header(e_Cnt_SeqNum=0, s_Timestamp=Timestamp(0, 0), e_Cnt_version=0)
    data_route = create_route_plan_split()
    route_plan = RoutePlan(s_Header=header, s_Data=data_route)
    return route_plan

@pytest.fixture
def route_plan_without_split():
    header = Header(e_Cnt_SeqNum=0, s_Timestamp=Timestamp(0, 0), e_Cnt_version=0)
    data_route = create_route_plan_no_split()
    route_plan = RoutePlan(s_Header=header, s_Data=data_route)
    return route_plan
    

@pytest.fixture
def scene_static_lane_split():
    scene_static = scene_static_straight()
    road_seg_idx_to_add_split = 3
    add_lane_split(scene_static, road_seg_idx_to_add_split)
    return scene_static

# ---------------------------------------------------------
# Constants for static data and tests
# ---------------------------------------------------------
NUM_STRAIGHT_ROAD_SEGS_LONG = 5
NUM_STRAIGHT_LANE_SEGS_CROSS = 3
NUM_SPLIT_LANE_SEGS_LONG = 3
NUM_NOM_PATH_PTS = 30

HORIZON_PERCEP_DIST = 50    # [m]
HALF_LANE_WIDTH = 2         # [m]
NOM_SPEED = 50.0            # [m/s]
DS = 1.0                    # [m]

ROAD_ID = 3
ROAD_ID_SPLIT = 4

START_EAST_X_VAL = -3e2 
START_NORTH_Y_VAL = -7e4

# ---------------------------------------------------------
# Sample route plans with and without split
# ---------------------------------------------------------
def route_plan_lane_seg_list_free(id1, id2, id3):
    return [RoutePlanLaneSegment(e_i_lane_segment_id=id1, e_cst_lane_occupancy_cost=0, e_cst_lane_end_cost=0),
            RoutePlanLaneSegment(e_i_lane_segment_id=id2, e_cst_lane_occupancy_cost=0, e_cst_lane_end_cost=0),
            RoutePlanLaneSegment(e_i_lane_segment_id=id3, e_cst_lane_occupancy_cost=0, e_cst_lane_end_cost=0)]

def create_route_plan_no_split():
    road_seg_ids = [30, 31, 32, 33, 34]
    cnt_lane_segs = [3, 3, 3, 3, 3]
    rp_lane_segs = []
    rp_lane_segs.append(route_plan_lane_seg_list_free(300, 301, 302))
    rp_lane_segs.append(route_plan_lane_seg_list_free(310, 311, 312))
    rp_lane_segs.append(route_plan_lane_seg_list_free(320, 321, 322))
    rp_lane_segs.append(route_plan_lane_seg_list_free(330, 331, 332))
    rp_lane_segs.append(route_plan_lane_seg_list_free(340, 341, 342))

    inner = []
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=430, e_cst_lane_occupancy_cost=1, e_cst_lane_end_cost=1))
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=431, e_cst_lane_occupancy_cost=1, e_cst_lane_end_cost=1))
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=432, e_cst_lane_occupancy_cost=1, e_cst_lane_end_cost=1))    
    rp_lane_segs.append(inner)

    return DataRoutePlan(e_b_is_valid=True,
                         e_Cnt_num_road_segments=len(road_seg_ids),
                         a_i_road_segment_ids=np.ndarray(road_seg_ids),
                         a_Cnt_num_lane_segments=np.ndarray(cnt_lane_segs),
                         as_route_plan_lane_segments=rp_lane_segs)

def create_route_plan_split():
    road_seg_ids = [30, 31, 32, 33, 43]
    cnt_lane_segs = [3, 3, 3, 3, 3]
    rp_lane_segs = []
    inner = []
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=300, e_cst_lane_occupancy_cost=0, e_cst_lane_end_cost=0))
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=301, e_cst_lane_occupancy_cost=0, e_cst_lane_end_cost=0))
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=302, e_cst_lane_occupancy_cost=1, e_cst_lane_end_cost=1))    
    rp_lane_segs.append(inner)
    
    inner = []
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=310, e_cst_lane_occupancy_cost=0, e_cst_lane_end_cost=0))
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=311, e_cst_lane_occupancy_cost=0, e_cst_lane_end_cost=0))
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=312, e_cst_lane_occupancy_cost=1, e_cst_lane_end_cost=1))    
    rp_lane_segs.append(inner)
    
    inner = []
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=320, e_cst_lane_occupancy_cost=0, e_cst_lane_end_cost=0))
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=321, e_cst_lane_occupancy_cost=0, e_cst_lane_end_cost=0))
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=322, e_cst_lane_occupancy_cost=1, e_cst_lane_end_cost=1))    
    rp_lane_segs.append(inner)

    inner = []
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=330, e_cst_lane_occupancy_cost=0, e_cst_lane_end_cost=0))
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=331, e_cst_lane_occupancy_cost=0, e_cst_lane_end_cost=1))
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=332, e_cst_lane_occupancy_cost=1, e_cst_lane_end_cost=1))    
    rp_lane_segs.append(inner)

    inner = []
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=340, e_cst_lane_occupancy_cost=1, e_cst_lane_end_cost=1))
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=341, e_cst_lane_occupancy_cost=1, e_cst_lane_end_cost=1))
    inner.append(RoutePlanLaneSegment(e_i_lane_segment_id=342, e_cst_lane_occupancy_cost=1, e_cst_lane_end_cost=1))    
    rp_lane_segs.append(inner)

    rp_lane_segs.append(route_plan_lane_seg_list_free(430, 431, 432))

    return DataRoutePlan(e_b_is_valid=True,
                         e_Cnt_num_road_segments=len(road_seg_ids),
                         a_i_road_segment_ids=np.ndarray(road_seg_ids),
                         a_Cnt_num_lane_segments=np.ndarray(cnt_lane_segs),
                         as_route_plan_lane_segments=rp_lane_segs)

# ---------------------------------------------------------
# Defines sample scene message
# ---------------------------------------------------------
def create_starting_nominal_path_pt():
    point = np.empty(len(NominalPathPoint))    
    point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value] = START_EAST_X_VAL
    point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value] = START_NORTH_Y_VAL
    point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_heading.value] = 0 
    point[NominalPathPoint.CeSYS_NominalPathPoint_e_il_curvature.value] = 0
    point[NominalPathPoint.CeSYS_NominalPathPoint_e_il2_curvature_rate.value] = 0
    point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_cross_slope.value] = 0
    point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_along_slope.value] = 0
    point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] = 0 
    point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_left_offset.value] = HALF_LANE_WIDTH
    point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_right_offset.value] = HALF_LANE_WIDTH
    return point


def scene_static_straight():
    scene_road_segments = []
    scene_lane_segments_base = []
    scene_lane_segments_geo = []
    lane_seg_idx = 0

    # Create road segments
    for road_idx in range(NUM_STRAIGHT_ROAD_SEGS_LONG):
        # Information for road segment connectivity
        road_id = ROAD_ID * 10 + road_idx
        road_seg_upstream_count = 1 if road_idx > 0 else 0
        road_seg_downstream_count = 1 if road_idx < (NUM_STRAIGHT_ROAD_SEGS_LONG - 1) else 0
        road_seg_upstream_id = road_id - 1 if road_seg_upstream_count == 1 else -1
        road_seg_downstream_id = road_id + 1 if road_seg_downstream_count == 1 else -1

        current_lane_seg_ids = []
        for lane_idx in range(NUM_STRAIGHT_LANE_SEGS_CROSS):
            lane_id = road_id * 10 + lane_idx
            current_lane_seg_ids.append(lane_id)

            # Starting nom path pt 
            if road_idx == 0:
                starting_nominal_path_pt = create_starting_nominal_path_pt()
                if lane_seg_idx != 0:
                    starting_nominal_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value] -= (HALF_LANE_WIDTH * 2)
            else:
                starting_nominal_path_pt = copy.deepcopy(scene_lane_segments_geo[lane_seg_idx - NUM_STRAIGHT_LANE_SEGS_CROSS].a_nominal_path_points[-1])

            # Nominal path points
            nominal_points = []
            for pt_idx in range(NUM_NOM_PATH_PTS):                
                if pt_idx == 0:
                    point = starting_nominal_path_pt
                else:
                    point = copy.deepcopy(nominal_points[pt_idx - 1])
                    point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value] += DS
                    point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] = pt_idx * DS
                nominal_points.append(point)

            # Boundary points
            left_boundary_point = BoundaryPoint(
                MapLaneMarkerType.MapLaneMarkerType_SolidSingleLine_BottsDots, 0, (DS * (NUM_NOM_PATH_PTS - 1))
            )
            right_boundary_point = BoundaryPoint(
                MapLaneMarkerType.MapLaneMarkerType_SolidSingleLine_BottsDots, 0, (DS * (NUM_NOM_PATH_PTS - 1))
            )

            # Downstream/Upstream
            downstream_lane_count = 1 if road_seg_downstream_count == 1 else -1
            upstream_lane_count = 1 if road_seg_upstream_count == 1 else -1 
            downstream_id = (lane_id + 10) if downstream_lane_count == 1 else -1
            upstream_id = (lane_id - 10) if upstream_lane_count == 1 else -1 

            downstream_lane_segment_connectivity = \
                LaneSegmentConnectivity(downstream_id, ManeuverType.STRAIGHT_CONNECTION) \
                if downstream_lane_count == 1 else LaneSegmentConnectivity(-1, ManeuverType.STRAIGHT_CONNECTION)
            upstream_lane_segment_connectivity = \
                LaneSegmentConnectivity(upstream_id, ManeuverType.STRAIGHT_CONNECTION) \
                if upstream_lane_count == 1 else LaneSegmentConnectivity(-1, ManeuverType.STRAIGHT_CONNECTION)

            # Rightmost lane, has left adj lanes
            if lane_idx == 0:
                left_adj_count = NUM_STRAIGHT_LANE_SEGS_CROSS - 1
                left_adj_lane_ids = [lane_id + i for i in range(1, (left_adj_count + 1))]
                left_adj_lanes = []
                for lane in left_adj_lane_ids:
                    left_adj_lanes.append(AdjacentLane(lane, MovingDirection.Adjacent_or_same_dir, MapLaneType.LocalRoadLane))
                right_adj_count = 0
                right_adj_lanes = [AdjacentLane(-1, MovingDirection.Adjacent_or_same_dir, MapLaneType.LocalRoadLane)]
            # Leftmost lane, has right adj lanes
            elif lane_idx == NUM_STRAIGHT_LANE_SEGS_CROSS - 1:
                right_adj_count = NUM_STRAIGHT_LANE_SEGS_CROSS - 1
                right_adj_lane_ids = [lane_id - i for i in range(1, (right_adj_count + 1))]
                right_adj_lanes = []
                for lane in right_adj_lane_ids:
                    right_adj_lanes.append(AdjacentLane(lane, MovingDirection.Adjacent_or_same_dir, MapLaneType.LocalRoadLane))
                left_adj_count = 0
                left_adj_lanes = [AdjacentLane(-1, MovingDirection.Adjacent_or_same_dir, MapLaneType.LocalRoadLane)]
            # Middle lane
            else:
                right_adj_count = lane_idx
                right_adj_lane_ids = [lane_id - 1 for i in range(1, (right_adj_count + 1))]
                right_adj_lanes = []
                for lane in right_adj_lane_ids:
                    right_adj_lanes.append(AdjacentLane(lane, MovingDirection.Adjacent_or_same_dir, MapLaneType.LocalRoadLane))
                
                left_adj_count = NUM_STRAIGHT_LANE_SEGS_CROSS - lane_idx
                left_adj_lane_ids = [lane_id + 1 for i in range(1, (left_adj_count + 1))]
                left_adj_lanes = []
                for lane in left_adj_lane_ids:
                    left_adj_lanes.append(AdjacentLane(lane, MovingDirection.Adjacent_or_same_dir, MapLaneType.LocalRoadLane))

            # Make SceneLaneSegment
            scene_lane_segment_base = SceneLaneSegmentBase(
                e_i_lane_segment_id=lane_id,
                e_i_road_segment_id=road_id,
                e_e_lane_type=MapLaneType.LocalRoadLane,
                e_Cnt_static_traffic_flow_control_count=0,
                as_static_traffic_flow_control=[],
                e_Cnt_dynamic_traffic_flow_control_count=0,
                as_dynamic_traffic_flow_control=[],
                e_Cnt_left_adjacent_lane_count=left_adj_count,
                as_left_adjacent_lanes=left_adj_lanes,
                e_Cnt_right_adjacent_lane_count=right_adj_count,
                as_right_adjacent_lanes=right_adj_lanes,
                e_Cnt_downstream_lane_count=downstream_lane_count,
                as_downstream_lanes=[downstream_lane_segment_connectivity],
                e_Cnt_upstream_lane_count=upstream_lane_count,
                as_upstream_lanes=[upstream_lane_segment_connectivity],
                e_v_nominal_speed=NOM_SPEED,
                e_i_downstream_road_intersection_id=0,
                e_Cnt_lane_coupling_count=0,
                as_lane_coupling=[],
                e_Cnt_num_active_lane_attributes=0,                
                a_i_active_lane_attribute_indices=np.array([]),    
                a_cmp_lane_attributes=np.array([]),                
                a_cmp_lane_attribute_confidences=np.array([])
            )
            scene_lane_segment_geo = SceneLaneSegmentGeometry(
                e_i_lane_segment_id=lane_id,
                e_i_road_segment_id=road_id,
                e_Cnt_nominal_path_point_count=len(nominal_points),
                a_nominal_path_points=np.asarray(nominal_points),
                e_Cnt_left_boundary_points_count=1,
                as_left_boundary_points=[left_boundary_point],
                e_Cnt_right_boundary_points_count=1,
                as_right_boundary_points=[right_boundary_point]
            )
            scene_lane_segments_base.append(scene_lane_segment_base)
            scene_lane_segments_geo.append(scene_lane_segment_geo)    
            lane_seg_idx += 1

        # Make SceneRoadSegment
        scene_road_segment = SceneRoadSegment(e_i_road_segment_id=road_id,
                                            e_i_road_id=ROAD_ID,
                                            e_Cnt_lane_segment_id_count=NUM_STRAIGHT_LANE_SEGS_CROSS,
                                            a_i_lane_segment_ids=np.array(current_lane_seg_ids),
                                            e_e_road_segment_type=MapRoadSegmentType.Normal,
                                            e_Cnt_upstream_segment_count=road_seg_upstream_count,
                                            a_i_upstream_road_segment_ids=np.array([road_seg_upstream_id]),
                                            e_Cnt_downstream_segment_count=road_seg_downstream_count,
                                            a_i_downstream_road_segment_ids=np.array([road_seg_downstream_id]))

        scene_road_segments.append(scene_road_segment)

    # Create header, map origin, populate data
    header = Header(e_Cnt_SeqNum=0, s_Timestamp=Timestamp(0, 0), e_Cnt_version=0)
    map_origin = MapOrigin(e_phi_latitude=.0, e_phi_longitude=.0, e_l_altitude=.0, s_Timestamp=Timestamp(0,0))
    base_data = DataSceneStaticBase(e_b_Valid=True,
                                    s_RecvTimestamp=Timestamp(0,0),
                                    s_ComputeTimestamp=Timestamp(0, 0),
                                    e_l_perception_horizon_front=HORIZON_PERCEP_DIST,
                                    e_l_perception_horizon_rear=HORIZON_PERCEP_DIST,
                                    e_Cnt_num_lane_segments=len(scene_lane_segments_base),
                                    as_scene_lane_segments=scene_lane_segments_base,
                                    e_Cnt_num_road_intersections=0,
                                    as_scene_road_intersection=[],
                                    e_Cnt_num_road_segments=len(scene_road_segments),
                                    as_scene_road_segment=scene_road_segments)
    geo_data = DataSceneStaticGeometry(e_b_Valid=True,
                                       s_RecvTimestamp=Timestamp(0, 0),
                                       s_ComputeTimestamp=Timestamp(0, 0),
                                       e_Cnt_num_lane_segments=len(scene_lane_segments_geo),
                                       as_scene_lane_segments=scene_lane_segments_geo)
    nav_plan_data = DataNavigationPlan(e_b_Valid=True,
                                       e_Cnt_num_road_segments=0,
                                       a_i_road_segment_ids=np.array([]))
    scene = SceneStatic(s_Header=header, s_MapOrigin=map_origin, s_SceneStaticGeometryData=geo_data, s_SceneStaticBaseData=base_data, s_NavigationPlanData=nav_plan_data)
    return scene


def add_lane_split(scene_static: SceneStatic, road_seg_split_idx: int):   
    ref_scene_road_segment = scene_static.s_SceneStaticBaseData.as_scene_road_segment[road_seg_split_idx]
    ref_scene_road_segment_id = ref_scene_road_segment.e_i_road_segment_id

    current_lane_seg_ids = []
    scene_lane_segments_base = []
    scene_lane_segments_geo = []
    # Create NUM_SPLIT_LANE_SEGS_LONG in sequence
    for lane_idx in range(NUM_SPLIT_LANE_SEGS_LONG):
        road_segment_id = ROAD_ID_SPLIT * 10 + (ref_scene_road_segment_id % (ROAD_ID * 10))  # Ones place from ref, tens from road id
        lane_id = road_segment_id * 10 + lane_idx        
        current_lane_seg_ids.append(lane_id)

        # Starting nom path pt = new path point
        if lane_idx == 0:
            ref_lane = scene_static.s_SceneStaticGeometryData.as_scene_lane_segments[road_seg_split_idx * 2]
            ref_nominal_path_pt = ref_lane.a_nominal_path_points[int(NUM_NOM_PATH_PTS/2), :]
              
            starting_nominal_path_pt = np.empty(len(NominalPathPoint))    
            starting_nominal_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value] = ref_nominal_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value] + HALF_LANE_WIDTH
            starting_nominal_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value] = ref_nominal_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value]
            starting_nominal_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_heading.value] = 0 
            starting_nominal_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_il_curvature.value] = 0
            starting_nominal_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_il2_curvature_rate.value] = 0
            starting_nominal_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_cross_slope.value] = 0
            starting_nominal_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_along_slope.value] = 0
            starting_nominal_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] = 0 
            starting_nominal_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_left_offset.value] = HALF_LANE_WIDTH
            starting_nominal_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_right_offset.value] = HALF_LANE_WIDTH
        else:
            starting_nominal_path_pt = copy.deepcopy(scene_lane_segments_geo[lane_idx-1].a_nominal_path_points[-1])

        # Nominal path points
        nominal_points = []
        for pt_idx in range(NUM_NOM_PATH_PTS):                
            if pt_idx == 0:
                point = starting_nominal_path_pt
            else:
                point = copy.deepcopy(nominal_points[pt_idx - 1])
                point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value] += DS
                point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] = pt_idx * DS
            nominal_points.append(point)

        # Boundary points
        left_boundary_point = BoundaryPoint(
            MapLaneMarkerType.MapLaneMarkerType_SolidSingleLine_BottsDots, 0, (DS * (NUM_NOM_PATH_PTS - 1))
        )
        right_boundary_point = BoundaryPoint(
            MapLaneMarkerType.MapLaneMarkerType_SolidSingleLine_BottsDots, 0, (DS * (NUM_NOM_PATH_PTS - 1))
        )

        # Downstream, all except last
        downstream_lane_count = 0 if lane_idx == (NUM_SPLIT_LANE_SEGS_LONG - 1) else 1
        downstream_id = (lane_id + 1) if downstream_lane_count == 1 else -1
        
        # Upstream = all, special case lane_idx ==0
        upstream_lane_count = 1 if lane_idx != 0 else -1
        upstream_id = (lane_id - 1) if upstream_lane_count == 1 else -1 
        if lane_idx == 0:
            upstream_lane_count = 1
            upstream_id = scene_static.s_SceneStaticBaseData.as_scene_lane_segments[road_seg_split_idx * NUM_STRAIGHT_LANE_SEGS_CROSS].e_i_lane_segment_id

        downstream_lane_segment_connectivity = \
            LaneSegmentConnectivity(downstream_id, ManeuverType.STRAIGHT_CONNECTION) \
            if downstream_lane_count == 1 else LaneSegmentConnectivity(-1, ManeuverType.STRAIGHT_CONNECTION)
        upstream_lane_segment_connectivity = \
            LaneSegmentConnectivity(upstream_id, ManeuverType.STRAIGHT_CONNECTION) \
            if upstream_lane_count == 1 else LaneSegmentConnectivity(-1, ManeuverType.STRAIGHT_CONNECTION)

        # No adjacent lanes, single lane split
        right_adj_count = 0
        right_adj_lane = AdjacentLane(-1, MovingDirection.Adjacent_or_same_dir, MapLaneType.LocalRoadLane)
        left_adj_count = 0
        left_adj_lane = AdjacentLane(-1, MovingDirection.Adjacent_or_same_dir, MapLaneType.LocalRoadLane)

        # Make SceneLaneSegment
        scene_lane_segment_base = SceneLaneSegmentBase(
            e_i_lane_segment_id=lane_id,
            e_i_road_segment_id=road_segment_id,
            e_e_lane_type=MapLaneType.LocalRoadLane,
            e_Cnt_static_traffic_flow_control_count=0,
            as_static_traffic_flow_control=[],
            e_Cnt_dynamic_traffic_flow_control_count=0,
            as_dynamic_traffic_flow_control=[],
            e_Cnt_left_adjacent_lane_count=left_adj_count,
            as_left_adjacent_lanes=[left_adj_lane],
            e_Cnt_right_adjacent_lane_count=right_adj_count,
            as_right_adjacent_lanes=[right_adj_lane],
            e_Cnt_downstream_lane_count=downstream_lane_count,
            as_downstream_lanes=[downstream_lane_segment_connectivity],
            e_Cnt_upstream_lane_count=upstream_lane_count,
            as_upstream_lanes=[upstream_lane_segment_connectivity],
            e_v_nominal_speed=NOM_SPEED,
            e_i_downstream_road_intersection_id=0,
            e_Cnt_lane_coupling_count=0,
            as_lane_coupling=[],
            e_Cnt_num_active_lane_attributes=0,                
            a_i_active_lane_attribute_indices=np.array([]),
            a_cmp_lane_attributes=np.array([]),               
            a_cmp_lane_attribute_confidences=np.array([])
        )
        scene_lane_segment_geo = SceneLaneSegmentGeometry(
            e_i_lane_segment_id=lane_id,
            e_i_road_segment_id=road_segment_id,
            e_Cnt_nominal_path_point_count=len(nominal_points),
            a_nominal_path_points=np.asarray(nominal_points),
            e_Cnt_left_boundary_points_count=1,
            as_left_boundary_points=[left_boundary_point],
            e_Cnt_right_boundary_points_count=1,
            as_right_boundary_points=[right_boundary_point]
        )

        scene_lane_segments_base.append(scene_lane_segment_base)
        scene_lane_segments_geo.append(scene_lane_segment_geo) 

    scene_road_segment = SceneRoadSegment(e_i_road_segment_id=road_segment_id,
                                          e_i_road_id=ROAD_ID_SPLIT,
                                          e_Cnt_lane_segment_id_count=NUM_SPLIT_LANE_SEGS_LONG,
                                          a_i_lane_segment_ids=np.array(current_lane_seg_ids),
                                          e_e_road_segment_type=MapRoadSegmentType.Normal,
                                          e_Cnt_upstream_segment_count=1,
                                          a_i_upstream_road_segment_ids=np.array([ref_scene_road_segment_id]),
                                          e_Cnt_downstream_segment_count=0,
                                          a_i_downstream_road_segment_ids=np.array([-1]))
    
    # Fix connectivity of straight scene road segment
    ref_scene_road_segment.e_Cnt_upstream_segment_count += 1
    ref_scene_road_segment.a_i_upstream_road_segment_ids = np.append(ref_scene_road_segment.a_i_upstream_road_segment_ids,
                                                                     ref_scene_road_segment_id)

    ref_scene_lane = scene_static.s_SceneStaticBaseData.as_scene_lane_segments[road_seg_split_idx * NUM_STRAIGHT_LANE_SEGS_CROSS]    
    ref_scene_lane.e_Cnt_downstream_lane_count += 1
    ref_scene_lane.as_downstream_lanes.append(
        LaneSegmentConnectivity(scene_lane_segments_base[0].e_i_lane_segment_id, 
                                ManeuverType.RIGHT_TURN_CONNECTION)
    )

    scene_static.s_SceneStaticBaseData.e_Cnt_num_lane_segments += NUM_SPLIT_LANE_SEGS_LONG
    scene_static.s_SceneStaticGeometryData.e_Cnt_num_lane_segments += NUM_SPLIT_LANE_SEGS_LONG
    for lane in scene_lane_segments_base:
        scene_static.s_SceneStaticBaseData.as_scene_lane_segments.append(lane) 
    for lane in scene_lane_segments_geo:
        scene_static.s_SceneStaticGeometryData.as_scene_lane_segments.append(lane)
    scene_static.s_SceneStaticBaseData.e_Cnt_num_road_segments += 1
    scene_static.s_SceneStaticBaseData.as_scene_road_segment.append(scene_road_segment)
