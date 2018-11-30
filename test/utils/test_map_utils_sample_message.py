from math import sqrt
import numpy as np

from common_data.interface.py.idl_generated_files.Rte_Types.TsSYS_SceneStatic import TsSYSSceneStatic
from common_data.interface.py.pubsub.Rte_Types_pubsub_topics import SCENE_STATIC
from common_data.lcm.config import pubsub_topics
from common_data.src.communication.pubsub.pubsub import PubSub

from decision_making.test.messages.static_scene_fixture import scene_static
from decision_making.src.messages.scene_common_messages import Header, MapOrigin, Timestamp
from decision_making.src.messages.scene_static_message import SceneStatic, DataSceneStatic, SceneRoadSegment, \
    MapRoadSegmentType, SceneLaneSegment, MapLaneType, LaneSegmentConnectivity, ManeuverType, NominalPathPoint, \
    MapLaneMarkerType, BoundaryPoint
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg

from decision_making.src.mapping.scene_model import SceneModel
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.utils.map_utils import MapUtils

# ---------------------------------------------------------
# Constants for static data and tests
# ---------------------------------------------------------
NUM_ROAD_SEGS = 1
NUM_LANE_SEGS = 4
NUM_NOM_PATH_PTS = 30

HORIZON_PERCEP_DIST = 50    # [m]
HALF_LANE_WIDTH = 2         # [m]
NOM_SPEED = 50.0            # [m/s]
FRENET_DS = 1.0             # [m]

ROAD_SEG_ID = 30
LANE_SEG_IDS = [301, 302, 303, 304]

START_EAST_X_VAL = -3e2 
START_NORTH_Y_VAL = -7e6

NAVIGATION_PLAN = NavigationPlanMsg(np.array([ROAD_SEG_ID]))

# ---------------------------------------------------------
# Defines sample scene message
# ---------------------------------------------------------
def scene_static_sample():
    scene_road_segments = []
    scene_road_segment = SceneRoadSegment(e_Cnt_road_segment_id=ROAD_SEG_ID, 
                                          e_Cnt_road_id=0,
                                          e_Cnt_lane_segment_id_count=NUM_ROAD_SEGS,
                                          a_Cnt_lane_segment_id=LANE_SEG_IDS,
                                          e_e_road_segment_type=MapRoadSegmentType.Normal,
                                          e_Cnt_upstream_segment_count=0,
                                          a_Cnt_upstream_road_segment_id=None,
                                          e_Cnt_downstream_segment_count=0,
                                          a_Cnt_downstream_road_segment_id=None)

    scene_road_segments.append(scene_road_segment)

    scene_lane_segments = []
    lane_idx = 0
    for lane_id in LANE_SEG_IDS:
        road_segment_id = ROAD_SEG_ID
        
        nominal_points = []
        for i in range(NUM_NOM_PATH_PTS):
            point = np.empty(len(NominalPathPoint))
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value] = START_EAST_X_VAL + i
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value] = START_NORTH_Y_VAL + i
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_heading.value] = 0 
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_il_curvature.value] = 0
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_il2_curvature_rate.value] = 0
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_cross_slope.value] = 0
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_along_slope.value] = 0
            if i == 0:
                point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] = 0 
            else:
                prev_pt = nominal_points[i-1]
                point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] = sqrt(pow(prev_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value], 2) + 
                                                                                  pow(prev_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value], 2))
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_left_offset.value] = HALF_LANE_WIDTH
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_right_offset.value] = HALF_LANE_WIDTH
            nominal_points.append(point)

        left_boundry_point = [BoundaryPoint(MapLaneMarkerType.MapLaneMarkerType_SolidSingleLine_BottsDots, 0, (FRENET_DS * (NUM_NOM_PATH_PTS-1)))]
        right_boundry_point = [BoundaryPoint(MapLaneMarkerType.MapLaneMarkerType_SolidSingleLine_BottsDots, 0, (FRENET_DS * (NUM_NOM_PATH_PTS-1)))]

        downstream_id = LANE_SEG_IDS[lane_idx + 1] if lane_idx < (NUM_LANE_SEGS - 1) else -1 
        upstream_id = LANE_SEG_IDS[lane_idx - 1] if lane_idx > 0 else -1 
        # print("Lane ", lane_id, ": downstream from lane ",downstream_id, ", upstream from lane ", upstream_id)    

        downstream_lane_segment_connectivity = LaneSegmentConnectivity(downstream_id, ManeuverType.STRAIGHT_CONNECTION) if downstream_id != -1 else None
        upstream_lane_segment_connectivity = LaneSegmentConnectivity(upstream_id, ManeuverType.STRAIGHT_CONNECTION) if upstream_id != -1 else None

        scene_lane_segments.append(SceneLaneSegment(e_i_lane_segment_id=lane_id,
                                                    e_i_road_segment_id=road_segment_id,
                                                    e_e_lane_type=MapLaneType.LocalRoadLane,
                                                    e_Cnt_static_traffic_flow_control_count=0,
                                                    as_static_traffic_flow_control=[],
                                                    e_Cnt_dynamic_traffic_flow_control_count=0,
                                                    as_dynamic_traffic_flow_control=[],
                                                    e_Cnt_left_adjacent_lane_count=0,
                                                    as_left_adjacent_lanes=[],
                                                    e_Cnt_right_adjacent_lane_count=0,
                                                    as_right_adjacent_lanes=[],
                                                    e_Cnt_downstream_lane_count=1,
                                                    as_downstream_lanes=[downstream_lane_segment_connectivity],
                                                    e_Cnt_upstream_lane_count=1,
                                                    as_upstream_lanes=[upstream_lane_segment_connectivity],
                                                    e_v_nominal_speed=NOM_SPEED,
                                                    e_Cnt_nominal_path_point_count=len(nominal_points),
                                                    a_nominal_path_points=np.asarray(nominal_points),
                                                    e_Cnt_left_boundary_points_count=len(left_boundry_point),
                                                    as_left_boundary_points=left_boundry_point,
                                                    e_Cnt_right_boundary_points_count=len(right_boundry_point),
                                                    as_right_boundary_points=right_boundry_point,
                                                    e_i_downstream_road_intersection_id=0,
                                                    e_Cnt_lane_coupling_count=0,
                                                    as_lane_coupling=[]))
        lane_idx += 1

    header = Header(e_Cnt_SeqNum=0, s_Timestamp=Timestamp(0, 0), e_Cnt_version=0)
    map_origin = MapOrigin(e_phi_latitude=.0, e_phi_longitude=.0, e_l_altitude=.0, s_Timestamp=Timestamp(0,0))
    data = DataSceneStatic(e_b_Valid=True,
                           s_ComputeTimestamp=Timestamp(0, 0),
                           e_l_perception_horizon_front=HORIZON_PERCEP_DIST,
                           e_l_perception_horizon_rear=HORIZON_PERCEP_DIST,
                           e_Cnt_num_lane_segments=len(scene_lane_segments),
                           as_scene_lane_segment=scene_lane_segments,
                           e_Cnt_num_road_intersections=0,
                           as_scene_road_intersection=[],
                           e_Cnt_num_road_segments=len(scene_road_segments),
                           as_scene_road_segment=scene_road_segments)

    scene = SceneStatic(s_Header=header, s_MapOrigin=map_origin, s_Data=data)
    return scene



def test_getLookaheadFrenetFrame_simple(scene_static: SceneStatic):
    pass


def main():
    scene_static = scene_static_sample()

    # Add scene to SceneModel
    SceneModel.get_instance().add_scene_static(scene_static)

    # Check SceneModel is populated
    assert SceneModel.get_instance().get_scene_static().s_Data.e_Cnt_num_road_segments == NUM_ROAD_SEGS
    assert SceneModel.get_instance().get_scene_static().s_Data.e_Cnt_num_lane_segments == NUM_LANE_SEGS
    assert SceneModel.get_instance().get_scene_static().s_Data.as_scene_road_segment[0].e_Cnt_road_segment_id == ROAD_SEG_ID
    
    lane_idx = 0
    for lane in SceneModel.get_instance().get_scene_static().s_Data.as_scene_lane_segment: 
        assert lane.e_i_lane_segment_id == LANE_SEG_IDS[lane_idx]
        lane_idx += 1

    #######################################################
    # Insert tests here
    #######################################################


if __name__ == '__main__':
    main()
