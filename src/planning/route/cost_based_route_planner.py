from collections import OrderedDict

from decision_making.src.messages.scene_static_message import SceneStatic,DataSceneStaticLite,DataNavigationPlan
from decision_making.src.messages.route_plan_message import RoutePlan,RoutePlanLaneSegment, DataRoutePlan

from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures import LaneMappingStatusType, LaneConstructionType,\
     GMAuthorityType, MapLaneDirection, RoutePlanLaneSegmentAttr, TsSYS_RoutePlanLaneSegment, TsSYS_DataRoutePlan
     
from route_planner import RoutePlanner
from decision_making.src.messages.scene_static_enums import RoutePlanLaneSegmentAttr, LaneMappingStatusType, MapLaneDirection, \
     GMAuthorityType, LaneConstructionType


class RoutePlannerInputData():
    
    def Update_DictData(self,Scene:DataSceneStaticLite):
        for i in range(Scene.e_Cnt_num_lane_segments):
            self.LaneSegmentDict[Scene.as_scene_lane_segment[i].e_i_lane_segment_id] = \
            Scene.s_SceneStaticData.as_scene_lane_segment[i]
        for i in range(Scene.e_Cnt_num_road_segments):
            self.RoadSegmentDict[Scene.as_scene_road_segment[i].e_Cnt_road_segment_id] = \
            Scene.s_SceneStaticData.as_scene_road_segment[i]
    pass

    def Update_RoutePlanData(self,Nav:DataNavigationPlan):
        self.route_roadsegments = Nav.a_i_road_segment_ids
        for road_seg in range(self.route_roadsegments):
            lane_seg = []
            for j in range(self.RoadSegmentDict[road_seg].e_Cnt_lane_segment_id_count):
                lane_seg.append(self.RoadSegmentDict[road_seg].a_Cnt_lane_segment_id[j])
            self.RoadSegmentDict[road_seg]=lane_seg
        
    def __init__(self,Scene:DataSceneStaticLite,Nav:DataNavigationPlan):
        self.route_roadsegments = [] # list: ordered RouteSegment's upto the variable size e_Cnt_num_road_segments
        self.route_lanesegments = OrderedDict() # dict:  key - road segment IDs (sorted as in routeplan) , value - list(LaneSegmentID)
        self.LaneSegmentDict = {} # dict : key - lane segment ID, value - LaneSegmentLite. 
        #The dict should contain all the lane segments listed in self.route_segment_ids. 
        self.RoadSegmentDict = {} # dict : key - road segment ID, value - Road Segments. 
        #The dict should contain all the lane segments listed in self.route_segment_ids. 
        self.Update_DictData(Scene)
        self.Update_RoutePlanData(Nav)




class CostBasedRoutePlanner(RoutePlanner):
    """Add comments"""
    
    @staticmethod
    def MappingStatusAttrBsdOccCost(MappingStatusAttr:LaneMappingStatusType): # Normalized cost (0 to 1). Current implementation is binary cost. 
        if((MappingStatusAttr==LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap) or 
           (MappingStatusAttr==LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_MDMap)):
            return 0
        return 1
    
    @staticmethod    
    def ConstructionZoneAttrBsdOccCost(ConstructionZoneAttr:LaneConstructionType): # Normalized cost (0 to 1). Current implementation is binary cost. 
        if((ConstructionZoneAttr==LaneConstructionType.CeSYS_e_LaneConstructionType_Normal) or 
           (ConstructionZoneAttr==LaneConstructionType.CeSYS_e_LaneConstructionType_Unknown)):
            return 0
        return 1
    
    @staticmethod
    def LaneRouteDirAttrBsdOccCost(LaneRouteDirAttr:MapLaneDirection): # Normalized cost (0 to 1). Current implementation is binary cost. 
        if((LaneRouteDirAttr==MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle) or 
           (LaneRouteDirAttr==MapLaneDirection.CeSYS_e_MapLaneDirection_Left_Towards_HostVehicle) or
            (LaneRouteDirAttr==MapLaneDirection.CeSYS_e_MapLaneDirection_Right_Towards_HostVehicle)):
            return 0
        return 1
    
    @staticmethod
    def GMAuthorityAttrBsdOccCost(GMAuthorityAttr:GMAuthorityType): # Normalized cost (0 to 1). Current implementation is binary cost. 
        if(GMAuthorityAttr==GMAuthorityType.CeSYS_e_GMAuthorityType_None):
            return 0
        return 1
    
    @staticmethod
    def LaneAttrBsdOccCost(EnumIdx:int,LaneAttr:int): # Equivalent of a switch-case logic 
        switcher = { 
        RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus: CostBasedRoutePlanner.MappingStatusAttrBsdOccCost(LaneAttr) , 
        RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA: CostBasedRoutePlanner.GMAuthorityAttrBsdOccCost(LaneAttr), 
        RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction: CostBasedRoutePlanner.ConstructionZoneAttrBsdOccCost(LaneAttr), 
        RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Direction: CostBasedRoutePlanner.LaneRouteDirAttrBsdOccCost(LaneAttr),
        } 
        return switcher[EnumIdx]

    def plan(self,RouteData:RoutePlannerInputData)->DataRoutePlan: # TODO: Set function annotaion
        """Add comments"""
        NewRoutePlan = DataRoutePlan
        for reverseroadsegidx, (roadsegID,lanesegIDs) in reversed(enumerate(RouteData.route_lanesegments.items())):
            roadsegidx = RouteData.route_lanesegments.len()-reverseroadsegidx
            for lanesegidx, lanesegID in enumerate(lanesegIDs):
                lanesegData = RouteData.LaneSegmentDict[lanesegID]
                LaneOccCost = 0
                LaneEndCost = 0
                for Idx in range(lanesegData.e_Cnt_num_active_lane_attributes):
                    LaneAttrIdx = lanesegData.e_i_active_lane_attribute_indices[Idx]
                    LaneAttr = lanesegData.e_cmp_lane_attributes[LaneAttrIdx]
                    LaneAttrConf = lanesegData.e_cmp_lane_attribute_confidences[LaneAttrIdx]
                    if (LaneAttrConf<0.7):
                        continue
                    LaneOccCost = LaneOccCost + CostBasedRoutePlanner.LaneAttrBsdOccCost(Idx,LaneAttr)
                LaneOccCost = min(LaneOccCost,1)
                if (LaneOccCost==1):
                    LaneEndCost = 1 # Can't occupy the lane, can't end up in the lane
                elif (reverseroadsegidx==0):
                    LaneEndCost = 0
                else:
                    MinDwnStreamLaneOccCost = 1
                    for Idx in range(lanesegData.e_Cnt_downstream_lane_count):
                        DownStreamlanesegID = lanesegData.as_downstream_lanes[Idx]
                        NextRoadSegLanes = RouteData.route_lanesegments[reverseroadsegidx-1]
                        if DownStreamlanesegID in NextRoadSegLanes:
                            DownStreamlanesegIdx = NextRoadSegLanes.index(DownStreamlanesegID)
                            DownStreamLaneOccCost= NewRoutePlan.as_route_plan_lane_segments[roadsegidx+1][DownStreamlanesegIdx]
                            MinDwnStreamLaneOccCost = min(MinDwnStreamLaneOccCost,DownStreamLaneOccCost)
                    LaneEndCost = MinDwnStreamLaneOccCost
                            
                    
                NewRoutePlan.as_route_plan_lane_segments[roadsegidx][lanesegidx]=RoutePlanLaneSegment(lanesegID,LaneOccCost,LaneEndCost)
            NewRoutePlan.e_Cnt_num_road_segments = reverseroadsegidx
            NewRoutePlan.a_i_road_segment_ids[roadsegidx] = roadsegID
            NewRoutePlan.a_Cnt_num_lane_segments[roadsegidx] = lanesegidx
            

                
        return NewRoutePlan


