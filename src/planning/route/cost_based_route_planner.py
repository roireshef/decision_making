from collections import OrderedDict

from decision_making.src.messages.scene_static_lite_message import SceneStaticLite,DataSceneStaticLite
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures import TeSYS_LaneMappingStatusType, TeSYS_LaneConstructionType,\
     TeSYS_GMAuthorityType, TeSYS_MapLaneDirection, TeSYS_RoutePlanLaneSegmentAttr, TsSYS_RoutePlanLaneSegment, TsSYS_DataRoutePlan

from route_planner import RoutePlanner


class RoutePlannerData():

    def Update_DictData(self,Scene:SceneStaticLite):
        for i in range(Scene.s_SceneStaticData.e_Cnt_num_lane_segments):
            self.LaneSegmentDict[Scene.s_SceneStaticData.as_scene_lane_segment[i].e_i_lane_segment_id] = \
            Scene.s_SceneStaticData.as_scene_lane_segment[i]
        for i in range(Scene.s_SceneStaticData.e_Cnt_num_road_segments):
            self.RoadSegmentDict[Scene.s_SceneStaticData.as_scene_road_segment[i].e_Cnt_road_segment_id] = \
            Scene.s_SceneStaticData.as_scene_road_segment[i]
    pass

    def Update_RoutePlanData(self,Scene:SceneStaticLite):
        self.route_roadsegments = Scene.s_NavigationPlanData.a_i_road_segment_ids
        for road_seg in range(self.route_roadsegments):
            lane_seg = []
            for j in range(self.RoadSegmentDict[road_seg].e_Cnt_lane_segment_id_count):
                lane_seg.append(self.RoadSegmentDict[road_seg].a_Cnt_lane_segment_id[j])
            self.RoadSegmentDict[road_seg]=lane_seg

    def __init__(self,Scene:SceneStaticLite):
        self.route_roadsegments = [] # list: ordered RouteSegment's upto the variable size e_Cnt_num_road_segments
        self.route_lanesegments = OrderedDict() # dict:  key - road segment IDs (sorted as in routeplan) , value - list(LaneSegmentID)
        self.LaneSegmentDict = {} # dict : key - lane segment ID, value - LaneSegmentLite.
        #The dict should contain all the lane segments listed in self.route_segment_ids.
        self.RoadSegmentDict = {} # dict : key - road segment ID, value - Road Segments.
        #The dict should contain all the lane segments listed in self.route_segment_ids.
        self.Update_DictData(Scene)
        self.Update_RoutePlanData(Scene)





class RoutePlanData(metaclass=ABCMeta):
    def __init__(self,RouteSegs: List[RouteSegment]):
    self.route_segments = RouteSegs # list: ordered RouteSegment's upto the variable size e_Cnt_num_road_segments
    self.LaneSegments = {} # dict : key - segment ID, value - LaneSegmentLite. The dict should contain all the lane segments listed in self.route_segment_ids.

    @abstractmethod
    def Update_LaneSegmentData(self):
        pass

    @abstractmethod
    def Update_RoutePlanData(self):

        pass

class DualCostRoutePlanner(RoutePlanner):
    """Add comments"""

    @staticmethod
    def MappingStatusAttrBsdOccCost(MappingStatusAttr:TeSYS_LaneMappingStatusType): # Normalized cost (0 to 1). Current implementation is binary cost.
        if((MappingStatusAttr==TeSYS_LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap) or
           (MappingStatusAttr==TeSYS_LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_MDMap)):
            return 0
        return 1

    @staticmethod
    def ConstructionZoneAttrBsdOccCost(ConstructionZoneAttr:TeSYS_LaneConstructionType): # Normalized cost (0 to 1). Current implementation is binary cost.
        if((ConstructionZoneAttr==TeSYS_LaneConstructionType.CeSYS_e_LaneConstructionType_Normal) or
           (ConstructionZoneAttr==TeSYS_LaneConstructionType.CeSYS_e_LaneConstructionType_Unknown)):
            return 0
        return 1

    @staticmethod
    def LaneRouteDirAttrBsdOccCost(LaneRouteDirAttr:TeSYS_MapLaneDirection): # Normalized cost (0 to 1). Current implementation is binary cost.
        if((LaneRouteDirAttr==TeSYS_MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle) or
           (LaneRouteDirAttr==TeSYS_MapLaneDirection.CeSYS_e_MapLaneDirection_Left_Towards_HostVehicle) or
            (LaneRouteDirAttr==TeSYS_MapLaneDirection.CeSYS_e_MapLaneDirection_Right_Towards_HostVehicle)):
            return 0
        return 1

    @staticmethod
    def GMAuthorityAttrBsdOccCost(GMAuthorityAttr:TeSYS_GMAuthorityType): # Normalized cost (0 to 1). Current implementation is binary cost.
        if(GMAuthorityAttr==TeSYS_GMAuthorityType.CeSYS_e_GMAuthorityType_None):
            return 0
        return 1

    @staticmethod
    def LaneAttrBsdOccCost(EnumIdx:int,LaneAttr:int): # Equivalent of a switch-case logic
        switcher = {
        TeSYS_RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus: CostBasedRoutePlanner.MappingStatusAttrBsdOccCost(LaneAttr) ,
        TeSYS_RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA: CostBasedRoutePlanner.GMAuthorityAttrBsdOccCost(LaneAttr),
        TeSYS_RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction: CostBasedRoutePlanner.ConstructionZoneAttrBsdOccCost(LaneAttr),
        TeSYS_RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Direction: CostBasedRoutePlanner.LaneRouteDirAttrBsdOccCost(LaneAttr),
        }
        return switcher[EnumIdx]

    def plan(self,RouteData:RoutePlannerData)->TsSYS_DataRoutePlan: # TODO: Set function annotaion
        """Add comments"""
        NewRoutePlan = TsSYS_DataRoutePlan
        for roadsegidx, (roadsegID,lanesegIDs) in enumerate(RouteData.route_lanesegments.items()):
            for lanesegidx, lanesegID in enumerate(lanesegIDs):
                lanesegData = RouteData.LaneSegmentDict[lanesegID]
                LaneOccCost = 0
                for Idx in range(lanesegData.e_Cnt_num_active_lane_attributes):
                    LaneAttrIdx = lanesegData.e_i_active_lane_attribute_indices[Idx]
                    LaneAttr = lanesegData.e_cmp_lane_attributes[LaneAttrIdx]
                    LaneAttrConf = lanesegData.e_cmp_lane_attribute_confidences[LaneAttrIdx]
                    if (LaneAttrConf<0.7):
                        continue
                    LaneOccCost = LaneOccCost + CostBasedRoutePlanner.LaneAttrBsdOccCost(Idx,LaneAttr)
                LaneOccCost = min(LaneOccCost,1)
                NewRoutePlan.as_route_plan_lane_segments[roadsegidx][lanesegidx]=SomeConstructor(lanesegID,LaneOccCost,0)
            NewRoutePlan.e_Cnt_num_road_segments = roadsegidx
            NewRoutePlan.a_i_road_segment_ids[roadsegidx] = roadsegID
            NewRoutePlan.a_Cnt_num_lane_segments[roadsegidx] = lanesegidx

        # Looping again over all the lanesegs for deriving the end cost from the following segment occupancy costs. Maybe there
        # is a more efficient (but more diff to read code ) way of handling it in the same loop where you try to populate the
        # previous segments end cost etc.But once you go to backtracking it will be more diffcult to handle anyway.
        for roadsegidx, (roadsegID,lanesegIDs) in enumerate(RouteData.route_lanesegments.items()):
            for lanesegidx, lanesegID in enumerate(lanesegIDs):
                if lanesegID in RoutePlannerData.LaneSegmentDict:
                    NewRoutePlan.as_route_plan_lane_segments[roadsegidx][lanesegidx].e_cst_lane_end_cost=0

        pass


