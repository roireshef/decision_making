from decision_making.src.messages.route_plan_message import RoutePlan,RoutePlanLaneSegment, DataRoutePlan

from common_data.interface.Rte_Types.python.sub_structures import TsSYSRoutePlanLaneSegment, TsSYSDataRoutePlan
     
from decision_making.src.planning.route.route_planner import RoutePlanner, RoutePlannerInputData
from decision_making.src.messages.scene_static_enums import RoutePlanLaneSegmentAttr, LaneMappingStatusType, MapLaneDirection, \
     GMAuthorityType, LaneConstructionType

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
        return switcher[RoutePlanLaneSegmentAttr(EnumIdx)]

    def plan(self, RouteData: RoutePlannerInputData) -> DataRoutePlan: # TODO: Set function annotaion
        """Add comments"""
        NewRoutePlan = DataRoutePlan
        NumOfRoadSeg = RouteData.route_lanesegments.__len__()
        for reverseroadsegidx, (roadsegID,lanesegIDs) in enumerate(reversed(RouteData.route_lanesegments.items())):
            roadsegidx = NumOfRoadSeg -reverseroadsegidx
            AllRouteLanesInThisRoadSeg = []
            for lanesegidx, lanesegID in enumerate(lanesegIDs):
                lanesegData = RouteData.LaneSegmentDict[lanesegID]
                LaneOccCost = 0
                LaneEndCost = 0
                
                # -------------------------------------------
                # Calculate lane occupancy costs
                # -------------------------------------------
                for Idx in range(lanesegData.e_Cnt_num_active_lane_attributes):
                    LaneAttrIdx = lanesegData.a_i_active_lane_attribute_indices[Idx]
                    LaneAttr = lanesegData.a_cmp_lane_attributes[LaneAttrIdx]
                    LaneAttrConf = lanesegData.a_cmp_lane_attribute_confidences[LaneAttrIdx]
                    if (LaneAttrConf<0.7):
                        continue
                    LaneOccCost = LaneOccCost + CostBasedRoutePlanner.LaneAttrBsdOccCost(LaneAttrIdx,LaneAttr)
                # Normalize to the [0, 1] range
                LaneOccCost = max(min(LaneOccCost,1),0)
                
                # -------------------------------------------
                # Calculate lane end costs
                # -------------------------------------------
                if (LaneOccCost==1):
                    LaneEndCost = 1 # Can't occupy the lane, can't end up in the lane
                elif (reverseroadsegidx>0):# if reverseroadsegidx=0 a.k.a last lane in current route view lane end cost = 0 
                    # as we don't know the next segment. 
                    MinDwnStreamLaneOccCost = 1
                    # Search iteratively for the next segment lanes that are downstream to the current lane and in the route.
                    # At this point assign the end cost of current lane = Min occ costs of all downstream lanes
                    for Idx in range(lanesegData.e_Cnt_downstream_lane_count): # search through all downstream lanes to to current lane
                        DownStreamlanesegID = lanesegData.as_downstream_lanes[Idx]
                        NextRoadSegLanes = RouteData.route_lanesegments[reverseroadsegidx-1] # All lane IDs in next roadsegment in route
                        if DownStreamlanesegID in NextRoadSegLanes: # verify if the downstream lane is in the route
                            DownStreamlanesegIdx = NextRoadSegLanes.index(DownStreamlanesegID)
                            DownStreamLaneOccCost= NewRoutePlan.as_route_plan_lane_segments[roadsegidx+1][DownStreamlanesegIdx]
                            # confirm -> roadsegidx+1 in the RoutPlan == reverseroadsegidx-1 in the reversed RoutePlan
                            MinDwnStreamLaneOccCost = min(MinDwnStreamLaneOccCost,DownStreamLaneOccCost)
                    LaneEndCost = MinDwnStreamLaneOccCost
                            
                #print("lanesegID ",lanesegID)
                #print("LaneOccCost ",LaneOccCost)    
                #print("LaneEndCost ",LaneEndCost)
                AllRouteLanesInThisRoadSeg.append(RoutePlanLaneSegment(e_i_lane_segment_id= lanesegID, \
                    e_cst_lane_occupancy_cost = LaneOccCost,e_cst_lane_end_cost = LaneEndCost))
            NewRoutePlan.e_Cnt_num_road_segments = reverseroadsegidx
            NewRoutePlan.a_i_road_segment_ids.append(roadsegID)
            NewRoutePlan.a_Cnt_num_lane_segments.append(lanesegidx)
            NewRoutePlan.as_route_plan_lane_segments.append(AllRouteLanesInThisRoadSeg)
            

                
        return NewRoutePlan


