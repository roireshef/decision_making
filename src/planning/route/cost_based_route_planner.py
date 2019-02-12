import numpy as np
import pprint
from typing import List


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
        #print("ConstructionZoneAttr ",ConstructionZoneAttr)
        if((ConstructionZoneAttr==LaneConstructionType.CeSYS_e_LaneConstructionType_Normal) or 
           (ConstructionZoneAttr==LaneConstructionType.CeSYS_e_LaneConstructionType_Unknown)):
            return 0
        return 1
    
    @staticmethod
    def LaneRouteDirAttrBsdOccCost(LaneRouteDirAttr:MapLaneDirection): # Normalized cost (0 to 1). Current implementation is binary cost. 
        #print("LaneRouteDirAttr ",LaneRouteDirAttr)
        if((LaneRouteDirAttr==MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle) or 
           (LaneRouteDirAttr==MapLaneDirection.CeSYS_e_MapLaneDirection_Left_Towards_HostVehicle) or
            (LaneRouteDirAttr==MapLaneDirection.CeSYS_e_MapLaneDirection_Right_Towards_HostVehicle)):
            return 0
        return 1
    
    @staticmethod
    def GMAuthorityAttrBsdOccCost(GMAuthorityAttr:GMAuthorityType): # Normalized cost (0 to 1). Current implementation is binary cost. 
        #print("GMAuthorityAttr ",GMAuthorityAttr)
        if(GMAuthorityAttr==GMAuthorityType.CeSYS_e_GMAuthorityType_None):
            return 0
        return 1
    
    @staticmethod
    def LaneAttrBsdOccCost(EnumIdx:int,LaneAttr:int): # if else logic 
        if(EnumIdx==RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus):
            return CostBasedRoutePlanner.MappingStatusAttrBsdOccCost(LaneAttr)
        elif(EnumIdx==RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA):
            return CostBasedRoutePlanner.GMAuthorityAttrBsdOccCost(LaneAttr)
        elif(EnumIdx==RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction):
            return CostBasedRoutePlanner.ConstructionZoneAttrBsdOccCost(LaneAttr)
        elif(EnumIdx==RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Direction):
            return CostBasedRoutePlanner.LaneRouteDirAttrBsdOccCost(LaneAttr)
        else:
           print("Error EnumIdx not supported ",EnumIdx)
           return 0
        

    


    def plan(self, RouteData: RoutePlannerInputData) -> DataRoutePlan: # TODO: Set function annotaion
        """Add comments"""
        #
        # NewRoutePlan = DataRoutePlan()
        valid = True
        num_road_segments = len(RouteData.route_roadsegments)
        a_i_road_segment_ids = []
        a_Cnt_num_lane_segments = []
        as_route_plan_lane_segments = []

        for reverseroadsegidx, (roadsegID,lanesegIDs) in enumerate(reversed(RouteData.route_lanesegments.items())):
            # roadsegidx = num_road_segments - reverseroadsegidx
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
                    LaneAttrOccCost = CostBasedRoutePlanner.LaneAttrBsdOccCost(LaneAttrIdx,LaneAttr)
                    #print("LaneAttrConf ",LaneAttrConf," LaneAttrIdx",LaneAttrIdx," LaneAttr ",LaneAttr," LaneAttrOccCost ",LaneAttrOccCost)
                    LaneOccCost = LaneOccCost + LaneAttrOccCost
                # Normalize to the [0, 1] range
                LaneOccCost = max(min(LaneOccCost,1),0)
                
                
                # -------------------------------------------
                # Calculate lane end costs
                # -------------------------------------------
                
                
                if (reverseroadsegidx==0) : # the last road segment in the route; put all endcosts to 0
                    LaneEndCost = 0
                elif (LaneOccCost==1):
                    LaneEndCost = 1 # Can't occupy the lane, can't end up in the lane
                elif (reverseroadsegidx>0):# if reverseroadsegidx=0 a.k.a last lane in current route view lane end cost = 0 
                    # as we don't know the next segment. 
                    MinDwnStreamLaneOccCost = 1
                    # Search iteratively for the next segment lanes that are downstream to the current lane and in the route.
                    # At this point assign the end cost of current lane = Min occ costs (of all downstream lanes)
                    for Idx in range(lanesegData.e_Cnt_downstream_lane_count): # search through all downstream lanes to to current lane
                        DownStreamlanesegID = lanesegData.as_downstream_lanes[Idx].e_i_lane_segment_id
                        NextRoadSegLanes = RouteData.route_lanesegments[prev_roadsegID] # All lane IDs in downstream roadsegment in route to the 
                        #currently indexed roadsegment in the loop
                        if DownStreamlanesegID in NextRoadSegLanes: # verify if the downstream lane is in the route
                            DownStreamlanesegIdx = NextRoadSegLanes.index(DownStreamlanesegID)
                            DownStreamLaneOccCost = as_route_plan_lane_segments[reverseroadsegidx-1][DownStreamlanesegIdx].e_cst_lane_occupancy_cost
                            # DownStreamLaneOccCost= NewRoutePlan.as_route_plan_lane_segments[roadsegidx+1][DownStreamlanesegIdx]
                            # confirm -> roadsegidx+1 in the RoutPlan == reverseroadsegidx-1 in the reversed RoutePlan
                            MinDwnStreamLaneOccCost = min(MinDwnStreamLaneOccCost,DownStreamLaneOccCost)
                        else:
                            print(" DownStreamlanesegID ",DownStreamlanesegID," not found in NextRoadSegLanes ",NextRoadSegLanes)
                            print("prev_roadsegID",prev_roadsegID)
                    LaneEndCost = MinDwnStreamLaneOccCost
                
                            
                CurrRoutePlanLaneSegment = RoutePlanLaneSegment(e_i_lane_segment_id= lanesegID, \
                    e_cst_lane_occupancy_cost = LaneOccCost,e_cst_lane_end_cost = LaneEndCost)
                #print("CurrRoutePlanLaneSegment",CurrRoutePlanLaneSegment)
                AllRouteLanesInThisRoadSeg.append(CurrRoutePlanLaneSegment)
            prev_roadsegID = roadsegID
            # NewRoutePlan.e_Cnt_num_road_segments = reverseroadsegidx
            a_i_road_segment_ids.insert(0,roadsegID)
            a_Cnt_num_lane_segments.insert(0,lanesegidx+1)
            as_route_plan_lane_segments.insert(0,AllRouteLanesInThisRoadSeg)


            #print("AllRouteLanesInThisRoadSeg ")
            #for RouteLanes in AllRouteLanesInThisRoadSeg:
            #    print(RouteLanes)
            #NewRoutePlan.as_route_plan_lane_segments.append(AllRouteLanesInThisRoadSeg)

        NewRoutePlan = DataRoutePlan(e_b_is_valid = valid, e_Cnt_num_road_segments= num_road_segments, a_i_road_segment_ids=np.array(a_i_road_segment_ids), \
                                        a_Cnt_num_lane_segments= np.array(a_Cnt_num_lane_segments), as_route_plan_lane_segments=as_route_plan_lane_segments )

        return NewRoutePlan


