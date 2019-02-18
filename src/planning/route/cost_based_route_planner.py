import numpy as np
import pprint
from typing import List
import traceback
from logging import Logger

from decision_making.src.messages.route_plan_message import RoutePlan, RoutePlanLaneSegment, DataRoutePlan
from decision_making.src.messages.scene_static_message import SceneLaneSegmentBase
from decision_making.src.exceptions import LaneNotFound


from common_data.interface.Rte_Types.python.sub_structures import TsSYSRoutePlanLaneSegment, TsSYSDataRoutePlan

from decision_making.src.planning.route.route_planner import RoutePlanner, RoutePlannerInputData
from decision_making.src.messages.scene_static_enums import RoutePlanLaneSegmentAttr, LaneMappingStatusType, MapLaneDirection, \
    GMAuthorityType, LaneConstructionType


class CostBasedRoutePlanner(RoutePlanner):
    """Add comments"""

    @staticmethod
    # Normalized cost (0 to 1). Current implementation is binary cost.
    def _MappingStatusAttrBsdOccCost(MappingStatusAttr: LaneMappingStatusType)->float:
        if((MappingStatusAttr == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap) or
           (MappingStatusAttr == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_MDMap)):
            return 0
        return 1

    @staticmethod
    # Normalized cost (0 to 1). Current implementation is binary cost.
    def _ConstructionZoneAttrBsdOccCost(ConstructionZoneAttr: LaneConstructionType)->float:
        #print("ConstructionZoneAttr ",ConstructionZoneAttr)
        if((ConstructionZoneAttr == LaneConstructionType.CeSYS_e_LaneConstructionType_Normal) or
           (ConstructionZoneAttr == LaneConstructionType.CeSYS_e_LaneConstructionType_Unknown)):
            return 0
        return 1

    @staticmethod
    # Normalized cost (0 to 1). Current implementation is binary cost.
    def _LaneRouteDirAttrBsdOccCost(LaneRouteDirAttr: MapLaneDirection)->float:
        #print("LaneRouteDirAttr ",LaneRouteDirAttr)
        if((LaneRouteDirAttr == MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle) or
           (LaneRouteDirAttr == MapLaneDirection.CeSYS_e_MapLaneDirection_Left_Towards_HostVehicle) or
                (LaneRouteDirAttr == MapLaneDirection.CeSYS_e_MapLaneDirection_Right_Towards_HostVehicle)):
            return 0
        return 1

    @staticmethod
    # Normalized cost (0 to 1). Current implementation is binary cost.
    def _GMAuthorityAttrBsdOccCost(GMAuthorityAttr: GMAuthorityType)->float:
        #print("GMAuthorityAttr ",GMAuthorityAttr)
        if(GMAuthorityAttr == GMAuthorityType.CeSYS_e_GMAuthorityType_None):
            return 0
        return 1

    # Normalized cost (0 to 1). This method is a wrapper on the individual lane attribute cost calculations and arbitrates 
    # according to the (input) lane attribute, which lane attribute method to invoke
    # Input : LaneAttrIdx, pointer to the concerned lane attribute in RoutePlanLaneSegmentAttr enum
    # Input : LaneAttrVal, value of the pointed lane attribute 
    # Output: lane occupancy cost based on the concerned lane attribute
    @staticmethod
    def _LaneAttrBsdOccCost(LaneAttrIdx: int, LaneAttrVal: int)->float:  # if else logic
        if(LaneAttrIdx == RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus):
            return CostBasedRoutePlanner._MappingStatusAttrBsdOccCost(LaneAttrVal)
        elif(LaneAttrIdx == RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA):
            return CostBasedRoutePlanner._GMAuthorityAttrBsdOccCost(LaneAttrVal)
        elif(LaneAttrIdx == RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction):
            return CostBasedRoutePlanner._ConstructionZoneAttrBsdOccCost(LaneAttrVal)
        elif(LaneAttrIdx == RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Direction):
            return CostBasedRoutePlanner._LaneRouteDirAttrBsdOccCost(LaneAttrVal)
        else:
            print("Error LaneAttrIdx not supported ", LaneAttrIdx)
            return 0


    # Calculates lane occupancy cost for a single lane segment
    # Input : SceneLaneSegmentBase for the concerned lane
    # Output: LaneOccupancyCost, cost to the AV if it occupies the lane. 
    @staticmethod
    def _lane_occupancy_cost_calc(lanesegBaseData: SceneLaneSegmentBase)->float:
        LaneOccCost = 0

        # Now iterate over all the active lane attributes for the lane segment
        for Idx in range(lanesegBaseData.e_Cnt_num_active_lane_attributes):
            # LaneAttrIdx gives the index lookup for lane attributes and confidences
            LaneAttrIdx = lanesegBaseData.a_i_active_lane_attribute_indices[Idx]
            LaneAttrVal = lanesegBaseData.a_cmp_lane_attributes[LaneAttrIdx]
            LaneAttrConf = lanesegBaseData.a_cmp_lane_attribute_confidences[LaneAttrIdx]
            if (LaneAttrConf < 0.7):  # change to a config param later
                continue
            LaneAttrOccCost = CostBasedRoutePlanner._LaneAttrBsdOccCost(LaneAttrIdx = LaneAttrIdx, LaneAttrVal = LaneAttrVal)
            #print("LaneAttrConf ",LaneAttrConf," LaneAttrIdx",LaneAttrIdx," LaneAttrVal ",LaneAttrVal," LaneAttrOccCost ",LaneAttrOccCost)
            # Add costs from all lane attributes
            LaneOccCost = LaneOccCost + LaneAttrOccCost

        # Normalize to the [0, 1] range
        LaneOccCost = max(min(LaneOccCost, 1), 0)
        return LaneOccCost


    # Calculates lane end cost for a single lane segment
    # Input : SceneLaneSegmentBase for the concerned lane
    # Input : as_route_plan_lane_segments is the array or routeplan lane segments (already evaluated, downstrem of the concerned lane).
    #         We mainly need the lane end cost from here.
    # Input : reverseroadsegidx, the index of the concerned roadsegment index to access the costs from the as_route_plan_lane_segments container
    # Input : NextRoadSegLanes, list of lane segment IDs in the next road segment in route
    # Output: LaneEndCost, cost to the AV if it reaches the lane end
    @staticmethod
    def _lane_end_cost_calc(lanesegBaseData: SceneLaneSegmentBase,reverseroadsegidx:int,NextRoadSegLanes,\
        as_route_plan_lane_segments:List[List[RoutePlanLaneSegment]])->float: 
        LaneEndCost = 0
        MinDwnStreamLaneOccCost = 1
        # Search iteratively for the next segment lanes that are downstream to the current lane and in the route.
        # At this point assign the end cost of current lane = Min occ costs (of all downstream lanes)
        # search through all downstream lanes to to current lane
        for Idx in range(lanesegBaseData.e_Cnt_downstream_lane_count):
            DownStreamlanesegID = lanesegBaseData.as_downstream_lanes[Idx].e_i_lane_segment_id
            # All lane IDs in downstream roadsegment in route to the
            # currently indexed roadsegment in the loop
            if DownStreamlanesegID in NextRoadSegLanes:  # verify if the downstream lane is in the route
                DownStreamlanesegIdx = NextRoadSegLanes.index(DownStreamlanesegID)
                DownStreamLaneOccCost = as_route_plan_lane_segments[reverseroadsegidx-1][DownStreamlanesegIdx].e_cst_lane_occupancy_cost
                MinDwnStreamLaneOccCost = min(MinDwnStreamLaneOccCost, DownStreamLaneOccCost)
            else:
                # Add exception later
                print(" DownStreamlanesegID ", DownStreamlanesegID, " not found in NextRoadSegLanes ", NextRoadSegLanes)        
                LaneEndCost = MinDwnStreamLaneOccCost
        return LaneEndCost


    # Calculates lane end and occupancy costs for all the lanes in the NAV plan 

    def plan(self, RouteData: RoutePlannerInputData) -> DataRoutePlan:  
        """Add comments"""
        #
        # NewRoutePlan = DataRoutePlan()
        valid = True
        num_road_segments = len(RouteData.route_roadsegments)
        a_i_road_segment_ids = []
        a_Cnt_num_lane_segments = []
        as_route_plan_lane_segments = []

        # iterate over all road segments in the route plan in the reverse sequence. Enumerate the iterable to get the index also
        # index -> reverseroadsegidx
        # key -> roadsegID
        # value -> lanesegIDs
        reverse_enumerated_route_lanesegments = enumerate(reversed(RouteData.route_lanesegments.items()))
        for reverseroadsegidx, (roadsegID, lanesegIDs) in reverse_enumerated_route_lanesegments:
            # roadsegidx = num_road_segments - reverseroadsegidx
            AllRouteLanesInThisRoadSeg = []

            # Now iterate over all the lane segments inside  the enumerate(road segment)
            # index -> lanesegidx
            # value -> lanesegID
            enumerated_lanesegIDs = enumerate(lanesegIDs)
            for lanesegidx, lanesegID in enumerated_lanesegIDs:
                # Access all the lane segment lite data from lane segment dict
                lanesegBaseData = RouteData.LaneSegmentDict[lanesegID]

                # -------------------------------------------
                # Calculate lane occupancy costs for a lane
                # -------------------------------------------
                LaneOccCost = CostBasedRoutePlanner._lane_occupancy_cost_calc(lanesegBaseData)
                # -------------------------------------------
                # Calculate lane end costs (from lane occupancy costs)
                # -------------------------------------------

                if (reverseroadsegidx == 0):  # the last road segment in the route; put all endcosts to 0
                    LaneEndCost = 0
                elif (LaneOccCost == 1):
                    # Can't occupy the lane, end cost must be MAX(=1)
                    LaneEndCost = 1
                elif (reverseroadsegidx > 0):
                    NextRoadSegLanes = RouteData.route_lanesegments[prev_roadsegID]
                    LaneEndCost = CostBasedRoutePlanner._lane_end_cost_calc(lanesegBaseData=lanesegBaseData,reverseroadsegidx =reverseroadsegidx,\
                        NextRoadSegLanes = NextRoadSegLanes, as_route_plan_lane_segments = as_route_plan_lane_segments)
                else:
                    print(" Bad value for reverseroadsegidx :",reverseroadsegidx)  # Add exception later

                # Construct RoutePlanLaneSegment for the lane and add to the RoutePlanLaneSegment container for this Road Segment
                CurrRoutePlanLaneSegment = RoutePlanLaneSegment(e_i_lane_segment_id=lanesegID,
                                                                e_cst_lane_occupancy_cost=LaneOccCost, e_cst_lane_end_cost=LaneEndCost)
                AllRouteLanesInThisRoadSeg.append(CurrRoutePlanLaneSegment)

            # At this point we have at least iterated once through the road segment loop once. prev_roadsegID is set to be used for
            # the next road segment loop
            prev_roadsegID = roadsegID

            # push back the road segment sepecific info , as the road seg iteration is reverse
            a_i_road_segment_ids.insert(0, roadsegID)
            a_Cnt_num_lane_segments.insert(0, lanesegidx+1)
            as_route_plan_lane_segments.insert(0, AllRouteLanesInThisRoadSeg)

        NewRoutePlan = DataRoutePlan(e_b_is_valid=valid, e_Cnt_num_road_segments=num_road_segments, a_i_road_segment_ids=np.array(a_i_road_segment_ids),
                                     a_Cnt_num_lane_segments=np.array(a_Cnt_num_lane_segments), as_route_plan_lane_segments=as_route_plan_lane_segments)

        return NewRoutePlan
