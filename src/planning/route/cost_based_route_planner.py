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
    def _mapping_status_attributeBsdOccCost(mapping_status_attribute: LaneMappingStatusType)->float:
        if((mapping_status_attribute == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap) or
           (mapping_status_attribute == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_MDMap)):
            return 0
        return 1

    @staticmethod
    # Normalized cost (0 to 1). Current implementation is binary cost.
    def _construction_zone_attributeBsdOccCost(construction_zone_attribute: LaneConstructionType)->float:
        #print("construction_zone_attribute ",construction_zone_attribute)
        if((construction_zone_attribute == LaneConstructionType.CeSYS_e_LaneConstructionType_Normal) or
           (construction_zone_attribute == LaneConstructionType.CeSYS_e_LaneConstructionType_Unknown)):
            return 0
        return 1

    @staticmethod
    # Normalized cost (0 to 1). Current implementation is binary cost.
    def _lane_dir_in_route_attributeBsdOccCost(lane_dir_in_route_attribute: MapLaneDirection)->float:
        #print("lane_dir_in_route_attribute ",lane_dir_in_route_attribute)
        if((lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle) or
           (lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_Left_Towards_HostVehicle) or
                (lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_Right_Towards_HostVehicle)):
            return 0
        return 1

    @staticmethod
    # Normalized cost (0 to 1). Current implementation is binary cost.
    def _gm_authority_attributeBsdOccCost(gm_authority_attribute: GMAuthorityType)->float:
        #print("gm_authority_attribute ",gm_authority_attribute)
        if(gm_authority_attribute == GMAuthorityType.CeSYS_e_GMAuthorityType_None):
            return 0
        return 1

    # Normalized cost (0 to 1). This method is a wrapper on the individual lane attribute cost calculations and arbitrates 
    # according to the (input) lane attribute, which lane attribute method to invoke
    # Input : lane_attribute_index, pointer to the concerned lane attribute in RoutePlanLaneSegmentAttr enum
    # Input : lane_attribute_value, value of the pointed lane attribute 
    # Output: lane occupancy cost based on the concerned lane attribute
    @staticmethod
    def _LaneAttrBsdOccCost(lane_attribute_index: int, lane_attribute_value: int)->float:  # if else logic
        if(lane_attribute_index == RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus):
            return CostBasedRoutePlanner._mapping_status_attributeBsdOccCost(lane_attribute_value)
        elif(lane_attribute_index == RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA):
            return CostBasedRoutePlanner._gm_authority_attributeBsdOccCost(lane_attribute_value)
        elif(lane_attribute_index == RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction):
            return CostBasedRoutePlanner._construction_zone_attributeBsdOccCost(lane_attribute_value)
        elif(lane_attribute_index == RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Direction):
            return CostBasedRoutePlanner._lane_dir_in_route_attributeBsdOccCost(lane_attribute_value)
        else:
            print("Error lane_attribute_index not supported ", lane_attribute_index)
            return 0


    # Calculates lane occupancy cost for a single lane segment
    # Input : SceneLaneSegmentBase for the concerned lane
    # Output: LaneOccupancyCost, cost to the AV if it occupies the lane. 
    @staticmethod
    def _lane_occupancy_cost_calc(laneseg_base_data: SceneLaneSegmentBase)->float:
        lane_occupancy_cost = 0

        # Now iterate over all the active lane attributes for the lane segment
        for Idx in range(laneseg_base_data.e_Cnt_num_active_lane_attributes):
            # lane_attribute_index gives the index lookup for lane attributes and confidences
            lane_attribute_index = laneseg_base_data.a_i_active_lane_attribute_indices[Idx]
            lane_attribute_value = laneseg_base_data.a_cmp_lane_attributes[lane_attribute_index]
            lane_attribute_confidence = laneseg_base_data.a_cmp_lane_attribute_confidences[lane_attribute_index]
            if (lane_attribute_confidence < 0.7):  # change to a config param later
                continue
            lane_attribute_based_occupancy_cost = \
                CostBasedRoutePlanner._LaneAttrBsdOccCost(lane_attribute_index = lane_attribute_index, lane_attribute_value = lane_attribute_value)
            #print("lane_attribute_confidence ",lane_attribute_confidence," lane_attribute_index",lane_attribute_index," lane_attribute_value ",lane_attribute_value," lane_attribute_based_occupancy_cost ",lane_attribute_based_occupancy_cost)
            # Add costs from all lane attributes
            lane_occupancy_cost = lane_occupancy_cost + lane_attribute_based_occupancy_cost

        # Normalize to the [0, 1] range
        lane_occupancy_cost = max(min(lane_occupancy_cost, 1), 0)
        return lane_occupancy_cost


    # Calculates lane end cost for a single lane segment
    # Input : SceneLaneSegmentBase for the concerned lane
    # Input : as_route_plan_lane_segments is the array or routeplan lane segments (already evaluated, downstrem of the concerned lane).
    #         We mainly need the lane end cost from here.
    # Input : reversed_roadseg_idx_in_route, the index of the concerned roadsegment index to access the costs from the as_route_plan_lane_segments container
    # Input : lanesegs_in_next_roadseg, list of lane segment IDs in the next road segment in route
    # Output: lane_end_cost, cost to the AV if it reaches the lane end
    @staticmethod
    def _lane_end_cost_calc(laneseg_base_data: SceneLaneSegmentBase,reversed_roadseg_idx_in_route:int,lanesegs_in_next_roadseg,\
        as_route_plan_lane_segments:List[List[RoutePlanLaneSegment]])->float: 
        lane_end_cost = 0
        min_down_stream_laneseg_occupancy_cost = 1
        # Search iteratively for the next segment lanes that are downstream to the current lane and in the route.
        # At this point assign the end cost of current lane = Min occ costs (of all downstream lanes)
        # search through all downstream lanes to to current lane
        for Idx in range(laneseg_base_data.e_Cnt_downstream_lane_count):
            down_stream_laneseg_id = laneseg_base_data.as_downstream_lanes[Idx].e_i_lane_segment_id
            # All lane IDs in downstream roadsegment in route to the
            # currently indexed roadsegment in the loop
            if down_stream_laneseg_id in lanesegs_in_next_roadseg:  # verify if the downstream lane is in the route
                down_stream_laneseg_idx = lanesegs_in_next_roadseg.index(down_stream_laneseg_id)
                down_stream_laneseg_occupancy_cost = \
                    as_route_plan_lane_segments[reversed_roadseg_idx_in_route-1][down_stream_laneseg_idx].e_cst_lane_occupancy_cost
                min_down_stream_laneseg_occupancy_cost = min(min_down_stream_laneseg_occupancy_cost, down_stream_laneseg_occupancy_cost)
            else:
                # Add exception later
                print(" down_stream_laneseg_id ", down_stream_laneseg_id, " not found in lanesegs_in_next_roadseg ", lanesegs_in_next_roadseg)        
                lane_end_cost = min_down_stream_laneseg_occupancy_cost
        return lane_end_cost


    # Calculates lane end and occupancy costs for all the lanes in the NAV plan 

    def plan(self, route_data: RoutePlannerInputData) -> DataRoutePlan:  
        """Add comments"""
        #
        # NewRoutePlan = DataRoutePlan()
        valid = True
        num_road_segments = len(route_data.route_roadsegments)
        a_i_road_segment_ids = []
        a_Cnt_num_lane_segments = []
        as_route_plan_lane_segments = []

        # iterate over all road segments in the route plan in the reverse sequence. Enumerate the iterable to get the index also
        # index -> reversed_roadseg_idx_in_route
        # key -> roadseg_id
        # value -> laneseg_ids
        reverse_enumerated_route_lanesegments = enumerate(reversed(route_data.route_lanesegments.items()))
        for reversed_roadseg_idx_in_route, (roadseg_id, laneseg_ids) in reverse_enumerated_route_lanesegments:
            # roadseg_idx = num_road_segments - reversed_roadseg_idx_in_route
            all_routesegs_in_this_roadseg = []

            # Now iterate over all the lane segments inside  the enumerate(road segment)
            # index -> laneseg_idx
            # value -> laneseg_id
            enumerated_laneseg_ids = enumerate(laneseg_ids)
            for laneseg_idx, laneseg_id in enumerated_laneseg_ids:
                # Access all the lane segment lite data from lane segment dict
                laneseg_base_data = route_data.LaneSegmentDict[laneseg_id]

                # -------------------------------------------
                # Calculate lane occupancy costs for a lane
                # -------------------------------------------
                lane_occupancy_cost = CostBasedRoutePlanner._lane_occupancy_cost_calc(laneseg_base_data)
                # -------------------------------------------
                # Calculate lane end costs (from lane occupancy costs)
                # -------------------------------------------

                if (reversed_roadseg_idx_in_route == 0):  # the last road segment in the route; put all endcosts to 0
                    lane_end_cost = 0
                elif (lane_occupancy_cost == 1):
                    # Can't occupy the lane, end cost must be MAX(=1)
                    lane_end_cost = 1
                elif (reversed_roadseg_idx_in_route > 0):
                    lanesegs_in_next_roadseg = route_data.route_lanesegments[prev_roadseg_id]
                    lane_end_cost = CostBasedRoutePlanner._lane_end_cost_calc(laneseg_base_data=laneseg_base_data,\
                    reversed_roadseg_idx_in_route =reversed_roadseg_idx_in_route,\
                        lanesegs_in_next_roadseg = lanesegs_in_next_roadseg, as_route_plan_lane_segments = as_route_plan_lane_segments)
                else:
                    print(" Bad value for reversed_roadseg_idx_in_route :",reversed_roadseg_idx_in_route)  # Add exception later

                # Construct RoutePlanLaneSegment for the lane and add to the RoutePlanLaneSegment container for this Road Segment
                current_routeseg = RoutePlanLaneSegment(e_i_lane_segment_id=laneseg_id,
                                                                e_cst_lane_occupancy_cost=lane_occupancy_cost, e_cst_lane_end_cost=lane_end_cost)
                all_routesegs_in_this_roadseg.append(current_routeseg)

            # At this point we have at least iterated once through the road segment loop once. prev_roadseg_id is set to be used for
            # the next road segment loop
            prev_roadseg_id = roadseg_id

            # push back the road segment sepecific info , as the road seg iteration is reverse
            a_i_road_segment_ids.insert(0, roadseg_id)
            a_Cnt_num_lane_segments.insert(0, laneseg_idx+1)
            as_route_plan_lane_segments.insert(0, all_routesegs_in_this_roadseg)

        NewRoutePlan = DataRoutePlan(e_b_is_valid=valid, e_Cnt_num_road_segments=num_road_segments, a_i_road_segment_ids=np.array(a_i_road_segment_ids),
                                     a_Cnt_num_lane_segments=np.array(a_Cnt_num_lane_segments), as_route_plan_lane_segments=as_route_plan_lane_segments)

        return NewRoutePlan
