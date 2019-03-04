import numpy as np
from numpy import ndarray
import sys
import pprint
import traceback

from logging import Logger
from typing import List, Dict

from common_data.interface.Rte_Types.python.sub_structures import TsSYSRoutePlanLaneSegment, TsSYSDataRoutePlan

from decision_making.src.exceptions import  RoadSegmentLaneSegmentMismatch, raises
from decision_making.src.global_constants import LANE_ATTRIBUTE_CONFIDENCE_THRESHOLD
from decision_making.src.messages.route_plan_message import RoutePlan, RoutePlanLaneSegment, DataRoutePlan, RoadSegRoutePlanLaneSegments, \
    RoadRoutePlanLaneSegments
from decision_making.src.messages.scene_static_enums import (
    RoutePlanLaneSegmentAttr,
    LaneMappingStatusType,
    MapLaneDirection,
    GMAuthorityType,
    LaneConstructionType)
from decision_making.src.messages.scene_static_message import SceneLaneSegmentBase
from decision_making.src.planning.route.route_planner import RoutePlanner, RoutePlannerInputData

class CostBasedRoutePlanner(RoutePlanner): # Should this be named binary cost based route planner ?
    """
    child class (of abstract class RoutePlanner), which contains implementation details of binary cost based route planner

    """

    @staticmethod
    def mapping_status_based_occupancy_cost(mapping_status_attribute: LaneMappingStatusType) -> float:
        """
        Cost of lane map type. Current implementation is binary cost.
        :param mapping_status_attribute: type of mapped 
        :return: normalized cost (0 to 1)
        """
        if((mapping_status_attribute == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap) or
           (mapping_status_attribute == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_MDMap)):
            return 0
        return 1

    @staticmethod
    def construction_zone_based_occupancy_cost(construction_zone_attribute: LaneConstructionType) -> float:
        """
        Cost of construction zone type. Current implementation is binary cost. 
        :param construction_zone_attribute: type of lane construction
        :return: Normalized cost (0 to 1)
        """
        if((construction_zone_attribute == LaneConstructionType.CeSYS_e_LaneConstructionType_Normal) or
           (construction_zone_attribute == LaneConstructionType.CeSYS_e_LaneConstructionType_Unknown)):
            return 0
        return 1

    @staticmethod
    def lane_dir_in_route_based_occupancy_cost(lane_dir_in_route_attribute: MapLaneDirection) -> float:
        """
        Cost of lane direction. Current implementation is binary cost. 
        :param lane_dir_in_route_attribute: map lane direction in respect to host
        :return: Normalized cost (0 to 1)
        """
        if((lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle) or
           (lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_Left_Towards_HostVehicle) or
                (lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_Right_Towards_HostVehicle)):
            return 0
        return 1

    @staticmethod
    def gm_authority_based_occupancy_cost(gm_authority_attribute: GMAuthorityType) -> float:
        """
        Cost of GM authorized driving area. Current implementation is binary cost.  
        :param gm_authority_attribute: type of GM authority
        :return: Normalized cost (0 to 1)
        """
        if(gm_authority_attribute == GMAuthorityType.CeSYS_e_GMAuthorityType_None):
            return 0
        return 1

    @staticmethod
    @raises(IndexError)
    def lane_attribute_based_occupancy_cost(lane_attribute_index: int, lane_attribute_value: int) -> float:  # if else logic
        """
        This method is a wrapper on the individual lane attribute cost calculations and arbitrates
        according to the (input) lane attribute, which lane attribute method to invoke
        :param lane_attribute_index: pointer to the concerned lane attribute in RoutePlanLaneSegmentAttr enum
        :param lane_attribute_value: value of the pointed lane attribute
        :return: Normalized lane occupancy cost based on the concerned lane attribute (0 to 1)
        """
        if(lane_attribute_index == RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus):
            return CostBasedRoutePlanner.mapping_status_based_occupancy_cost(lane_attribute_value)
        elif(lane_attribute_index == RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA):
            return CostBasedRoutePlanner.gm_authority_based_occupancy_cost(lane_attribute_value)
        elif(lane_attribute_index == RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction):
            return CostBasedRoutePlanner.construction_zone_based_occupancy_cost(lane_attribute_value)
        elif(lane_attribute_index == RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Direction):
            return CostBasedRoutePlanner.lane_dir_in_route_based_occupancy_cost(lane_attribute_value)
        else:
            raise IndexError("Cost Based Route Planner: lane_attribute_index not supported",lane_attribute_index)
            return 0

    @staticmethod
    @raises(IndexError)
    def lane_occupancy_cost_calc(lane_segment_base_data: SceneLaneSegmentBase) -> float:
        """
        Calculates lane occupancy cost for a single lane segment
        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        :return: LaneOccupancyCost, cost to the AV if it occupies the lane.
        """
        lane_occupancy_cost = 0

        # Now iterate over all the active lane attributes for the lane segment
        for lane_attribute_index in lane_segment_base_data.a_i_active_lane_attribute_indices:   
            # lane_attribute_index gives the index lookup for lane attributes and confidences
            if lane_attribute_index < len(lane_segment_base_data.a_cmp_lane_attributes):
                lane_attribute_value = lane_segment_base_data.a_cmp_lane_attributes[lane_attribute_index]
            else:
                raise IndexError("Cost Based Route Planner: lane_attribute_index doesnt have corresponding lane attribute value"
                                ,lane_attribute_index)

            if lane_attribute_index < len(lane_segment_base_data.a_cmp_lane_attribute_confidences):
                lane_attribute_confidence = lane_segment_base_data.a_cmp_lane_attribute_confidences[lane_attribute_index]
            else:
                raise IndexError("Cost Based Route Planner: lane_attribute_index doesnt have corresponding lane attribute confidence value"
                                ,lane_attribute_index)

            
            if (lane_attribute_confidence < LANE_ATTRIBUTE_CONFIDENCE_THRESHOLD):  # change to a config param later
                continue
            lane_attribute_occupancy_cost = \
                CostBasedRoutePlanner.lane_attribute_based_occupancy_cost(lane_attribute_index=lane_attribute_index, 
                                                                          lane_attribute_value=lane_attribute_value)
            # Add costs from all lane attributes
            lane_occupancy_cost = lane_occupancy_cost + lane_attribute_occupancy_cost

        # Normalize to the [0, 1] range
        lane_occupancy_cost = max(min(lane_occupancy_cost, 1), 0)
        return lane_occupancy_cost

    @staticmethod
    @raises(IndexError)
    def lane_end_cost_calc(lane_segment_base_data: SceneLaneSegmentBase,route_plan_lane_segments: RoadRoutePlanLaneSegments) -> (float, bool):
        """
        Calculates lane end cost for a single lane segment
        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        :param route_plan_lane_segments: route_plan_lane_segments is the array or routeplan lane segments 
        (already evaluated, downstrem of the concerned lane). We mainly need the lane occupancy cost from here.
        :return: 
        lane_end_cost, cost to the AV if it reaches the lane end
        at_least_one_downstream_lane_to_current_lane_found_in_downstream_road_segment_in_route, diagnostics info, whether at least
        one downstream lane segment (as described in the map) is in the downstream route road segement
        """
        min_downstream_lane_segment_occupancy_cost = 1

        # Search iteratively for the next segment lanes that are downstream to the current lane and in the route.
        # At this point assign the end cost of current lane = Min occ costs (of all downstream lanes)
        # search through all downstream lanes to to current lane
        at_least_one_downstream_lane_to_current_lane_found_in_downstream_road_segment_in_route = False

        all_route_lane_segments_in_downstream_road_segment:RoadSegRoutePlanLaneSegments = route_plan_lane_segments[-1]
        lanesegs_in_downstream_road_segment_in_route:int = [] # list of lane segment IDs in the next road segment in route
        for route_lane_segment in all_route_lane_segments_in_downstream_road_segment:
            lanesegs_in_downstream_road_segment_in_route.append(route_lane_segment.e_i_lane_segment_id)
        lanesegs_in_downstream_road_segment_in_route = np.array(lanesegs_in_downstream_road_segment_in_route)

        for downstream_lane_segment in lane_segment_base_data.as_downstream_lanes:
            downstream_lane_segment_id = downstream_lane_segment.e_i_lane_segment_id

            # All lane IDs in downstream roadsegment in route to the currently indexed roadsegment in the loop
            if downstream_lane_segment_id in lanesegs_in_downstream_road_segment_in_route:  # verify if the downstream lane is in the route (it may not be ex: fork/exit)
                at_least_one_downstream_lane_to_current_lane_found_in_downstream_road_segment_in_route = True
                # find the index corresponding to the lane seg ID in the road segment
                downstream_lane_segment_idx = np.where(lanesegs_in_downstream_road_segment_in_route==downstream_lane_segment_id)[0][0]

                if(downstream_lane_segment_idx <len(all_route_lane_segments_in_downstream_road_segment)):
                    downstream_routeseg = all_route_lane_segments_in_downstream_road_segment[downstream_lane_segment_idx]  # 0 th index is the last segment pushed into this struct
                                                                                                 # which is the downstream roadseg.
                else:
                    raise IndexError("Cost Based Route Planner: downstream_lane_segment_idx not present in route_plan_lane_segments")
                

                downstream_lane_segment_occupancy_cost = downstream_routeseg.e_cst_lane_occupancy_cost
                min_downstream_lane_segment_occupancy_cost = min(min_downstream_lane_segment_occupancy_cost, downstream_lane_segment_occupancy_cost)   
            else:
                # Downstream lane segment not in route. Do nothing.
                pass

        lane_end_cost = min_downstream_lane_segment_occupancy_cost
        return lane_end_cost,at_least_one_downstream_lane_to_current_lane_found_in_downstream_road_segment_in_route



    @staticmethod
    def lane_cost_calc(lane_segment_base_data: SceneLaneSegmentBase,route_plan_lane_segments: RoadRoutePlanLaneSegments,\
                       no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route:bool) -> (RoutePlanLaneSegment, bool):

        """
        Calculates lane end and occupancy cost for a single lane segment
        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        :param route_plan_lane_segments: route_plan_lane_segments is the array or routeplan lane segments 
        (already evaluated, downstrem of the concerned lane). We mainly need the lane occupancy cost from here.
        :return: 
        RoutePlanLaneSegment, combined end and occupancy cost info for the lane
        at_least_one_downstream_lane_to_current_lane_found_in_downstream_road_segment_in_route, diagnostics info, whether at least
        one downstream lane segment (as described in the map) is in the downstream route road segement
        """

        lane_segment_id = lane_segment_base_data.e_i_lane_segment_id

        # -------------------------------------------
        # Calculate lane occupancy costs for a lane
        # -------------------------------------------
        lane_occupancy_cost = CostBasedRoutePlanner.lane_occupancy_cost_calc(lane_segment_base_data)
        
        # -------------------------------------------
        # Calculate lane end costs (from lane occupancy costs)
        # -------------------------------------------                
        
        if not route_plan_lane_segments: # if route_plan_lane_segments is empty indicating the last segment in route
            if (lane_occupancy_cost == 1):# Can't occupy the lane, can't occupy the end either. end cost must be MAX(=1)
                lane_end_cost = 1
            else :
                lane_end_cost = 0
            no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route = False
            # Because this is the last road segment in (current) route  we don't want to trigger RoadSegmentLaneSegmentMismatch 
            # exception by running diagnostics on the downstream to the last road segment, in route.


        else:

            lane_end_cost_calc_from_downstream_segments, at_least_one_downstream_lane_to_current_lane_found_in_downstream_road_segment_in_route = \
                CostBasedRoutePlanner.lane_end_cost_calc(lane_segment_base_data=lane_segment_base_data,
                                                         route_plan_lane_segments=route_plan_lane_segments)
            no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route = \
                no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route and \
                not(at_least_one_downstream_lane_to_current_lane_found_in_downstream_road_segment_in_route)

            if (lane_occupancy_cost == 1):# Can't occupy the lane, can't occupy the end either. end cost must be MAX(=1)
                lane_end_cost = 1 
            else:   
                lane_end_cost = lane_end_cost_calc_from_downstream_segments

        # Construct RoutePlanLaneSegment for the lane and add to the RoutePlanLaneSegment container for this Road Segment
        current_route_lane_segment = RoutePlanLaneSegment(e_i_lane_segment_id=lane_segment_id,
                                                          e_cst_lane_occupancy_cost=lane_occupancy_cost, 
                                                          e_cst_lane_end_cost=lane_end_cost)
        return (current_route_lane_segment,no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route)




    @raises(IndexError)
    @raises(RoadSegmentLaneSegmentMismatch)
    @raises(KeyError)
    def plan(self, route_data: RoutePlannerInputData) -> DataRoutePlan:
        """
        Calculates lane end and occupancy costs for all the lanes in the NAV plan
        :input:  RoutePlannerInputData, pre-processed data for RoutePlan cost calcualtions. More details at 
                 RoutePlannerInputData() class definition.
        :return: DataRoutePlan , the complete route plan information ready to be serialized and published
        """
        
        valid = True
        num_road_segments = len(route_data.route_lane_segments)
        road_segment_ids:List[int] = []
        num_lane_segments:List[int] = []
        route_plan_lane_segments:RoadRoutePlanLaneSegments = []


        # iterate over all road segments in the route plan in the reverse sequence. Enumerate the iterable to get the index also
        # index -> reversed_road_segment_idx_in_route
        # key -> road_segment_id
        # value -> lane_segment_ids
        for  (road_segment_id, lane_segment_ids) in reversed(route_data.route_lane_segments.items()):
            all_route_lane_segments_in_this_road_segment:RoadSegRoutePlanLaneSegments = []
            no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route = True # as the name suggests
            # if there is NO downstream lane (as defined in map) to the current road segment (any of its lanes) that is in the route

            # Now iterate over all the lane segments inside  the lane_segment_ids (ndarray)
            # value -> lane_segment_id
            for lane_segment_id in lane_segment_ids:

                if lane_segment_id in route_data.route_lane_segments_base_as_dict:
                    # Access all the lane segment lite data from lane segment dict
                    current_lane_segment_base_data = route_data.route_lane_segments_base_as_dict[lane_segment_id]
                else:
                    raise KeyError("Cost Based Route Planner: Lane segment not found in route_lane_segments_base_as_dict. Not \
                                    found lane_segment_id = ",lane_segment_id)
                
                current_route_lane_segment, no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route = \
                    CostBasedRoutePlanner.lane_cost_calc(lane_segment_base_data = current_lane_segment_base_data,
                                                         route_plan_lane_segments = route_plan_lane_segments,
                                   no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route = \
                                    no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route)

                all_route_lane_segments_in_this_road_segment.append(current_route_lane_segment)

            if(no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route):
                raise RoadSegmentLaneSegmentMismatch("Cost Based Route Planner: Not a single downstream lane segment to the current \
                    road segment (lane segments) were found in the downstream road segment described in the navigation route plan",\
                    "road_segment_id",road_segment_id)



            # append the road segment sepecific info , as the road seg iteration is reverse
            road_segment_ids.append(road_segment_id)
            num_lane_segments.append(len(lane_segment_ids))
            route_plan_lane_segments.append(all_route_lane_segments_in_this_road_segment)
        
        # Two step append (O(n)) and reverse (O(n)) is less costly than one step insert (o(n^2)) at the beginning of the list
        # at each road segment loop (of length n)
        road_segment_ids.reverse()
        num_lane_segments.reverse()
        route_plan_lane_segments.reverse()

        return DataRoutePlan(e_b_is_valid=valid, 
                             e_Cnt_num_road_segments=num_road_segments, 
                             a_i_road_segment_ids=np.array(road_segment_ids),
                             a_Cnt_num_lane_segments=np.array(num_lane_segments), 
                             as_route_plan_lane_segments=route_plan_lane_segments)
