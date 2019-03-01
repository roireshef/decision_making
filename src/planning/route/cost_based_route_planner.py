import numpy as np
from numpy import ndarray
import sys
import pprint
import traceback

from logging import Logger
from typing import List, Dict

from common_data.interface.Rte_Types.python.sub_structures import TsSYSRoutePlanLaneSegment, TsSYSDataRoutePlan

from decision_making.src.exceptions import UnwantedNegativeIndex, RoadSegmentLaneSegmentMismatch, raises
from decision_making.src.global_constants import LANE_ATTRIBUTE_CONFIDENCE_THRESHOLD
from decision_making.src.messages.route_plan_message import RoutePlan, RoutePlanLaneSegment, DataRoutePlan
from decision_making.src.messages.scene_static_enums import (
    RoutePlanLaneSegmentAttr,
    LaneMappingStatusType,
    MapLaneDirection,
    GMAuthorityType,
    LaneConstructionType)
from decision_making.src.messages.scene_static_message import SceneLaneSegmentBase
from decision_making.src.planning.route.route_planner import RoutePlanner, RoutePlannerInputData

class CostBasedRoutePlanner(RoutePlanner): # Should this be named binary cost based route planner ?
    """TODO Add comments"""

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
            print("Error lane_attribute_index not supported ",
                  lane_attribute_index)
            return 0

    @staticmethod
    def lane_occupancy_cost_calc(laneseg_base_data: SceneLaneSegmentBase) -> float:
        """
        Calculates lane occupancy cost for a single lane segment
        :param laneseg_base_data: SceneLaneSegmentBase for the concerned lane
        :return: LaneOccupancyCost, cost to the AV if it occupies the lane.
        """
        lane_occupancy_cost = 0

        # Now iterate over all the active lane attributes for the lane segment
        for lane_attribute_index in laneseg_base_data.a_i_active_lane_attribute_indices:   
            # lane_attribute_index gives the index lookup for lane attributes and confidences
            lane_attribute_value = laneseg_base_data.a_cmp_lane_attributes[lane_attribute_index]
            lane_attribute_confidence = laneseg_base_data.a_cmp_lane_attribute_confidences[lane_attribute_index]
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
    def lane_end_cost_calc(laneseg_base_data: SceneLaneSegmentBase, lanesegs_in_downstream_roadseg_in_route: ndarray,
                           route_plan_lane_segments: List[List[RoutePlanLaneSegment]]) -> (float, bool):
        """
        Calculates lane end cost for a single lane segment
        :param laneseg_base_data: SceneLaneSegmentBase for the concerned lane
        :param lanesegs_in_downstream_roadseg_in_route: list of lane segment IDs in the next road segment in route
        :param route_plan_lane_segments: route_plan_lane_segments is the array or routeplan lane segments 
        (already evaluated, downstrem of the concerned lane). We mainly need the lane end cost from here.
        :return: 
        lane_end_cost, cost to the AV if it reaches the lane end
        at_least_one_downstream_lane_to_current_lane_found_in_downstream_road_segment_in_route-> diagnostics info, whether at least
        one downstream lane segment (as described in the map) is in the downstream route road segement
        """
        min_down_stream_laneseg_occupancy_cost = 1

        # Search iteratively for the next segment lanes that are downstream to the current lane and in the route.
        # At this point assign the end cost of current lane = Min occ costs (of all downstream lanes)
        # search through all downstream lanes to to current lane
        laneseg_id = laneseg_base_data.e_i_lane_segment_id
        downstream_lane_segment_ids = []
        at_least_one_downstream_lane_to_current_lane_found_in_downstream_road_segment_in_route = False
        for down_stream_laneseg in laneseg_base_data.as_downstream_lanes:
            down_stream_laneseg_id = down_stream_laneseg.e_i_lane_segment_id
            downstream_lane_segment_ids.append(down_stream_laneseg_id)

            # All lane IDs in downstream roadsegment in route to the currently indexed roadsegment in the loop
            if down_stream_laneseg_id in lanesegs_in_downstream_roadseg_in_route:  # verify if the downstream lane is in the route (it may not be ex: fork/exit)
                at_least_one_downstream_lane_to_current_lane_found_in_downstream_road_segment_in_route = True
                # find the index corresponding to the lane seg ID in the road segment
                down_stream_laneseg_idx = np.where(lanesegs_in_downstream_roadseg_in_route==down_stream_laneseg_id)[0][0]
                down_stream_routeseg = route_plan_lane_segments[0][down_stream_laneseg_idx]  # 0 th index is the last segment pushed into this struct
                                                                                                # which is the downstream roadseg.
                # TODO if the route_plan_lane_segments struct is reversed access -1 index instead of 0
                down_stream_laneseg_occupancy_cost = down_stream_routeseg.e_cst_lane_occupancy_cost
                min_down_stream_laneseg_occupancy_cost = min(min_down_stream_laneseg_occupancy_cost, down_stream_laneseg_occupancy_cost)   
            else:
                # Downstream lane segment not in route. Do nothing.
                pass

        lane_end_cost = min_down_stream_laneseg_occupancy_cost
        return lane_end_cost,at_least_one_downstream_lane_to_current_lane_found_in_downstream_road_segment_in_route

    
    @raises(UnwantedNegativeIndex)
    @raises(RoadSegmentLaneSegmentMismatch)
    def plan(self, route_data: RoutePlannerInputData) -> DataRoutePlan:
        """
        Calculates lane end and occupancy costs for all the lanes in the NAV plan
        :param X: TODO
        :return: TODO
        """
        valid = True
        num_road_segments = len(route_data.route_lanesegments)
        a_i_road_segment_ids = []
        a_Cnt_num_lane_segments = []
        route_plan_lane_segments = []
        # TODO add types of above variables

        no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route = True

        # iterate over all road segments in the route plan in the reverse sequence. Enumerate the iterable to get the index also
        # index -> reversed_roadseg_idx_in_route
        # key -> roadseg_id
        # value -> laneseg_ids
        reverse_enumerated_route_lanesegments = enumerate( reversed(route_data.route_lanesegments.items()))
        # TODO check if the reverse operation and enumerate step is done every time in for loop
        for reversed_roadseg_idx_in_route, (roadseg_id, laneseg_ids) in reverse_enumerated_route_lanesegments:
            all_route_lanesegs_in_this_roadseg = []

            # Now iterate over all the lane segments inside  the enumerate(road segment)
            # index -> laneseg_idx
            # value -> laneseg_id
            enumerated_laneseg_ids = enumerate(laneseg_ids)
            # TODO
            for laneseg_idx, laneseg_id in enumerated_laneseg_ids:
                # Access all the lane segment lite data from lane segment dict
                current_laneseg_base_data = route_data.route_lanesegs_base_as_dict[laneseg_id]

                # -------------------------------------------
                # Calculate lane occupancy costs for a lane
                # -------------------------------------------
                lane_occupancy_cost = CostBasedRoutePlanner.lane_occupancy_cost_calc(current_laneseg_base_data)
                
                # -------------------------------------------
                # Calculate lane end costs (from lane occupancy costs)
                # -------------------------------------------                

                if (reversed_roadseg_idx_in_route == 0): # the last road segment in the route; put all endcosts to 0
                    if (lane_occupancy_cost == 1):# Can't occupy the lane, can't occupy the end either. end cost must be MAX(=1)
                        lane_end_cost = 1
                    else :
                        lane_end_cost = 0
                    no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route = False
                    # Because this is the last road segment in (current) route  we don't want to trigger RoadSegmentLaneSegmentMismatch 
                    # exception by running diagnostics on the downstream to the last road segment, in route.


                elif (reversed_roadseg_idx_in_route > 0):
                    lanesegs_in_downstream_roadseg_in_route = route_data.route_lanesegments[downstream_road_segment_id]
                    lane_end_cost_calc_from_downstream_segments, at_least_one_downstream_lane_to_current_lane_found_in_downstream_road_segment_in_route = \
                        CostBasedRoutePlanner.lane_end_cost_calc(laneseg_base_data=current_laneseg_base_data,
                                                                 lanesegs_in_downstream_roadseg_in_route=lanesegs_in_downstream_roadseg_in_route,
                                                                 route_plan_lane_segments=route_plan_lane_segments)
                    no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route = \
                        no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route and \
                        not(at_least_one_downstream_lane_to_current_lane_found_in_downstream_road_segment_in_route)

                    if (lane_occupancy_cost == 1):# Can't occupy the lane, can't occupy the end either. end cost must be MAX(=1)
                        lane_end_cost = 1 
                    else:   
                        lane_end_cost = lane_end_cost_calc_from_downstream_segments
                else:
                    raise UnwantedNegativeIndex("Cost Based Route Planner: Negative value for reversed_roadseg_idx_in_route")

                # Construct RoutePlanLaneSegment for the lane and add to the RoutePlanLaneSegment container for this Road Segment
                current_route_laneseg = RoutePlanLaneSegment(e_i_lane_segment_id=laneseg_id,
                                                        e_cst_lane_occupancy_cost=lane_occupancy_cost, 
                                                        e_cst_lane_end_cost=lane_end_cost)
                all_route_lanesegs_in_this_roadseg.append(current_route_laneseg)

            if(no_downstream_lane_to_current_road_segment_found_in_downstream_road_segment_in_route):
                raise RoadSegmentLaneSegmentMismatch("Cost Based Route Planner: Not a single downstream lane segment to the current \
                    road segment (lane segments) were found in the downstream road segment described in the navigation route plan")
                

            # At this point we have at least iterated once through the road segment loop once. downstream_road_segment_id is set to be used for
            # the next road segment loop
            downstream_road_segment_id = roadseg_id           
            
            # push back the road segment sepecific info , as the road seg iteration is reverse
            a_i_road_segment_ids.insert(0, roadseg_id)
            a_Cnt_num_lane_segments.insert(0, laneseg_idx+1)
            route_plan_lane_segments.insert(0, all_route_lanesegs_in_this_roadseg)
            # TODO append and then reverse 

        return DataRoutePlan(e_b_is_valid=valid, 
                             e_Cnt_num_road_segments=num_road_segments, 
                             a_i_road_segment_ids=np.array(a_i_road_segment_ids),
                             a_Cnt_num_lane_segments=np.array(a_Cnt_num_lane_segments), 
                             as_route_plan_lane_segments=route_plan_lane_segments)
