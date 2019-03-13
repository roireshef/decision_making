import numpy as np

from typing import (
    List, 
    Dict)



from common_data.interface.Rte_Types.python.sub_structures import (
    TsSYSRoutePlanLaneSegment, 
    TsSYSDataRoutePlan)

from decision_making.src.exceptions import  (
    RoadSegmentLaneSegmentMismatch, 
    raises)

from decision_making.src.global_constants import (
    LANE_ATTRIBUTE_CONFIDENCE_THRESHOLD, 
    TRUE_COST, 
    FALSE_COST  )

from decision_making.src.messages.route_plan_message import (
    RoutePlan, 
    RoutePlanLaneSegment, 
    DataRoutePlan, 
    RoadSegRoutePlanLaneSegments, 
    RoadRoutePlanLaneSegments)

from decision_making.src.messages.scene_static_enums import (
    RoutePlanLaneSegmentAttr,
    LaneMappingStatusType,
    MapLaneDirection,
    GMAuthorityType,
    LaneConstructionType)

from decision_making.src.messages.scene_static_message import (
    SceneLaneSegmentBase)

from decision_making.src.planning.route.route_planner import (
    RoutePlanner, 
    RoutePlannerInputData)

class BinaryCostBasedRoutePlanner(RoutePlanner): 
    """
    child class (of abstract class RoutePlanner), which contains implementation details of binary cost based route planner
    """
    def __init__(self):
        self.__route_plan_lane_segments:RoadRoutePlanLaneSegments = []

    def get_route_plan_lane_segments(self) -> RoadRoutePlanLaneSegments:
        return self.__route_plan_lane_segments


    @staticmethod
    # Following method is kept public in order to unit test the method from outside the class
    def mapping_status_based_occupancy_cost(mapping_status_attribute: LaneMappingStatusType) -> float:
        """
        Cost of lane map type. Current implementation is binary cost.
        :param mapping_status_attribute: type of mapped 
        :return: normalized cost (FALSE_COST to TRUE_COST)
        """
        if ( (mapping_status_attribute == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap) or
             (mapping_status_attribute == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_MDMap) ):
            return FALSE_COST
        return TRUE_COST

    @staticmethod
    # Following method is kept public in order to unit test the method from outside the class
    def construction_zone_based_occupancy_cost(construction_zone_attribute: LaneConstructionType) -> float:
        """
        Cost of construction zone type. Current implementation is binary cost. 
        :param construction_zone_attribute: type of lane construction
        :return: Normalized cost (FALSE_COST to TRUE_COST)
        """
        if ( (construction_zone_attribute == LaneConstructionType.CeSYS_e_LaneConstructionType_Normal) or
             (construction_zone_attribute == LaneConstructionType.CeSYS_e_LaneConstructionType_Unknown) ):
            return FALSE_COST
        return TRUE_COST

    @staticmethod
    # Following method is kept public in order to unit test the method from outside the class
    def lane_dir_in_route_based_occupancy_cost(lane_dir_in_route_attribute: MapLaneDirection) -> float:
        """
        Cost of lane direction. Current implementation is binary cost. 
        :param lane_dir_in_route_attribute: map lane direction in respect to host
        :return: Normalized cost (FALSE_COST to TRUE_COST)
        """
        if ( (lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle) or
             (lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_Left_Towards_HostVehicle) or
             (lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_Right_Towards_HostVehicle) ):
            return FALSE_COST
        return TRUE_COST

    @staticmethod
    # Following method is kept public in order to unit test the method from outside the class
    def gm_authority_based_occupancy_cost(gm_authority_attribute: GMAuthorityType) -> float:
        """
        Cost of GM authorized driving area. Current implementation is binary cost.  
        :param gm_authority_attribute: type of GM authority
        :return: Normalized cost (FALSE_COST to TRUE_COST)
        """
        if (gm_authority_attribute == GMAuthorityType.CeSYS_e_GMAuthorityType_None):
            return FALSE_COST
        return TRUE_COST


    @staticmethod
    @raises(IndexError)
    # Following method is kept public in order to unit test the method from outside the class
    def lane_attribute_based_occupancy_cost(lane_attribute_index: int, lane_attribute_value: int) -> float:  # if else logic
        """
        This method is a wrapper on the individual lane attribute cost calculations and arbitrates
        according to the (input) lane attribute, which lane attribute method to invoke
        :param lane_attribute_index: pointer to the concerned lane attribute in RoutePlanLaneSegmentAttr enum
        :param lane_attribute_value: value of the pointed lane attribute
        :return: Normalized lane occupancy cost based on the concerned lane attribute (FALSE_COST to TRUE_COST)
        """
        if 'attribute_based_occupancy_cost_methods' not in BinaryCostBasedRoutePlanner.lane_attribute_based_occupancy_cost.__dict__:
            # The above if check and then setting of attribute_based_occupancy_cost_methods within the if block is equivalent of 
            # making attribute_based_occupancy_cost_methods a static dictionary (of [lane_attribute_index, lane attribute based occupancy_cost 
            # calculation methods])
            attribute_based_occupancy_cost_methods = {
            RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus:BinaryCostBasedRoutePlanner.mapping_status_based_occupancy_cost,
            
            RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA:BinaryCostBasedRoutePlanner.gm_authority_based_occupancy_cost,

            RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction:BinaryCostBasedRoutePlanner.construction_zone_based_occupancy_cost,

            RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Direction: BinaryCostBasedRoutePlanner.lane_dir_in_route_based_occupancy_cost
            }
        
        if lane_attribute_index in attribute_based_occupancy_cost_methods:
            # Following is equivalent of (pythonic way) executing a switch statement 
            occupancy_cost_method = attribute_based_occupancy_cost_methods[lane_attribute_index]                                                                                        # 
            return occupancy_cost_method(lane_attribute_value)
        else:
            raise IndexError("Cost Based Route Planner: lane_attribute_index not supported", lane_attribute_index)
            return FALSE_COST

        

    @staticmethod
    @raises(IndexError)
    # Following method is kept public in order to unit test the method from outside the class
    def lane_occupancy_cost_calc(lane_segment_base_data: SceneLaneSegmentBase) -> float:
        """
        Calculates lane occupancy cost for a single lane segment
        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        :return: LaneOccupancyCost, cost to the AV if it occupies the lane.
        """

        # Now iterate over all the active lane attributes for the lane segment
        for lane_attribute_index in lane_segment_base_data.a_i_active_lane_attribute_indices:   
            # lane_attribute_index gives the index lookup for lane attributes and confidences
            if lane_attribute_index < len(lane_segment_base_data.a_cmp_lane_attributes):
                lane_attribute_value = lane_segment_base_data.a_cmp_lane_attributes[lane_attribute_index]
            else:
                raise IndexError("Cost Based Route Planner: lane_attribute_index doesnt have corresponding lane attribute value"
                                 , lane_attribute_index)

            if lane_attribute_index < len(lane_segment_base_data.a_cmp_lane_attribute_confidences):
                lane_attribute_confidence = lane_segment_base_data.a_cmp_lane_attribute_confidences[lane_attribute_index]
            else:
                raise IndexError("Cost Based Route Planner: lane_attribute_index doesnt have corresponding lane attribute confidence value"
                                 , lane_attribute_index)

            
            if (lane_attribute_confidence < LANE_ATTRIBUTE_CONFIDENCE_THRESHOLD):
                continue
            
            lane_attribute_occupancy_cost = BinaryCostBasedRoutePlanner.lane_attribute_based_occupancy_cost(lane_attribute_index=lane_attribute_index, 
                                                                                                            lane_attribute_value=lane_attribute_value )
            # Check if the lane is unoccupiable 
            if(lane_attribute_occupancy_cost==TRUE_COST):
                return TRUE_COST

        return FALSE_COST


    @raises(IndexError)
    # Following method is kept public in order to unit test the method from outside the class
    def lane_end_cost_calc(self, lane_segment_base_data: SceneLaneSegmentBase) -> (float, bool, List[int]):
        """
        Calculates lane end cost for a single lane segment
        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        (already evaluated, downstrem of the concerned lane). We mainly need the lane occupancy cost from here.
        :return: 
            (float): lane_end_cost, cost to the AV if it reaches the lane end
            (bool):  lane conectivity diagnostics info, whether at least one downstream lane segment (as described in the map) 
                     is in the downstream route road segement
            (List[int]): list of all downstream lane segment ids 
        """
        min_downstream_lane_segment_occupancy_cost = TRUE_COST

        # Search iteratively for the next segment lanes that are downstream to the current lane and in the route.
        # At this point assign the end cost of current lane = Min occ costs (of all downstream lanes)
        # search through all downstream lanes to to current lane
        downstream_lane_found_in_route = False

        downstream_route_lane_segments: RoadSegRoutePlanLaneSegments = self.get_route_plan_lane_segments()[-1]

        downstream_route_lane_segment_ids = np.array([route_lane_segment.e_i_lane_segment_id for route_lane_segment in downstream_route_lane_segments ])

        for downstream_base_lane_segment in lane_segment_base_data.as_downstream_lanes:

            downstream_lane_segment_id = downstream_base_lane_segment.e_i_lane_segment_id

            # All lane IDs in downstream roadsegment in route to the currently indexed roadsegment in the loop
            if downstream_lane_segment_id in downstream_route_lane_segment_ids:  # verify if the downstream lane is in the route (it may not be ex: fork/exit)
                
                downstream_lane_found_in_route = True

                # find the index corresponding to the lane seg ID in the road segment
                downstream_route_lane_segment_idx = np.where(downstream_route_lane_segment_ids == downstream_lane_segment_id)[0][0]

                if ( downstream_route_lane_segment_idx < len(downstream_route_lane_segments) ):
                    downstream_route_lane_segment = downstream_route_lane_segments[downstream_route_lane_segment_idx]
                else:
                    raise IndexError("Cost Based Route Planner: downstream_route_lane_segment_idx not present in route_plan_lane_segments")
                
                downstream_lane_segment_occupancy_cost = downstream_route_lane_segment.e_cst_lane_occupancy_cost

                min_downstream_lane_segment_occupancy_cost = min(min_downstream_lane_segment_occupancy_cost, downstream_lane_segment_occupancy_cost)   
            else:
                # Downstream lane segment not in route. Do nothing.
                pass

        lane_end_cost = min_downstream_lane_segment_occupancy_cost
        return lane_end_cost, downstream_lane_found_in_route, downstream_route_lane_segment_ids

    # Following method is kept public in order to unit test the method from outside the class
    def lane_cost_calc(self, lane_segment_base_data: SceneLaneSegmentBase) -> (RoutePlanLaneSegment, bool, List[int]):
        """
        Calculates lane end and occupancy cost for a single lane segment
        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        :return: 
            RoutePlanLaneSegment: combined end and occupancy cost info for the lane
            (bool): True if any downstream lane segment is found
            (List[int]): list of all downstream lane segment ids
        """
        lane_segment_id = lane_segment_base_data.e_i_lane_segment_id
        downstream_lane_segment_ids:List[int] = []

        # Calculate lane occupancy costs for a lane
        lane_occupancy_cost = BinaryCostBasedRoutePlanner.lane_occupancy_cost_calc(lane_segment_base_data)
        
        # Calculate lane end costs (from lane occupancy costs)
        if  not self.get_route_plan_lane_segments(): # if route_plan_lane_segments is empty indicating the last segment in route
            if (lane_occupancy_cost == TRUE_COST):  # Can't occupy the lane, can't occupy the end either. end cost must be MAX(=TRUE_COST)
                lane_end_cost = TRUE_COST
            else :
                lane_end_cost = FALSE_COST

            downstream_lane_found_in_route = True
            # Because this is the last road segment in (current) route  we don't want to trigger RoadSegmentLaneSegmentMismatch 
            # exception by running diagnostics on the downstream to the last road segment, in route.
        else:

            lane_end_cost, downstream_lane_found_in_route, downstream_lane_segment_ids = (self.lane_end_cost_calc
                                                                                               (lane_segment_base_data=lane_segment_base_data))
                                                                                        
            
            if (lane_occupancy_cost == TRUE_COST):# Can't occupy the lane, can't occupy the end either. end cost must be MAX(=TRUE_COST)
                lane_end_cost = TRUE_COST 

        # Construct RoutePlanLaneSegment for the lane and add to the RoutePlanLaneSegment container for this Road Segment
        current_route_lane_segment = RoutePlanLaneSegment(e_i_lane_segment_id=lane_segment_id,
                                                          e_cst_lane_occupancy_cost=lane_occupancy_cost, 
                                                          e_cst_lane_end_cost=lane_end_cost)
        
        return current_route_lane_segment, downstream_lane_found_in_route, downstream_lane_segment_ids


    @raises(RoadSegmentLaneSegmentMismatch)
    def road_segment_cost_calc(self, road_segment_id:int) -> RoadSegRoutePlanLaneSegments:
        """
        Itreratively uses lane_cost_calc method to calculate lane costs (occupancy and end) for all lane segments in a road segment
        :return: 
        RoadSegRoutePlanLaneSegments, which is List[RoutePlanLaneSegments]
        Also raises RoadSegmentLaneSegmentMismatch internally if it can't find any downstream lane segment in the route 
        """
        route_lane_segments:RoadSegRoutePlanLaneSegments = []

        downstream_road_segment_not_found = True # as the name suggests
                                                 # if there is NO downstream lane (as defined in map) to the current 
                                                 # road segment (any of its lanes) that is in the route
        
        if not self.get_route_plan_lane_segments(): # check if this is the last road segment in the nav plan
           downstream_road_segment_not_found = False

        downstream_lane_segment_ids_for_road_segment:List[List[int]] = []
        downstream_lane_segment_ids:List[int] = []

        route_data:RoutePlannerInputData = RoutePlannerInputData().get_instance()

        lane_segment_ids = route_data.get_lane_segment_ids_for_road_segment(road_segment_id)

        # Now iterate over all the lane segments inside  the lane_segment_ids (ndarray)
        # value -> lane_segment_id
        for lane_segment_id in lane_segment_ids:

            lane_segment_base_data = route_data.get_lane_segment_base(lane_segment_id)
            
            route_lane_segment, downstream_lane_found_in_route , downstream_lane_segment_ids = (self.lane_cost_calc
                                                                                                    (lane_segment_base_data=lane_segment_base_data))
                                                                        

            downstream_road_segment_not_found = downstream_road_segment_not_found and not(downstream_lane_found_in_route)

            downstream_lane_segment_ids_for_road_segment.append(downstream_lane_segment_ids)

            route_lane_segments.append(route_lane_segment)

    
        if (downstream_road_segment_not_found):
            raise RoadSegmentLaneSegmentMismatch("Cost Based Route Planner: Not a single downstream lane segment to the current" +
                                                 "road segment (lane segments) were found in the downstream road segment described" +
                                                 "in the navigation route plan road_segment_id ", road_segment_id, " lane_segment_ids", 
                                                 lane_segment_ids, " next(road_segment_id)", 
                                                 route_data.get_next_road_segment(road_segment_id), " route_plan", route_data.get_nav_plan(),
                                                 "downstream_lane_segment_ids_for_road_segment ", downstream_lane_segment_ids_for_road_segment)
        
        return route_lane_segments


    @raises(IndexError, KeyError)
    def plan(self ) -> DataRoutePlan:
        """
        Calculates lane end and occupancy costs for all the lanes in the NAV plan
        :input:  RoutePlannerInputData, pre-processed data for RoutePlan cost calcualtions. More details at 
                 RoutePlannerInputData() class definition.
        :return: DataRoutePlan , the complete route plan information ready to be serialized and published
        """  
        road_segment_ids: List[int] = []
        num_lane_segments: List[int] = []
        route_data: RoutePlannerInputData = RoutePlannerInputData().get_instance()

        # iterate over all road segments in the route plan in the reverse sequence. Enumerate the iterable to get the index also
        # key -> road_segment_id
        # value -> lane_segment_ids

        self.__route_plan_lane_segments:RoadRoutePlanLaneSegments = []

        for (road_segment_id, lane_segment_ids) in reversed(route_data.get_lane_segment_ids_for_route().items()):
            
            route_lane_segments = self.road_segment_cost_calc(road_segment_id=road_segment_id)
            
            # append the road segment sepecific info , as the road seg iteration is reverse
            road_segment_ids.append(road_segment_id)
            
            num_lane_segments.append( len(lane_segment_ids) )

            self.__route_plan_lane_segments.append(route_lane_segments)
        
        # Two step append (O(n)) and reverse (O(n)) is less costly than one step insert (o(n^2)) at the beginning of the list
        # at each road segment loop (of length n)
        road_segment_ids.reverse()
        num_lane_segments.reverse()
        self.__route_plan_lane_segments.reverse()

        num_road_segments = len(road_segment_ids)

        valid = True

        return DataRoutePlan(e_b_is_valid=valid, 
                             e_Cnt_num_road_segments=num_road_segments, 
                             a_i_road_segment_ids=np.array(road_segment_ids),
                             a_Cnt_num_lane_segments=np.array(num_lane_segments), 
                             as_route_plan_lane_segments=self.get_route_plan_lane_segments())