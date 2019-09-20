import numpy as np
from abc import abstractmethod
from typing import List
import rte.python.profiler as prof
from decision_making.src.messages.route_plan_message import DataRoutePlan, RoutePlanRoadSegments, RoutePlanLaneSegment, \
    RoutePlanRoadSegment
from decision_making.src.messages.scene_static_message import SceneLaneSegmentBase
from decision_making.src.exceptions import raises, LaneAttributeNotFound, RoadSegmentLaneSegmentMismatch, DownstreamLaneDataNotFound
from decision_making.src.messages.scene_static_enums import RoutePlanLaneSegmentAttr, LaneMappingStatusType, \
    MapLaneDirection, GMAuthorityType, LaneConstructionType
from decision_making.src.global_constants import LANE_ATTRIBUTE_CONFIDENCE_THRESHOLD, MAX_COST, MIN_COST
from decision_making.src.planning.route.route_planner import RoutePlanner, RoutePlannerInputData


class CostBasedRoutePlanner(RoutePlanner):
    def __init__(self):
        super().__init__()

    @staticmethod
    # Following method is kept public in order to unit test the method from outside the class
    def mapping_status_based_occupancy_cost(mapping_status_attribute: LaneMappingStatusType) -> float:
        """
        Cost of lane map type. Current implementation is binary cost.
        :param mapping_status_attribute: type of mapped
        :return: normalized cost (MIN_COST to MAX_COST)
        """
        if ((mapping_status_attribute == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap) or
                (mapping_status_attribute == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_MDMap)):
            return MIN_COST
        return MAX_COST

    @staticmethod
    # Following method is kept public in order to unit test the method from outside the class
    def construction_zone_based_occupancy_cost(construction_zone_attribute: LaneConstructionType) -> float:
        """
        Cost of construction zone type. Current implementation is binary cost.
        :param construction_zone_attribute: type of lane construction
        :return: Normalized cost (MIN_COST to MAX_COST)
        """
        if ((construction_zone_attribute == LaneConstructionType.CeSYS_e_LaneConstructionType_Normal) or
                (construction_zone_attribute == LaneConstructionType.CeSYS_e_LaneConstructionType_Unknown)):
            return MIN_COST
        return MAX_COST

    @staticmethod
    # Following method is kept public in order to unit test the method from outside the class
    def lane_dir_in_route_based_occupancy_cost(lane_dir_in_route_attribute: MapLaneDirection) -> float:
        """
        Cost of lane direction. Current implementation is binary cost.
        :param lane_dir_in_route_attribute: map lane direction in respect to host
        :return: Normalized cost (MIN_COST to MAX_COST)
        """
        if ((lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle) or
                (lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_Left_Towards_HostVehicle) or
                (lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_Right_Towards_HostVehicle)):
            return MIN_COST
        return MAX_COST

    @staticmethod
    # Following method is kept public in order to unit test the method from outside the class
    def gm_authority_based_occupancy_cost(gm_authority_attribute: GMAuthorityType) -> float:
        """
        Cost of GM authorized driving area. Current implementation is binary cost.
        :param gm_authority_attribute: type of GM authority
        :return: Normalized cost (MIN_COST to MAX_COST)
        """
        if gm_authority_attribute == GMAuthorityType.CeSYS_e_GMAuthorityType_None:
            return MIN_COST
        return MAX_COST

    @staticmethod
    @raises(LaneAttributeNotFound)
    # Following method is kept public in order to unit test the method from outside the class
    def lane_attribute_based_occupancy_cost(lane_attribute_index: int, lane_attribute_value: int) -> float:
        """
        This method is a wrapper on the individual lane attribute cost calculations and arbitrates
        according to the (input) lane attribute, which lane attribute method to invoke
        :param lane_attribute_index: pointer to the concerned lane attribute in RoutePlanLaneSegmentAttr enum
        :param lane_attribute_value: value of the pointed lane attribute
        :return: Normalized lane occupancy cost based on the concerned lane attribute (MIN_COST to MAX_COST)
        """
        attribute_based_occupancy_cost_methods = {}
        if 'attribute_based_occupancy_cost_methods' not in CostBasedRoutePlanner.lane_attribute_based_occupancy_cost.__dict__:
            # The above if check and then setting of attribute_based_occupancy_cost_methods within the if block is equivalent of
            # making attribute_based_occupancy_cost_methods a static dictionary (of [lane_attribute_index, lane attribute based occupancy_cost
            # calculation methods])
            attribute_based_occupancy_cost_methods = {
                RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus:
                    CostBasedRoutePlanner.mapping_status_based_occupancy_cost,

                RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA:
                    CostBasedRoutePlanner.gm_authority_based_occupancy_cost,

                RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction:
                    CostBasedRoutePlanner.construction_zone_based_occupancy_cost,

                RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Direction:
                    CostBasedRoutePlanner.lane_dir_in_route_based_occupancy_cost
            }

        if lane_attribute_index in attribute_based_occupancy_cost_methods:
            # Following is equivalent of (pythonic way) executing a switch statement
            occupancy_cost_method = attribute_based_occupancy_cost_methods[lane_attribute_index]
            return occupancy_cost_method(lane_attribute_value)
        else:
            raise LaneAttributeNotFound('Cost Based Route Planner: lane_attribute_index {0} not supported'.format(lane_attribute_index))

    @staticmethod
    @raises(LaneAttributeNotFound)
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
                raise LaneAttributeNotFound('Cost Based Route Planner: lane_attribute_index {0} doesn\'t have corresponding lane attribute value'
                                            .format(lane_attribute_index))

            if lane_attribute_index < len(lane_segment_base_data.a_cmp_lane_attribute_confidences):
                lane_attribute_confidence = lane_segment_base_data.a_cmp_lane_attribute_confidences[lane_attribute_index]
            else:
                raise LaneAttributeNotFound('Cost Based Route Planner: lane_attribute_index {0} doesn\'t have corresponding lane attribute '
                                            'confidence value'.format(lane_attribute_index))

            if (lane_attribute_confidence < LANE_ATTRIBUTE_CONFIDENCE_THRESHOLD):
                continue

            lane_attribute_occupancy_cost = CostBasedRoutePlanner.lane_attribute_based_occupancy_cost(
                lane_attribute_index=lane_attribute_index, lane_attribute_value=lane_attribute_value)
            # Check if the lane is unoccupiable
            if (lane_attribute_occupancy_cost == MAX_COST):
                return MAX_COST

        return MIN_COST

    @abstractmethod
    @raises(DownstreamLaneDataNotFound)
    def _lane_end_cost_calc(self, lane_segment_base_data: SceneLaneSegmentBase) -> (float, bool):
        """
        Calculates lane end cost for a single lane segment
        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        :return:
            (float): lane_end_cost, cost to the AV if it reaches the lane end
            (bool):  lane connectivity diagnostics info, whether at least one downstream lane segment
                     (as described in the map) is in the downstream route road segment
        """
        pass

    def _lane_cost_calc(self, lane_segment_base_data: SceneLaneSegmentBase) -> (RoutePlanLaneSegment, bool):
        """
        Calculates lane end and occupancy cost for a single lane segment
        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        :return:
            RoutePlanLaneSegment: combined end and occupancy cost info for the lane
            (bool): True if any downstream lane segment is found
        """
        lane_segment_id = lane_segment_base_data.e_i_lane_segment_id

        # Calculate lane occupancy costs for a lane
        lane_occupancy_cost = CostBasedRoutePlanner.lane_occupancy_cost_calc(lane_segment_base_data)

        # Calculate lane end costs
        if not self._route_plan_lane_segments_reversed:  # if route_plan_lane_segments is empty indicating the last segment in route
            if lane_occupancy_cost == MAX_COST:  # Can't occupy the lane, can't occupy the end either. end cost must be MAX(=MAX_COST)
                lane_end_cost = MAX_COST
            else:
                lane_end_cost = MIN_COST

            downstream_lane_found_in_route = True
            # Because this is the last road segment in (current) route  we don't want to trigger RoadSegmentLaneSegmentMismatch
            # exception by running diagnostics on the downstream to the last road segment, in route.
        else:

            lane_end_cost, downstream_lane_found_in_route = self._lane_end_cost_calc(lane_segment_base_data=lane_segment_base_data)

            if lane_occupancy_cost == MAX_COST:  # Can't occupy the lane, can't occupy the end either. end cost must be MAX(=MAX_COST)
                lane_end_cost = MAX_COST

        # Construct RoutePlanLaneSegment for the lane and add to the RoutePlanLaneSegment container for this Road Segment
        current_route_lane_segment = RoutePlanLaneSegment(e_i_lane_segment_id=lane_segment_id,
                                                          e_cst_lane_occupancy_cost=lane_occupancy_cost,
                                                          e_cst_lane_end_cost=lane_end_cost)

        return current_route_lane_segment, downstream_lane_found_in_route

    @raises(RoadSegmentLaneSegmentMismatch)
    def _road_segment_cost_calc(self, road_segment_id: int) -> RoutePlanRoadSegment:
        """
        Itreratively uses lane_cost_calc method to calculate lane costs (occupancy and end) for all lane segments in a road segment
        :return:
        RoutePlanRoadSegment, which is List[RoutePlanLaneSegments]
        Also raises RoadSegmentLaneSegmentMismatch internally if it can't find any downstream lane segment in the route
        """
        route_lane_segments: RoutePlanRoadSegment = []

        # As the name suggests, if there is NO downstream lane (as defined in map) to the current
        # road segment (any of its lanes) that is in the route
        downstream_road_segment_not_found = True

        if not self._route_plan_lane_segments_reversed:  # check if this is the last road segment in the nav plan
            downstream_road_segment_not_found = False

        lane_segment_ids = self._route_plan_input_data.get_lane_segment_ids_for_road_segment(road_segment_id)

        # Now iterate over all the lane segments inside  the lane_segment_ids (ndarray)
        # value -> lane_segment_id
        for lane_segment_id in lane_segment_ids:

            lane_segment_base_data = self._route_plan_input_data.get_lane_segment_base(lane_segment_id)

            route_lane_segment, downstream_lane_found_in_route = self._lane_cost_calc(lane_segment_base_data=lane_segment_base_data)

            downstream_road_segment_not_found = downstream_road_segment_not_found and not(downstream_lane_found_in_route)

            route_lane_segments.append(route_lane_segment)

        if downstream_road_segment_not_found:
            raise RoadSegmentLaneSegmentMismatch('Cost Based Route Planner: Not a single downstream lane segment for the current '
                                                 'road segment ID {0} were found in the route plan downstream road segment ID {1} '
                                                 'described in the navigation plan'.format(
                                                    road_segment_id,
                                                    self._route_plan_input_data.get_next_road_segment_id(road_segment_id)))
        return route_lane_segments

    @prof.ProfileFunction()
    def plan(self, route_plan_input_data: RoutePlannerInputData) -> DataRoutePlan:
        """
        Calculates lane end and occupancy costs for all the lanes in the NAV plan
        :input:  RoutePlannerInputData, pre-processed data for RoutePlan cost calcualtions. More details at
                 RoutePlannerInputData() class definition.
        :return: DataRoutePlan , the complete route plan information ready to be serialized and published
        """
        road_segment_ids_reversed: List[int] = []
        num_lane_segments_reversed: List[int] = []

        # iterate over all road segments in the route plan in the reverse sequence. Enumerate the iterable to get the index also
        # key -> road_segment_id
        # value -> lane_segment_ids

        self._route_plan_lane_segments_reversed: RoutePlanRoadSegments = []
        self._route_plan_input_data = route_plan_input_data

        for (road_segment_id, lane_segment_ids) in reversed(self._route_plan_input_data.get_lane_segment_ids_for_route().items()):
            # If self._route_plan_lane_segments_reversed is empty, it is the first time through this loop.
            route_lane_segments = self._road_segment_cost_calc(road_segment_id=road_segment_id)

            # append the road segment specific info , as the road seg iteration is reverse
            road_segment_ids_reversed.append(road_segment_id)

            num_lane_segments_reversed.append(len(lane_segment_ids))

            self._route_plan_lane_segments_reversed.append(route_lane_segments)

        return DataRoutePlan(e_b_is_valid=True,
                             e_Cnt_num_road_segments=len(road_segment_ids_reversed),
                             a_i_road_segment_ids=np.array(list(reversed(road_segment_ids_reversed))),
                             a_Cnt_num_lane_segments=np.array(list(reversed(num_lane_segments_reversed))),
                             as_route_plan_lane_segments=list(reversed(self._route_plan_lane_segments_reversed)))
