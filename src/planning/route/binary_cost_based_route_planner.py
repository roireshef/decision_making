from abc import abstractmethod, ABCMeta
from typing import List

import numpy as np
import rte.python.profiler as prof
from decision_making.src.exceptions import raises, LaneAttributeNotFound, RoadSegmentLaneSegmentMismatch, \
    DownstreamLaneDataNotFound
from decision_making.src.global_constants import MAX_COST, MIN_COST
from decision_making.src.messages.route_plan_message import DataRoutePlan, RoutePlanRoadSegments, RoutePlanLaneSegment, \
    RoutePlanRoadSegment
from decision_making.src.messages.scene_static_message import SceneLaneSegmentBase
from decision_making.src.planning.route.route_planner_input_data import RoutePlannerInputData
from decision_making.src.planning.route.route_utils import RouteUtils


class RoutePlanner(metaclass=ABCMeta):

    @prof.ProfileFunction()
    def plan(self, route_plan_input_data: RoutePlannerInputData) -> DataRoutePlan:
        """
        Calculates lane end and occupancy costs for all the lanes in the NAV plan. This is done by starting at the end of the nav. plan
        and working towards the beginning. Other than on the last road segment in the nav. plan, the costs are dictated by the occupancy
        and end costs of the downstream lanes.
        :input:  RoutePlannerInputData, pre-processed data for RoutePlan cost calculations. More details at
                 RoutePlannerInputData() class definition.
        :return: DataRoutePlan , the complete route plan information ready to be serialized and published
        """
        road_segment_ids_reversed: List[int] = []
        num_lane_segments_reversed: List[int] = []

        # iterate over all road segments in the route plan in the reverse sequence. Enumerate the iterable to get the index also
        # key -> road_segment_id
        # value -> lane_segment_ids

        route_plan_lane_segments_reversed: RoutePlanRoadSegments = []

        for road_segment_id, lane_segment_ids in reversed(route_plan_input_data.get_lane_segment_ids_for_route().items()):
            # If route_plan_lane_segments_reversed is empty, it is the first time through this loop.
            route_lane_segments = self._road_segment_cost_calc(road_segment_id=road_segment_id,
                                                               route_plan_lane_segments_reversed=route_plan_lane_segments_reversed,
                                                               route_plan_input_data=route_plan_input_data)

            # append the road segment specific info , as the road seg iteration is reverse
            road_segment_ids_reversed.append(road_segment_id)

            num_lane_segments_reversed.append(len(lane_segment_ids))

            route_plan_lane_segments_reversed.append(route_lane_segments)

        return DataRoutePlan(e_b_is_valid=True,
                             e_Cnt_num_road_segments=len(road_segment_ids_reversed),
                             a_i_road_segment_ids=np.array(list(reversed(road_segment_ids_reversed))),
                             a_Cnt_num_lane_segments=np.array(list(reversed(num_lane_segments_reversed))),
                             as_route_plan_lane_segments=list(reversed(route_plan_lane_segments_reversed)))

    @abstractmethod
    def _calculate_end_cost_from_downstream_lane(self, lane_segment_id: int, downstream_lane_costs: RoutePlanLaneSegment) -> float:
        """
        Calculates lane end cost based on downstream lane segment
        :param lane_segment_id: Lane ID
        :param downstream_lane_costs: Downstream lane segment cost information from RP
        :return: Lane end cost related to provided downstream lane segment
        """
        pass

    @raises(LaneAttributeNotFound)
    def _lane_occupancy_cost_calc(self, lane_segment_base_data: SceneLaneSegmentBase) -> float:
        """
        Calculates lane occupancy cost for a single lane segment
        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        :return: LaneOccupancyCost, cost to the AV if it occupies the lane.
        """
        return MIN_COST if RouteUtils.is_lane_segment_drivable(lane_segment_base_data) else MAX_COST

    @raises(DownstreamLaneDataNotFound)
    def _lane_end_cost_calc(self, lane_segment_base_data: SceneLaneSegmentBase,
                            route_plan_lane_segments_reversed: RoutePlanRoadSegments) -> (float, bool):
        """
        Calculates lane end cost for a single lane segment

        If a downstream lane with an occupancy cost less then MAX_COST doesn't exist, then the end cost will also be
        set to MAX_COST; otherwise, the end cost will be equal to the minimum end cost that is calculated based on
        each downstream lane.

        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        :return:
            (float): lane_end_cost, cost to the AV if it reaches the lane end
            (bool):  lane connectivity diagnostics info, whether at least one downstream lane segment
                     (as described in the map) is in the downstream route road segment
        """
        min_downstream_lane_segment_occupancy_cost = MAX_COST
        min_lane_segment_end_cost = MAX_COST

        # Search iteratively for the next lane segments that are downstream to the current lane and in the route.
        downstream_lane_found_in_route = False

        downstream_route_lane_segments: RoutePlanRoadSegment = route_plan_lane_segments_reversed[-1]

        downstream_route_lane_segment_ids = np.array([route_lane_segment.e_i_lane_segment_id
                                                      for route_lane_segment in downstream_route_lane_segments])

        for downstream_base_lane_segment in lane_segment_base_data.as_downstream_lanes:

            downstream_lane_segment_id = downstream_base_lane_segment.e_i_lane_segment_id

            # Verify that the downstream lane is in the route (it may not be ex: fork/exit)
            if downstream_lane_segment_id in downstream_route_lane_segment_ids:

                downstream_lane_found_in_route = True

                # find the index corresponding to the lane seg ID in the road segment
                downstream_route_lane_segment_idx = np.where(downstream_route_lane_segment_ids == downstream_lane_segment_id)[0][0]

                if downstream_route_lane_segment_idx < len(downstream_route_lane_segments):
                    downstream_route_lane_segment = downstream_route_lane_segments[downstream_route_lane_segment_idx]
                else:
                    raise DownstreamLaneDataNotFound('Backpropagating Route Planner: downstream lane segment ID {0} for lane segment ID {1} '
                                                     'not present in route_plan_lane_segment structure for downstream road segment'
                                                     .format(downstream_lane_segment_id, lane_segment_base_data.e_i_lane_segment_id))

                # Keep track of minimum downstream lane occupancy cost
                min_downstream_lane_segment_occupancy_cost = min(min_downstream_lane_segment_occupancy_cost,
                                                                 downstream_route_lane_segment.e_cst_lane_occupancy_cost)

                # Determine the lane end cost relating to the downstream lane and keep track of the minimum
                lane_segment_end_cost = self._calculate_end_cost_from_downstream_lane(lane_segment_base_data.e_i_lane_segment_id,
                                                                                      downstream_route_lane_segment)

                min_lane_segment_end_cost = min(min_lane_segment_end_cost, lane_segment_end_cost)
            else:
                # Downstream lane segment not in route. Do nothing.
                continue

        if min_downstream_lane_segment_occupancy_cost == MAX_COST:
            lane_end_cost = MAX_COST
        else:
            lane_end_cost = min_lane_segment_end_cost

        return lane_end_cost, downstream_lane_found_in_route

    def _lane_cost_calc(self, lane_segment_base_data: SceneLaneSegmentBase, route_plan_lane_segments_reversed: RoutePlanRoadSegments) -> \
            (RoutePlanLaneSegment, bool):
        """
        Calculates lane end and occupancy cost for a single lane segment
        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        :return:
            RoutePlanLaneSegment: combined end and occupancy cost info for the lane
            (bool): True if any downstream lane segment is found
        """
        # Calculate lane end costs
        if not route_plan_lane_segments_reversed:
            # If _route_plan_lane_segments_reversed is empty, we will reach here.
            # Since the road segments in the navigation plan are processed in reverse order (i.e. furthest to closest),
            # _route_plan_lane_segments_reversed will only be empty when the lane costs for the furthest road segment
            # are being calculated. We don't have any downstream information.
            # So, downstream_lane_found_in_route is set to True because we do not want to raise any exceptions,
            # and lane_end_cost is set to MIN_COST for the following two reasons:
            #  1. Generally, the host will be sufficiently far away from the end of the navigation plan so even
            #     backpropagating these costs should not affect the host's behavior.
            #  2. The host will be close to the end of the navigation plan as the destination is approached.
            #     We do not want to force all lanes to have MAX_COST and cause a takeover to happen just before
            #     reaching the destination.
            lane_end_cost = MIN_COST
            downstream_lane_found_in_route = True
        else:
            lane_end_cost, downstream_lane_found_in_route = self._lane_end_cost_calc(
                lane_segment_base_data=lane_segment_base_data, route_plan_lane_segments_reversed=route_plan_lane_segments_reversed)

        # Calculate lane occupancy costs for a lane
        lane_occupancy_cost = self._lane_occupancy_cost_calc(lane_segment_base_data)

        # If we can't occupy a lane, then we can't be in it at the end either. Override the lane end cost here.
        if lane_occupancy_cost == MAX_COST:
            lane_end_cost = MAX_COST

        # Construct RoutePlanLaneSegment for the lane and add to the RoutePlanLaneSegment container for this Road Segment
        current_route_lane_segment = RoutePlanLaneSegment(e_i_lane_segment_id=lane_segment_base_data.e_i_lane_segment_id,
                                                          e_cst_lane_occupancy_cost=lane_occupancy_cost,
                                                          e_cst_lane_end_cost=lane_end_cost)

        return current_route_lane_segment, downstream_lane_found_in_route

    @raises(RoadSegmentLaneSegmentMismatch)
    def _road_segment_cost_calc(self, road_segment_id: int, route_plan_lane_segments_reversed: RoutePlanRoadSegments,
                                route_plan_input_data: RoutePlannerInputData) -> RoutePlanRoadSegment:
        """
        Iteratively uses lane_cost_calc method to calculate lane costs (occupancy and end) for all lane segments in a road segment
        :return:
        RoutePlanRoadSegment, which is List[RoutePlanLaneSegments]
        Also raises RoadSegmentLaneSegmentMismatch internally if it can't find any downstream lane segment in the route
        """
        route_lane_segments: RoutePlanRoadSegment = []

        # As the name suggests, if there is NO downstream lane (as defined in map) to the current
        # road segment (any of its lanes) that is in the route
        downstream_road_segment_not_found = True

        if not route_plan_lane_segments_reversed:  # check if this is the last road segment in the nav plan
            downstream_road_segment_not_found = False

        lane_segment_ids = route_plan_input_data.get_lane_segment_ids_for_road_segment(road_segment_id)

        # Now iterate over all the lane segments inside  the lane_segment_ids (ndarray)
        # value -> lane_segment_id
        for lane_segment_id in lane_segment_ids:

            lane_segment_base_data = route_plan_input_data.get_lane_segment_base(lane_segment_id)

            route_lane_segment, downstream_lane_found_in_route = self._lane_cost_calc(
                lane_segment_base_data=lane_segment_base_data, route_plan_lane_segments_reversed=route_plan_lane_segments_reversed)

            downstream_road_segment_not_found = downstream_road_segment_not_found and not downstream_lane_found_in_route

            route_lane_segments.append(route_lane_segment)

        if downstream_road_segment_not_found:
            raise RoadSegmentLaneSegmentMismatch('RoutePlanner: Not a single downstream lane segment for the current '
                                                 'road segment ID {0} were found in the route plan downstream road segment ID {1} '
                                                 'described in the navigation plan'.format(
                                                    road_segment_id,
                                                    route_plan_input_data.get_next_road_segment_id(road_segment_id)))
        return route_lane_segments
