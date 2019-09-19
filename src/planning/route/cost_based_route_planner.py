import numpy as np
import rte.python.profiler as prof
from decision_making.src.exceptions import RoadSegmentLaneSegmentMismatch, raises, DownstreamLaneDataNotFound
from decision_making.src.global_constants import MAX_COST, MIN_COST, ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR
from decision_making.src.messages.route_plan_message import RoutePlanLaneSegment, DataRoutePlan, \
    RoutePlanRoadSegment, RoutePlanRoadSegments
from decision_making.src.messages.scene_static_message import SceneLaneSegmentBase
from decision_making.src.planning.route.route_planner import RoutePlanner, RoutePlannerInputData
from decision_making.src.utils.map_utils import MapUtils
from typing import List


class CostBasedRoutePlanner(RoutePlanner):
    """
    child class (of abstract class RoutePlanner), which contains implementation details of binary cost based route planner
    """

    def __init__(self):
        super().__init__()

    @raises(DownstreamLaneDataNotFound)
    def _lane_end_cost_calc(self, lane_segment_base_data: SceneLaneSegmentBase) -> (float, bool):
        """
        Calculates lane end cost for a single lane segment
        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        (already evaluated, downstrem of the concerned lane). We mainly need the lane occupancy cost from here.
        :return:
            (float): lane_end_cost, cost to the AV if it reaches the lane end
            (bool):  lane connectivity diagnostics info, whether at least one downstream lane segment
                     (as described in the map) is in the downstream route road segment
        """
        min_downstream_lane_segment_occupancy_cost = MAX_COST

        # Search iteratively for the next segment lanes that are downstream to the current lane and in the route.
        # Assign the end cost of current lane = Min of (occ. cost + discounted end cost) for all downstream lanes
        # search through all downstream lanes to current lane
        downstream_lane_found_in_route = False

        downstream_route_lane_segments: RoutePlanRoadSegment = self._route_plan_lane_segments_reversed[-1]

        downstream_route_lane_segment_ids = np.array([route_lane_segment.e_i_lane_segment_id
                                                      for route_lane_segment in downstream_route_lane_segments])

        for downstream_base_lane_segment in lane_segment_base_data.as_downstream_lanes:

            downstream_lane_segment_id = downstream_base_lane_segment.e_i_lane_segment_id

            # All lane IDs in downstream road segment in route to the currently indexed road segment in the loop
            if downstream_lane_segment_id in downstream_route_lane_segment_ids:  # verify if the downstream lane is in the route (it may not be ex: fork/exit)

                downstream_lane_found_in_route = True

                # find the index corresponding to the lane seg ID in the road segment
                downstream_route_lane_segment_idx = np.where(downstream_route_lane_segment_ids == downstream_lane_segment_id)[0][0]

                if downstream_route_lane_segment_idx < len(downstream_route_lane_segments):
                    downstream_route_lane_segment = downstream_route_lane_segments[downstream_route_lane_segment_idx]
                else:
                    raise DownstreamLaneDataNotFound('Binary Cost Based Route Planner: downstream lane segment ID {0} for lane segment ID {1} '
                                                     'not present in route_plan_lane_segment structure for downstream road segment'
                                                     .format(downstream_lane_segment_id, lane_segment_base_data.e_i_lane_segment_id))

                downstream_lane_segment_occupancy_cost = downstream_route_lane_segment.e_cst_lane_occupancy_cost

                downstream_lane_segment_length = MapUtils.get_lane(downstream_lane_segment_id).e_l_length

                backprop_downstream_lane_segment_end_cost = downstream_route_lane_segment.e_cst_lane_end_cost * \
                                                            ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR ** \
                                                            downstream_lane_segment_length

                min_downstream_lane_segment_occupancy_cost = min(min(min_downstream_lane_segment_occupancy_cost,
                                                                     downstream_lane_segment_occupancy_cost +
                                                                     backprop_downstream_lane_segment_end_cost),
                                                                 MAX_COST)
            else:
                # Downstream lane segment not in route. Do nothing.
                pass

        lane_end_cost = min_downstream_lane_segment_occupancy_cost
        return lane_end_cost, downstream_lane_found_in_route

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
        lane_occupancy_cost = RoutePlanner.lane_occupancy_cost_calc(lane_segment_base_data)

        # Calculate lane end costs (from lane occupancy costs and downstream lane end costs)
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
            raise RoadSegmentLaneSegmentMismatch('Binary Cost Based Route Planner: Not a single downstream lane segment for the current '
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
