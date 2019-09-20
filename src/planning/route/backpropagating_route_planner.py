import numpy as np
from decision_making.src.exceptions import raises, DownstreamLaneDataNotFound
from decision_making.src.global_constants import MAX_COST, ROUTE_PLAN_BACKPROP_DISCOUNT_FACTOR
from decision_making.src.messages.route_plan_message import RoutePlanRoadSegment
from decision_making.src.messages.scene_static_message import SceneLaneSegmentBase
from decision_making.src.planning.route.cost_based_route_planner import CostBasedRoutePlanner
from decision_making.src.utils.map_utils import MapUtils


class BackpropagatingRoutePlanner(CostBasedRoutePlanner):
    """
    child class CostBasedRoutePlanner that contains implementation details of backpropagating route planner
    """
    def __init__(self):
        super().__init__()

    @raises(DownstreamLaneDataNotFound)
    def _lane_end_cost_calc(self, lane_segment_base_data: SceneLaneSegmentBase) -> (float, bool):
        """
        Calculates lane end cost for a single lane segment

        TODO: Add implementation details

        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        :return:
            (float): lane_end_cost, cost to the AV if it reaches the lane end
            (bool):  lane connectivity diagnostics info, whether at least one downstream lane segment
                     (as described in the map) is in the downstream route road segment
        """
        min_downstream_lane_segment_occupancy_cost = MAX_COST

        # Search iteratively for the next lane segments that are downstream to the current lane and in the route.
        downstream_lane_found_in_route = False

        downstream_route_lane_segments: RoutePlanRoadSegment = self._route_plan_lane_segments_reversed[-1]

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
