from decision_making.src.global_constants import BACKPROP_DISCOUNT_FACTOR, LANE_LENGTH_SCALE_FACTOR, MIN_COST, BACKPROP_COST_THRESHOLD
from decision_making.src.messages.route_plan_message import RoutePlanLaneSegment
from decision_making.src.planning.route.binary_cost_based_route_planner import CostBasedRoutePlanner
from decision_making.src.utils.map_utils import MapUtils


class BackpropagatingRoutePlanner(CostBasedRoutePlanner):
    """
    child class CostBasedRoutePlanner that contains implementation details of backpropagating route planner
    """
    def __init__(self):
        super().__init__()

    def _calculate_end_cost_from_downstream_lane(self, downstream_lane_segment: RoutePlanLaneSegment) -> float:
        """
        Calculates lane end cost based on downstream lane segment

        The end cost from the downstream lane is propagated backwards using the following equation:

                                                   (l / alpha)
            backpropagated end cost = cost * gamma^

            where cost = end cost for downstream lane
                 gamma = discount factor
                     l = length of downstream lane
                 alpha = scaling factor for lane length

        :param downstream_lane_segment: Downstream lane segment information from RP
        :return: Lane end cost related to provided downstream lane segment
        """
        downstream_lane_segment_length = MapUtils.get_lane(downstream_lane_segment.e_i_lane_segment_id).e_l_length

        backprop_end_cost = downstream_lane_segment.e_cst_lane_end_cost \
                            * (BACKPROP_DISCOUNT_FACTOR ** (downstream_lane_segment_length / LANE_LENGTH_SCALE_FACTOR))

        lane_segment_end_cost = backprop_end_cost if backprop_end_cost > BACKPROP_COST_THRESHOLD else MIN_COST

        return lane_segment_end_cost
