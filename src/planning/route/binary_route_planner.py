from decision_making.src.global_constants import MAX_COST, MIN_COST
from decision_making.src.messages.route_plan_message import RoutePlanLaneSegment
from decision_making.src.planning.route.binary_cost_based_route_planner import CostBasedRoutePlanner


class BinaryRoutePlanner(CostBasedRoutePlanner):
    """
    child class CostBasedRoutePlanner that contains implementation details of binary route planner
    """
    def __init__(self):
        super().__init__()

    def _calculate_end_cost_from_downstream_lane(self, lane_segment_id: int, downstream_lane_costs: RoutePlanLaneSegment) -> float:
        """
        Calculates lane end cost based on downstream lane segment

        This is a binary end cost implementation based on the downstream lane occupancy cost.

        :param lane_segment_id: Lane ID
        :param downstream_lane_costs: Downstream lane segment cost information from RP
        :return: Lane end cost related to provided downstream lane segment
        """
        if downstream_lane_costs.e_cst_lane_occupancy_cost == MIN_COST:
            return MIN_COST
        else:
            return MAX_COST
