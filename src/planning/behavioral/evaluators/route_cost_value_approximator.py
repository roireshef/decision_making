from logging import Logger

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.evaluators.value_approximator import ValueApproximator
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.global_constants import LANE_END_COST_IND


from typing import Any


class RouteCostValueApproximator(ValueApproximator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def approximate(self, behavioral_state: BehavioralGridState, route_plan: RoutePlan, goal: Any) -> float:
        """
        Returns the cost of traversing the SAME_LANE GFF based on the route plan costs.
        :param behavioral_state:
        :param route_plan:
        :param goal: Currently unused
        :return:  total cost
        """
        route_costs_dict = route_plan.to_costs_dict()
        target_gff = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE]

        total_cost = 0.

        for lane_id in target_gff.segment_ids:
            if route_costs_dict.get(lane_id):
                total_cost += route_costs_dict[lane_id][LANE_END_COST_IND]

            # if lane_id is not in the route_plan, add a saturated cost, as it is untraversable
            else:
                total_cost += 1

        return total_cost

