import numpy as np
from logging import Logger

from decision_making.src.global_constants import WERLING_TIME_RESOLUTION, EFFICIENCY_COST_WEIGHT, \
    RIGHT_LANE_COST_WEIGHT, BP_METRICS_LANE_DEVIATION_COST_WEIGHT
from decision_making.src.planning.behavioral.architecture.components.evaluators.cost_functions import \
    PlanEfficiencyMetric
from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState


class ValueApproximator:
    def __init__(self, logger: Logger):
        self.logger = logger

    #def evaluate_state(self, behavioral_state: SemanticBehavioralGridState) -> float:
    def evaluate_state(self, time_period: float, vel: float, lane: int, lane_change_time: float) -> float:
        if time_period < 0:
            return 0
        efficiency_cost = PlanEfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([vel]))[0] * \
                          time_period / WERLING_TIME_RESOLUTION
        right_lane_cost = lane * time_period / WERLING_TIME_RESOLUTION
        lane_deviation_cost = lane * lane_change_time / WERLING_TIME_RESOLUTION

        cost = efficiency_cost * EFFICIENCY_COST_WEIGHT + \
               right_lane_cost * RIGHT_LANE_COST_WEIGHT + \
               lane_deviation_cost * BP_METRICS_LANE_DEVIATION_COST_WEIGHT
        return cost
