import numpy as np
from logging import Logger

from decision_making.src.global_constants import WERLING_TIME_RESOLUTION, EFFICIENCY_COST_WEIGHT, \
    RIGHT_LANE_COST_WEIGHT, BP_METRICS_LANE_DEVIATION_COST_WEIGHT
from decision_making.src.planning.behavioral.architecture.components.evaluators.cost_functions import \
    PlanEfficiencyMetric, PlanComfortMetric
from decision_making.src.planning.behavioral.architecture.components.evaluators.velocity_profile import VelocityProfile
from decision_making.src.planning.behavioral.architecture.data_objects import AggressivenessLevel
from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState


class ValueApproximator:
    def __init__(self, logger: Logger):
        self.logger = logger

    #def evaluate_state(self, behavioral_state: SemanticBehavioralGridState) -> float:
    def evaluate_state(self, time_period: float, vel: float, lane: int, T_d: float, T_d_full: float) -> float:
        if time_period < 0:
            return 0
        efficiency_cost = PlanEfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([vel]))[0] * \
                          EFFICIENCY_COST_WEIGHT * time_period / WERLING_TIME_RESOLUTION

        right_lane_cost = lane_deviation_cost = comfort_cost = 0
        if lane > 0:
            right_lane_cost = lane * time_period * RIGHT_LANE_COST_WEIGHT / WERLING_TIME_RESOLUTION
            lane_deviation_cost = lane * T_d * BP_METRICS_LANE_DEVIATION_COST_WEIGHT / WERLING_TIME_RESOLUTION
            comfort_cost = lane * PlanComfortMetric.calc_cost(VelocityProfile(0, 0, 0, 0, 0, 0), T_d_full, T_d_full,
                                                              AggressivenessLevel.CALM)

        cost = efficiency_cost + comfort_cost + right_lane_cost + lane_deviation_cost
        return cost
