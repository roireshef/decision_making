from typing import List
import numpy as np

from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.performance_metrics.metric import Metric, PMState


class Reward:
    def __init__(self, metrics_list: List[Metric], metric_weights: np.array):
        """
        initialization of Reward class
        :param metrics_list: list of difference metrics
        :param metric_weights: array of metrics' weights
        """
        assert len(metrics_list) == metric_weights.shape[0]
        self.metrics = metrics_list
        self.weights = metric_weights

    def calc_reward(self, pm_state: PMState, params: TrajectoryCostParams) -> np.array:
        """
        calc total reward function (cost) per trajectory for the given state, as weighted sum of costs in metrics list
        :param pm_state: contains the current state, trajectories, road parameters, predictor
        :param params: parameters for the cost function (from behavioral layer)
        :return: array of total rewards per input trajectory
        """
        costs = np.array([metric.calc_cost(pm_state, params) for metric in self.metrics])
        weights = np.transpose(np.tile(self.weights, (costs.shape[1], 1)))
        return np.sum(costs * weights, axis=0)

    def calc_pointwise_reward(self, pm_state: PMState, params: TrajectoryCostParams) -> np.array:
        """
        calc total reward function (cost) per trajectory point for the given state,
        as weighted sum of costs in metrics list
        :param pm_state: contains the current state, trajectories, road parameters, predictor
        :param params: parameters for the cost function (from behavioral layer)
        :return: NxM matrix of total rewards per input trajectory and time point
        """
        costs = np.array([metric.calc_pointwise_cost(pm_state, params) for metric in self.metrics])
        weights = np.swapaxes(np.swapaxes(np.tile(self.weights, (costs.shape[1], costs.shape[2], 1)), 1, 2), 0, 1)
        return np.sum(costs * weights, axis=0)
