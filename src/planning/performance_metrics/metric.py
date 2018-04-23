import numpy as np
from decision_making.src.messages.str_serializable import StrSerializable
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.types import CartesianExtendedTrajectories, FrenetTrajectories2D
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State


class PMState:
    def __init__(self, state: State, ctrajectories: CartesianExtendedTrajectories, ftrajectories: FrenetTrajectories2D,
                 global_time_samples: np.ndarray, predictor: Predictor,
                 lane_width: float=None, reference_route_lat: float=None):
        """
        Extended state for performance metrics, containing:
        the current state, trajectories, time samples, road data, predictor
        :param state: the current state
        :param ctrajectories: cartesian trajectories; the metrics are calculated on their points
        :param ftrajectories: frenet trajectories, transformed from ctrajectories
        :param global_time_samples: array([sec]) array of global time samples starting from ego.timestamp_sec.
                we assume that global_time_samples are evenly spaced (jerk derives accelerations by time)
        :param predictor: predictor for other objects
        :param lane_width: lane width on the current road
        :param reference_route_lat: [m] latitude of the reference route
        """
        self.state = state
        self.ctrajectories = ctrajectories
        self.ftrajectories = ftrajectories
        self.time_samples = global_time_samples  # assume that global_time_samples are evenly spaced
        self.predictor = predictor
        self.lane_width = lane_width
        self.reference_route_lat = reference_route_lat


class Metric(StrSerializable):
    @staticmethod
    def calc_pointwise_cost(pm_state: PMState, params: TrajectoryCostParams) -> np.ndarray:
        """
        Abstract method that is implemented for specific metrics
        :param pm_state: contains the current state, trajectories, road parameters, predictor
        :param params: parameters for the cost function (from behavioral layer)
        :return: 2D matrix of size NxT of point-wise costs, where N = ctrajectory.shape[0] trajectories number,
        T = len(time_samples) time samples number or trajectory length.
        """
        pass

    @classmethod
    def calc_cost(cls, pm_state: PMState, params: TrajectoryCostParams) -> np.ndarray:
        """
        calculate cost for each trajectory in pm_state
        :param pm_state: contains the current state, trajectories, road parameters, predictor
        :param params: parameters for the cost function (from behavioral layer)
        :return: array of costs per trajectory in pm_state
        """
        pointwise_costs = cls.calc_pointwise_cost(pm_state, params)
        return np.sum(pointwise_costs, axis=1)
