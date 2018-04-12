from abc import ABCMeta
from logging import Logger
from typing import List
import numpy as np

from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.state.state import State


class AdvancedPredictor(metaclass=ABCMeta):
    """
    Base class for advanced prediction logic, which is able to account for reactions to ego's action
    """

    def __init__(self, logger: Logger):
        self._logger = logger

    def predict_state(self, state: State, prediction_timestamps: np.ndarray, action_trajectory: SamplableTrajectory)\
            -> (List[State]):
        """
        Predictes the future states of the given state, for the specified timestamps
        :param state: the initial state to begin prediction from
        :param prediction_timestamps: np array of timestamps in [sec] to predict states for. In ascending order.
        Global, not relative
        :param action_trajectory: the ego's planned action trajectory
        :return: a list of non markov predicted states for the requested prediction_timestamp, and a full state for the
        terminal predicted state
        """
        pass
