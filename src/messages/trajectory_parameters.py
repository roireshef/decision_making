import numpy as np

from src.messages.dds_message import DDSMessage


class TrajectoryParameters(DDSMessage):
    def __init__(self, reference_route: np.ndarray, target_state: np.array, cost_params: list):
        """
        The struct used for communicating the behavioral plan to the trajectory planner.
        :param reference_route: of type np.ndarray, with rows of [(x ,y, theta)] where x, y, theta are floats
        :param target_state: of type np.array (x,y, theta, v) all of which are floats.
        :param cost_params: list of parameters for our predefined functions. TODO define this
        """
        self._reference_route = reference_route
        self._target_state = target_state
        self._cost_params = cost_params

    @property
    def reference_route(self): return self._reference_route

    @property
    def target_state(self): return self._target_state

    @property
    def cost_params(self): return self._cost_params
