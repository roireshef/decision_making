from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from src.planning.trajectory.cost_function import CostParams
from src.state.enriched_state import State as EnrichedState


class TrajectoryPlanner(metaclass=ABCMeta):
    # TODO: object type-hint should be changed to DDSMessage type once commited
    @abstractmethod
    def plan(self, state: EnrichedState, reference_route: np.ndarray, goal: np.ndarray, cost_params: CostParams) -> \
            Tuple[np.ndarray, float, object]:
        """
        Plans a trajectory according to the specifications in the arguments
        :param state: environment & ego state object
        :param reference_route: a reference route (often the center of lane). A numpy array of the shape [-1, 2]
        :param goal: A numpy array of the desired ego-state to plan towards, from utils.columns (ego coord-frame)
        :param cost_params: a dictionary of parameters that specify how to build the planning's cost function
        :return: a tuple of: (numpy array: trajectory - each row is [x, y, yaw, velocity], trajectory cost,
        debug results dictionary)
        """
        pass
