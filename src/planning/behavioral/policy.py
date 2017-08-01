from abc import ABCMeta, abstractmethod

import numpy as np
from decision_making.src.messages.trajectory_parameters import TrajectoryParameters, TrajectoryCostParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState


class Policy(metaclass=ABCMeta):
    def __init__(self, policy_params: dict):
        self._policy_params = policy_params

    @abstractmethod
    def plan(self, behavioral_state: BehavioralState) -> (TrajectoryParameters, BehavioralVisualizationMsg):
        pass


class DefaultPolicy(Policy):
    def __init__(self, policy_params: dict):
        super().__init__(policy_params=policy_params)

    def plan(self, behavioral_state: BehavioralState):
        reference_route = np.array([0, 0, 0])
        target_state = np.array([0, 0, 0, 0])
        cost_params = TrajectoryCostParams(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        trajectory_parameters = TrajectoryParameters(reference_route=reference_route, target_state=target_state,
                                                     cost_params=cost_params)

        visualization_message = BehavioralVisualizationMsg(reference_route=reference_route)
        return trajectory_parameters, visualization_message
