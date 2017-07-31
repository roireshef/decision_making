from abc import ABCMeta, abstractmethod

from decision_making.src.messages.trajectory_parameters import TrajectoryParameters
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMessage
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState


class Policy(metaclass=ABCMeta):
    def __init__(self, policy_params: dict):
        self._policy_params = policy_params

    @abstractmethod
    def plan(self, behavioral_state: BehavioralState) -> (TrajectoryParameters, BehavioralVisualizationMessage):
        pass

