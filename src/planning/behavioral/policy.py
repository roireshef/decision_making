from abc import ABCMeta, abstractmethod

from src.messages.visualization_message import RvizVisualizationMessage

from src.messages.trajectory_parameters import TrajectoryParameters
from src.planning.behavioral.behavioral_state import BehavioralState


class Policy(metaclass=ABCMeta):
    def __init__(self, policy_params: dict):
        self._policy_params = policy_params

    @abstractmethod
    def plan(self, behavioral_state: BehavioralState) -> (TrajectoryParameters, list[RvizVisualizationMessage]):
        pass

