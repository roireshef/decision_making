from abc import ABCMeta, abstractmethod
from src.planning.behavioral.behavioral_state import BehavioralState
from src.planning.messages.behavioral_debug_data import BehavioralDebugData
from src.planning.messages.trajectory_parameters import TrajectoryParameters


class Policy(metaclass=ABCMeta):
    def __init__(self, policy_params: dict):
        self._policy_params = policy_params

    @abstractmethod
    def plan(self, behavioral_state: BehavioralState,
             behavioral_debug_obj: BehavioralDebugData) -> TrajectoryParameters:
        pass

