from abc import ABCMeta, abstractmethod
from logging import Logger

from decision_making.src.messages.dds_typed_message import DDSTypedMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParameters
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState


class PolicyConfig(DDSTypedMsg):
    pass


class Policy(metaclass=ABCMeta):
    def __init__(self, logger: Logger, policy_config: PolicyConfig):
        self._policy_config = policy_config
        self.logger = logger

    @abstractmethod
    def plan(self, behavioral_state: BehavioralState) -> (TrajectoryParameters, BehavioralVisualizationMsg):
        pass

