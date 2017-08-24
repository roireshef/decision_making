from abc import ABCMeta, abstractmethod
from logging import Logger

from decision_making.src.messages.dds_typed_message import DDSTypedMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState


class PolicyConfig(DDSTypedMsg):
    """
    Parameters configuration class, loaded from parameter server
    """
    def __init__(self):
        pass


class Policy(metaclass=ABCMeta):
    def __init__(self, logger: Logger, policy_config: PolicyConfig):
        """
        Receives configuration and logger
        :param logger: logger
        :param policy_config: parameters configuration class, loaded from parameter server
        """
        self._policy_config = policy_config
        self.logger = logger

    @abstractmethod
    def plan(self, behavioral_state: BehavioralState) -> (TrajectoryParams, BehavioralVisualizationMsg):
        """
        Plan according to the behavioral state and return trajectory parametrs that
        allow the Trajectory planner to evaluate & choose the best available trajectory
        :param behavioral_state:
        :return: TrajectoryParameters for behavioral planner, BehavioralVisualizationMsg for visualization purposes
        """
        pass

