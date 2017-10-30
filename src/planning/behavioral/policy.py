from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import Type

from decision_making.src.messages.dds_typed_message import DDSTypedMsg
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State
from mapping.src.model.map_api import MapAPI


class PolicyConfig(DDSTypedMsg):
    """
    Parameters configuration class, loaded from parameter server
    """

    def __init__(self):
        pass


class Policy(metaclass=ABCMeta):
    def __init__(self, logger: Logger, policy_config: PolicyConfig, behavioral_state: BehavioralState,
                 predictor: Predictor, map_api: MapAPI):
        """
        Receives configuration and logger
        :param logger: logger
        :param policy_config: parameters configuration class, loaded from parameter server
        :param behavioral_state: initial state of the system. Can be empty, i.e. initialized with default values.
        :param predictor: used for predicting ego and other dynamic objects in future states
        :param map_api: Map API
        """
        self._map_api = map_api
        self._policy_config = policy_config
        self._predictor = predictor
        self._behavioral_state = behavioral_state
        self.logger = logger

    @abstractmethod
    def plan(self, state: State, nav_plan: NavigationPlanMsg) -> (TrajectoryParams, BehavioralVisualizationMsg):
        """
        Plan according to the behavioral state and return trajectory parametrs that
        allow the Trajectory planner to evaluate & choose the best available trajectory
        :param nav_plan: car's navigation plan
        :param state: world state
        :return: TrajectoryParameters for behavioral planner, BehavioralVisualizationMsg for visualization purposes
        """
        pass
