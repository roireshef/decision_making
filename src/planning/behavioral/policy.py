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


class Policy(metaclass=ABCMeta):
    def __init__(self, logger: Logger, predictor: Predictor):
        """
        Receives configuration and logger
        :param logger: logger
        :param predictor: used for predicting ego and other dynamic objects in future states
        """
        self._predictor = predictor
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
