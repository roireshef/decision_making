from enum import Enum
from typing import Tuple

from src.planning.trajectory.cost_function import TrajectoryCostParams
from src.planning.trajectory.trajectory_planner import TrajectoryPlanner
from src.state.enriched_state import State as EnrichedState

import numpy as np


class TrajectoryPlanningFacade:
    """
        The trajectory planning facade handles trajectory planning requests and redirects them to the relevant planner
    """

    def __init__(self, strategy_handlers: dict):
        """
        :param strategy_handlers: a dictionary of trajectory planners as strategy handlers -
        types are {TrajectoryPlanningStrategy: TrajectoryPlanner}
        """
        self.__validate_strategy_handlers(strategy_handlers)
        self.__strategy_handlers = strategy_handlers

    def plan(self, strategy: TrajectoryPlanningStrategy):
        """
        will execute planning with using the implementation for the desired planning-strategy provided
        :param strategy: desired planning strategy
        :return: no return value. results are published in self.__publish_results()
        """
        state = self.__read_current_state()
        ref_route, goal, cost_params = self.__read_mission_specs()

        trajectory, cost, debug_results = self.__strategy_handlers[strategy].plan(state, ref_route, goal, cost_params)

        # TODO: publish cost to behavioral layer?

        self.__publish_trajectory(trajectory)
        self.__publish_debug(debug_results)

    # TODO: should also be published to DDS logger
    @staticmethod
    def __validate_strategy_handlers(strategy_handlers):
        for elem in TrajectoryPlanningStrategy.__members__.values():
            if not strategy_handlers.keys().__contains__(elem):
                raise KeyError('strategy_handlers does not contain a  record for ' + elem)
            if not isinstance(strategy_handlers[elem], TrajectoryPlanner):
                raise ValueError('strategy_handlers does not contain a TrajectoryPlanner impl. for ' + elem)

    # TODO: implement message passing
    def __read_current_state(self) -> EnrichedState:
        pass

    def __read_mission_specs(self) -> Tuple[np.ndarray, np.ndarray, TrajectoryCostParams]:
        pass

    def __publish_trajectory(self, results):
        pass

    def __publish_debug(self, debug_data):
        pass


class TrajectoryPlanningStrategy(Enum):
    HIGHWAY = 0
    TRAFFIC_JAM = 1
    PARKING = 2
