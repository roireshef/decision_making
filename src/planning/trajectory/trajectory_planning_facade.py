import time
from enum import Enum
from typing import Tuple

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner
from decision_making.src.state.enriched_state import State as EnrichedState

import numpy as np

class TrajectoryPlanningStrategy(Enum):
    HIGHWAY = 0
    TRAFFIC_JAM = 1
    PARKING = 2

class TrajectoryPlanningFacade(DmModule):
    """
        The trajectory planning facade handles trajectory planning requests and redirects them to the relevant planner
    """

    def __init__(self, dds : DdsPubSub, logger, strategy_handlers: dict):
        """
        :param strategy_handlers: a dictionary of trajectory planners as strategy handlers -
        types are {TrajectoryPlanningStrategy: TrajectoryPlanner}
        """
        super().__init__(dds=dds, logger=logger)
        self.__validate_strategy_handlers(strategy_handlers)
        self.__strategy_handlers = strategy_handlers

    # TODO: implement
    def _start_impl(self):
        pass

    # TODO: implement
    def _stop_impl(self):
        pass

    # TODO: implement. call plan with the configured strategy
    def _periodic_action_impl(self):
        pass

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
        input_state = self.DDS.get_latest_sample(topic='TrajectoryPlannerSub::StateReader', timeout=1)
        return input_state

    def __read_mission_specs(self) -> Tuple[np.ndarray, np.ndarray, TrajectoryCostParams]:
        pass

    def __publish_trajectory(self, results):
        pass

    def __publish_debug(self, debug_data):
        pass


if __name__ == '__main__':
    strategy_handlers = dict()

    logger = None
    dds_object = DdsPubSub("DecisionMakingParticipantLibrary::TrajectoryPlanner",
                    '../../../../common_data/dds/generatedFiles/xml/decisionMakingMain.xml')

    trajecotry_planning_module = TrajectoryPlanningFacade(dds=dds_object, logger=logger,
                                                          strategy_handlers=strategy_handlers)
    trajecotry_planning_module.start()

    while True:
        trajecotry_planning_module.plan(strategy=TrajectoryPlanningStrategy.HIGHWAY)
        time.sleep(1)

    trajecotry_planning_module.stop()