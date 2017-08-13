from enum import Enum

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import TRAJECTORY_STATE_READER_TOPIC, TRAJECTORY_PUBLISH_TOPIC
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.exceptions import MsgDeserializationError
from decision_making.src.messages.trajectory_parameters import TrajectoryParameters
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner
from decision_making.src.state.enriched_state import EnrichedState
from rte.python.logger.AV_logger import AV_Logger


class TrajectoryPlanningStrategy(Enum):
    HIGHWAY = 0
    TRAFFIC_JAM = 1
    PARKING = 2


class TrajectoryPlanningFacade(DmModule):
    def __init__(self, dds: DdsPubSub, logger: AV_Logger, strategy_handlers: dict):
        """
        The trajectory planning facade handles trajectory planning requests and redirects them to the relevant planner
        :param dds: communication layer (DDS) instance
        :param logger: logger
        :param strategy_handlers: a dictionary of trajectory planners as strategy handlers -
        types are {TrajectoryPlanningStrategy: TrajectoryPlanner}
        """
        super().__init__(dds=dds, logger=logger)

        self.__validate_strategy_handlers(strategy_handlers)
        self._strategy_handlers = strategy_handlers

    # TODO: implement
    def _start_impl(self):
        pass

    # TODO: implement
    def _stop_impl(self):
        pass

    # TODO: implement. call plan with the configured strategy
    def _periodic_action_impl(self):
        """
        will execute planning with using the implementation for the desired planning-strategy provided
        :param strategy: desired planning strategy
        :return: no return value. results are published in self.__publish_results()
        """
        try:
            state = self.__read_current_state()
            params = self.__read_mission_specs()

            # plan a trajectory according to params (from upper DM level) and most-recent vehicle-state
            trajectory, cost, debug_results = self._strategy_handlers[params.strategy].plan(
                state, params.reference_route, params.target_state, params.cost_params)

            # publish results to the lower DM level
            self.__publish_trajectory(TrajectoryPlanMsg(trajectory=trajectory, reference_route=params.reference_route,
                                                        current_speed=state.ego_state.v_x))

            # TODO: publish cost to behavioral layer?

            # publish visualization/debug data
            self.__publish_debug(debug_results)

        except MsgDeserializationError as e:
            self.logger.debug(str(e))
            self.logger.warn("MsgDeserializationError was raised. skipping planning. " +
                             "turn on debug logging level for more details.")

    @staticmethod
    def __validate_strategy_handlers(handlers: dict):
        for elem in TrajectoryPlanningStrategy.__members__.values():
            if not handlers.keys().__contains__(elem):
                raise KeyError('strategy_handlers does not contain a  record for ' + elem)
            if not isinstance(handlers[elem], TrajectoryPlanner):
                raise ValueError('strategy_handlers does not contain a TrajectoryPlanner impl. for ' + elem)

    def __get_current_state(self) -> EnrichedState:
        input_state = self.dds.get_latest_sample(topic=TRAJECTORY_STATE_READER_TOPIC, timeout=1)
        self.logger.debug('Received state: %s', input_state)
        return EnrichedState.deserialize(input_state)

    def __get_mission_params(self) -> TrajectoryParameters:
        input_params = self.dds.get_latest_sample(topic=TRAJECTORY_STATE_READER_TOPIC, timeout=1)
        self.logger.debug('Received state: %s', input_params)
        return TrajectoryParameters.deserialize(input_params)

    # TODO: add type hints
    def __publish_trajectory(self, results):
        self.dds.publish(TRAJECTORY_PUBLISH_TOPIC, results.serialize())

    # TODO: implement message passing
    def __publish_debug(self, debug_data):
        pass
