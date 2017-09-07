from logging import Logger
from typing import Dict

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.exceptions import MsgDeserializationError
from decision_making.src.global_constants import *
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.state.state import State


class TrajectoryPlanningFacade(DmModule):
    def __init__(self, dds: DdsPubSub, logger: Logger,
                 strategy_handlers: Dict[TrajectoryPlanningStrategy, TrajectoryPlanner]):
        """
        The trajectory planning facade handles trajectory planning requests and redirects them to the relevant planner
        :param dds: communication layer (DDS) instance
        :param logger: logger
        :param strategy_handlers: a dictionary of trajectory planners as strategy handlers -
        types are {TrajectoryPlanningStrategy: TrajectoryPlanner}
        """
        super().__init__(dds=dds, logger=logger)

        self._validate_strategy_handlers()
        self._strategy_handlers = strategy_handlers

    def _start_impl(self):
        pass

    def _stop_impl(self):
        pass

    # TODO: implement. call plan with the configured strategy
    def _periodic_action_impl(self):
        """
        will execute planning with using the implementation for the desired planning-strategy provided
        :return: no return value. results are published in self.__publish_results()
        """
        try:
            state = self._get_current_state()
            params = self._get_mission_params()

            # plan a trajectory according to params (from upper DM level) and most-recent vehicle-state
            trajectory, cost, debug_results = self._strategy_handlers[params.strategy].plan(
                state, params.reference_route, params.target_state, params.time, params.cost_params)

            # TODO: should publish v_x?
            # publish results to the lower DM level
            self._publish_trajectory(TrajectoryPlanMsg(trajectory=trajectory, reference_route=params.reference_route,
                                                       current_speed=state.ego_state.v_x))

            # TODO: publish cost to behavioral layer?

            # publish visualization/debug data
            self._publish_debug(debug_results)

        except MsgDeserializationError as e:
            self.logger.debug(str(e))
            self.logger.warn("MsgDeserializationError was raised. skipping planning. " +
                             "turn on debug logging level for more details.")

    def _validate_strategy_handlers(self) -> None:
        for elem in TrajectoryPlanningStrategy.__members__.values():
            if not self._strategy_handlers.keys().__contains__(elem):
                raise KeyError('strategy_handlers does not contain a  record for ' + elem)
            if not isinstance(self._strategy_handlers[elem], TrajectoryPlanner):
                raise ValueError('strategy_handlers does not contain a TrajectoryPlanner impl. for ' + elem)

    def _get_current_state(self) -> State:
        input_state = self.dds.get_latest_sample(topic=TRAJECTORY_STATE_READER_TOPIC, timeout=1)
        self.logger.debug('Received state: %s', input_state)
        return State.deserialize(input_state)

    def _get_mission_params(self) -> TrajectoryParams:
        input_params = self.dds.get_latest_sample(topic=TRAJECTORY_PARAMS_READER_TOPIC, timeout=1)
        self.logger.debug('Received state: %s', input_params)
        return TrajectoryParams.deserialize(input_params)

    def _publish_trajectory(self, results: TrajectoryPlanMsg) -> None:
        self.dds.publish(TRAJECTORY_PUBLISH_TOPIC, results.serialize())

    def _publish_debug(self, debug_msg: TrajectoryVisualizationMsg) -> None:
        self.dds.publish(TRAJECTORY_VISUALIZATION_TOPIC, debug_msg.serialize())
