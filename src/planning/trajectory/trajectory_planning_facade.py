import traceback
from logging import Logger
from typing import Dict

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.exceptions import MsgDeserializationError, NoValidTrajectoriesFound
from decision_making.src.global_constants import TRAJECTORY_STATE_READER_TOPIC, TRAJECTORY_PARAMS_READER_TOPIC, \
    TRAJECTORY_PUBLISH_TOPIC, TRAJECTORY_VISUALIZATION_TOPIC
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.state.state import State
import time


# Define a dictionary of trajectory planners as strategy handlers
StrategyHandlersDict = Dict[TrajectoryPlanningStrategy, TrajectoryPlanner]


class TrajectoryPlanningFacade(DmModule):
    def __init__(self, dds: DdsPubSub, logger: Logger,
                 strategy_handlers: StrategyHandlersDict):
        """
        The trajectory planning facade handles trajectory planning requests and redirects them to the relevant planner
        :param dds: communication layer (DDS) instance
        :param logger: logger
        :param strategy_handlers: a dictionary of trajectory planners as strategy handlers
        """
        super().__init__(dds=dds, logger=logger)

        self._strategy_handlers = strategy_handlers
        self._validate_strategy_handlers()

    def _start_impl(self):
        pass

    def _stop_impl(self):
        pass

    # TODO: implement. call plan with the configured strategy
    def _periodic_action_impl(self):
        """
        will execute planning using the implementation for the desired planning-strategy provided
        :return: no return value. results are published in self.__publish_results()
        """
        try:
            # TODO: Read time from central time module to support also simulation & recording time.
            # TODO: If it is done only for measuring RT performance, then add documentation and change name accordingly
            start_time = time.time()

            state = self._get_current_state()
            params = self._get_mission_params()

            self.logger.debug("input: target_state: %s ", params.target_state)
            self.logger.debug("input: reference_route[0]: %s", params.reference_route[0])
            self.logger.debug("input: ego: pos: (x: %f y: %f)", state.ego_state.x, state.ego_state.y)
            self.logger.debug("input: ego: v_x: %f, v_y: %f", state.ego_state.v_x, state.ego_state.v_y)
            self.logger.info("state: %d objects detected", len(state.dynamic_objects))

            # plan a trajectory according to params (from upper DM level) and most-recent vehicle-state
            trajectory, cost, debug_results = self._strategy_handlers[params.strategy].plan(
                state, params.reference_route, params.target_state, params.time, params.cost_params)

            # TODO: should publish v_x?
            # publish results to the lower DM level (Control)
            self._publish_trajectory(TrajectoryPlanMsg(trajectory=trajectory, reference_route=params.reference_route,
                                                       current_speed=state.ego_state.v_x))

            # TODO: publish cost to behavioral layer?
            # publish visualization/debug data
            self._publish_debug(debug_results)

            self.logger.info("TrajectoryPlanningFacade._periodic_action_impl time %f", time.time()-start_time)

        except MsgDeserializationError as e:
            self.logger.warn("MsgDeserializationError was raised. skipping planning. " +
                             "turn on debug logging level for more details.")
            self.logger.debug(str(e))
        except NoValidTrajectoriesFound as e:
            # TODO - we need to handle this as an emergency.
            self.logger.warn("NoValidTrajectoriesFound was raised. skipping planning. " +
                             "turn on debug logging level for more details.")
            self.logger.debug(str(e))
        # TODO: remove this handler
        except Exception as e:
            self.logger.critical("UNHANDLED EXCEPTION in trajectory planning: %s. %s ", e, traceback.format_exc())

    def _validate_strategy_handlers(self) -> None:
        for elem in TrajectoryPlanningStrategy.__members__.values():
            if not self._strategy_handlers.keys().__contains__(elem):
                raise KeyError('strategy_handlers does not contain a  record for ' + elem)
            if not isinstance(self._strategy_handlers[elem], TrajectoryPlanner):
                raise ValueError('strategy_handlers does not contain a TrajectoryPlanner impl. for ' + elem)

    def _get_current_state(self) -> State:
        """
        Returns the last received world state.
        We assume that if no updates have been received since the last call,
        then we will output the last received state.
        :return: deserialized State
        """
        input_state = self.dds.get_latest_sample(topic=TRAJECTORY_STATE_READER_TOPIC, timeout=1)
        self.logger.debug('Received state: %s', input_state)
        return State.deserialize(input_state)

    def _get_mission_params(self) -> TrajectoryParams:
        """
        Returns the last received mission (trajectory) parameters.
        We assume that if no updates have been received since the last call,
        then we will output the last received trajectory parameters.
        :return: deserialized trajectory parameters
        """
        input_params = self.dds.get_latest_sample(topic=TRAJECTORY_PARAMS_READER_TOPIC, timeout=1)
        self.logger.debug('Received state: %s', input_params)
        return TrajectoryParams.deserialize(input_params)

    def _publish_trajectory(self, results: TrajectoryPlanMsg) -> None:
        self.dds.publish(TRAJECTORY_PUBLISH_TOPIC, results.serialize())

    def _publish_debug(self, debug_msg: TrajectoryVisualizationMsg) -> None:
        self.dds.publish(TRAJECTORY_VISUALIZATION_TOPIC, debug_msg.serialize())
