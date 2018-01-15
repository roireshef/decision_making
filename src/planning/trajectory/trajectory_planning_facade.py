import traceback
from logging import Logger
from typing import Dict

from decision_making.src.exceptions import MsgDeserializationError, NoValidTrajectoriesFound
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State

from common_data.src.communication.pubsub.pubsub import PubSub
from common_data.lcm.config import pubsub_topics

import time


class TrajectoryPlanningFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger,
                 strategy_handlers: Dict[TrajectoryPlanningStrategy, TrajectoryPlanner],
                 short_time_predictor: Predictor):
        """
        The trajectory planning facade handles trajectory planning requests and redirects them to the relevant planner
        :param pubsub: communication layer (DDS/LCM/...) instance
        :param logger: logger
        :param strategy_handlers: a dictionary of trajectory planners as strategy handlers
        :param short_time_predictor: predictor used to align all objects in state to ego's timestamp.
        """
        super().__init__(pubsub=pubsub, logger=logger)

        self._predictor = short_time_predictor
        self._strategy_handlers = strategy_handlers
        self._validate_strategy_handlers()

    def _start_impl(self):
        self.pubsub.subscribe(pubsub_topics.TRAJECTORY_PARAMS_TOPIC, None)
        self.pubsub.subscribe(pubsub_topics.STATE_TOPIC, None)

    # TODO: unsubscibe once logic is fixed in LCM
    def _stop_impl(self):
        pass

    # TODO: implement. call plan with the configured strategy
    def _periodic_action_impl(self):
        """
        will execute planning using the implementation for the desired planning-strategy provided
        :return: no return value. results are published in self.__publish_results()
        """
        try:
            start_time = time.time()

            state = self._get_current_state()

            # Update state: align all object to most recent timestamp, based on ego and dynamic objects timestamp
            state_aligned = self._predictor.align_objects_to_most_recent_timestamp(state=state)

            params = self._get_mission_params()

            self.logger.debug("input: target_state: %s", params.target_state)
            self.logger.debug("input: reference_route[0]: %s", params.reference_route[0])
            self.logger.debug("input: ego: pos: (x: %f y: %f)", state_aligned.ego_state.x, state_aligned.ego_state.y)
            self.logger.debug("input: ego: v_x: %f, v_y: %f", state_aligned.ego_state.v_x, state_aligned.ego_state.v_y)
            self.logger.info("state: %d objects detected", len(state_aligned.dynamic_objects))

            # plan a trajectory according to params (from upper DM level) and most-recent vehicle-state
            trajectory, cost, debug_results = self._strategy_handlers[params.strategy].plan(
                state_aligned, params.reference_route, params.target_state, params.time, params.cost_params)

            # TODO: should publish v_x?
            # publish results to the lower DM level (Control)
            self._publish_trajectory(
                TrajectoryPlanMsg(trajectory=trajectory, current_speed=state_aligned.ego_state.v_x))

            # TODO: publish cost to behavioral layer?
            # publish visualization/debug data
            self._publish_debug(debug_results)

            self.logger.info("TrajectoryPlanningFacade._periodic_action_impl time %f", time.time() - start_time)

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
        input_state = self.pubsub.get_latest_sample(topic=pubsub_topics.STATE_TOPIC, timeout=1)
        object_state = State.deserialize(input_state)
        self.logger.debug('Received state: {}'.format(object_state))
        return object_state

    def _get_mission_params(self) -> TrajectoryParams:
        """
        Returns the last received mission (trajectory) parameters.
        We assume that if no updates have been received since the last call,
        then we will output the last received trajectory parameters.
        :return: deserialized trajectory parameters
        """
        input_params = self.pubsub.get_latest_sample(topic=pubsub_topics.TRAJECTORY_PARAMS_TOPIC, timeout=1)
        object_params = TrajectoryParams.deserialize(input_params)
        self.logger.debug('Received mission params: {}'.format(object_params))
        return object_params

    def _publish_trajectory(self, results: TrajectoryPlanMsg) -> None:
        self.pubsub.publish(pubsub_topics.TRAJECTORY_TOPIC, results.serialize())

    def _publish_debug(self, debug_msg: TrajectoryVisualizationMsg) -> None:
        self.pubsub.publish(pubsub_topics.TRAJECTORY_VISUALIZATION_TOPIC, debug_msg.serialize())
