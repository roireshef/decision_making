import traceback
from logging import Logger
from typing import Dict

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.exceptions import MsgDeserializationError, NoValidTrajectoriesFound
from decision_making.src.global_constants import *
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.utils.columns import EGO_X, EGO_Y
from decision_making.src.state.state import State, EgoState
import time
import numpy as np
import copy


class TrajectoryPlanningFacade(DmModule):
    def __init__(self, dds: DdsPubSub, logger: Logger,
                 strategy_handlers: Dict[TrajectoryPlanningStrategy, TrajectoryPlanner],
                 last_trajectory: SamplableTrajectory=None,
                 last_state_time: SamplableTrajectory=None):
        """
        The trajectory planning facade handles trajectory planning requests and redirects them to the relevant planner
        :param dds: communication layer (DDS) instance
        :param logger: logger
        :param strategy_handlers: a dictionary of trajectory planners as strategy handlers -
        types are {TrajectoryPlanningStrategy: TrajectoryPlanner}
        :param last_trajectory: last chosen trajectory's representation
        :param last_state_time: last iteration's processed state timestamp
        """
        super().__init__(dds=dds, logger=logger)

        self._strategy_handlers = strategy_handlers
        self._validate_strategy_handlers()
        self._last_trajectory = last_trajectory
        self._last_state_time = last_state_time

    def _start_impl(self):
        pass

    def _stop_impl(self):
        pass

    def _periodic_action_impl(self):
        """
        will execute planning with using the implementation for the desired planning-strategy provided
        :return: no return value. results are published in self.__publish_results()
        """
        try:
            start_time = time.time()

            state = self._get_current_state()
            params = self._get_mission_params()

            self.logger.debug("input: target_state:{}".format(params.target_state))
            self.logger.debug("input: reference_route[0]:{}".format(params.reference_route[0]))
            self.logger.debug("input: ego: pos: (x: {} y: {})".format(state.ego_state.x, state.ego_state.y))
            self.logger.debug("input: ego: v_x: {}, v_y: {}".format(state.ego_state.v_x, state.ego_state.v_y))
            self.logger.info("state: {} objects detected".format(len(state.dynamic_objects)))

            # TODO: this currently applies to location only (not yaw, velocities, accelerations, etc.)
            if self._is_actual_state_close_to_previous_plan(state):
                updated_state = self._get_modified_state(state)
            else:
                updated_state = state

            # plan a trajectory according to params (from upper DM level) and most-recent vehicle-state
            trajectory_points, cost, samplable_trajectory, debug_results = self._strategy_handlers[params.strategy].\
                plan(updated_state, params.reference_route, params.target_state, params.time, params.cost_params)

            # TODO: should publish v_x?
            # publish results to the lower DM level
            self._publish_trajectory(TrajectoryPlanMsg(trajectory=trajectory_points,
                                                       reference_route=params.reference_route,
                                                       current_speed=state.ego_state.v_x))

            self._last_state_time = state.ego_state.timestamp

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
        except Exception as e:
            self.logger.critical("UNHANDLED EXCEPTION in trajectory planning: %s. %s ", e, traceback.format_exc())

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

    def _is_actual_state_close_to_previous_plan(self, current_state: State):
        if self._last_state_time is None or self._last_trajectory is None:
            return False

        time_diff = current_state.ego_state.timestamp - self._last_state_time
        current_expected_location = self._last_trajectory.sample(np.array([time_diff]))
        current_actual_location = np.array([current_state.ego_state.x, current_state.ego_state.y])
        euclidean_distance = np.linalg.norm(np.subtract(current_expected_location, current_actual_location))

        return euclidean_distance <= NEGLIGIBLE_LOCATION_DIFF

    def _get_modified_state(self, state: State):
        time_diff = state.ego_state.timestamp - self._last_state_time
        expected_state = self._last_trajectory.sample(np.array([time_diff]))[0]

        updated_state = copy.deepcopy(state)
        updated_state.ego_state = EgoState(obj_id=state.ego_state.obj_id,
                                           timestamp=state.ego_state.timestamp,
                                           x=expected_state[EGO_X], y=expected_state[EGO_Y], z=state.ego_state.z,
                                           yaw=state.ego_state.yaw, size=state.ego_state.size,
                                           confidence=state.ego_state.confidence,
                                           v_x=state.ego_state.v_x, v_y=state.ego_state.v_y,
                                           acceleration_lon=state.ego_state.acceleration_lon,
                                           omega_yaw=state.ego_state.omega_yaw,
                                           steering_angle=state.ego_state.steering_angle,
                                           road_localization=None
                                           )

        return updated_state
