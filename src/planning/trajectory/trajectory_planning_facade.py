import traceback
from logging import Logger
from typing import Dict

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.exceptions import MsgDeserializationError, NoValidTrajectoriesFound
from decision_making.src.global_constants import TRAJECTORY_STATE_READER_TOPIC, TRAJECTORY_PARAMS_READER_TOPIC, \
    TRAJECTORY_PUBLISH_TOPIC, TRAJECTORY_VISUALIZATION_TOPIC, TRAJECTORY_TIME_RESOLUTION, TRAJECTORY_NUM_POINTS, \
    NEGLIGIBLE_DISPOSITION_LON, NEGLIGIBLE_DISPOSITION_LAT
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.state.state import State
import time
import numpy as np
import copy
from mapping.src.transformations.geometry_utils import CartesianFrame


class TrajectoryPlanningFacade(DmModule):
    def __init__(self, dds: DdsPubSub, logger: Logger,
                 strategy_handlers: Dict[TrajectoryPlanningStrategy, TrajectoryPlanner],
                 last_trajectory: SamplableTrajectory=None):
        """
        The trajectory planning facade handles trajectory planning requests and redirects them to the relevant planner
        :param dds: communication layer (DDS) instance
        :param logger: logger
        :param strategy_handlers: a dictionary of trajectory planners as strategy handlers -
        types are {TrajectoryPlanningStrategy: TrajectoryPlanner}
        """
        super().__init__(dds=dds, logger=logger)

        self._strategy_handlers = strategy_handlers
        self._validate_strategy_handlers()
        self._last_trajectory = last_trajectory

    def _start_impl(self):
        pass

    def _stop_impl(self):
        pass

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

            # TODO: this currently applies to location only (not yaw, velocities, accelerations, etc.)
            if self._is_actual_state_close_to_expected_state(state.ego_state):
                updated_state = self._get_state_with_expected_ego(state)
            else:
                updated_state = state

            # plan a trajectory according to params (from upper DM level) and most-recent vehicle-state
            samplable_trajectory, cost, debug_results = self._strategy_handlers[params.strategy].\
                plan(updated_state, params.reference_route, params.target_state, params.time, params.cost_params)

            # TODO: validate that sampling is consistent with controller!
            trajectory_points = samplable_trajectory.sample(
                np.linspace(start=TRAJECTORY_TIME_RESOLUTION,
                            stop=TRAJECTORY_NUM_POINTS*TRAJECTORY_TIME_RESOLUTION,
                            num=TRAJECTORY_NUM_POINTS))

            # TODO: should publish v_x?
            # publish results to the lower DM level (Control)
            self._publish_trajectory(TrajectoryPlanMsg(trajectory=trajectory_points, current_speed=state.ego_state.v_x))

            self._last_trajectory = samplable_trajectory

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

    def _is_actual_state_close_to_expected_state(self, current_ego_state: EgoState) -> bool:
        """
        checks if the actual ego state at time t[current] is close (currently in terms of Euclidean distance of position
        [x,y] only) to the desired state at t[current] according to the plan of the last trajectory.
        :param current_ego_state: the current EgoState object representing the actual state of ego vehicle
        :return: true if actual state is closer than NEGLIGIBLE_LOCATION_DIFF to the planned state. false otherwise
        """
        if self._last_trajectory is None:
            return False

        time_diff = current_ego_state.timestamp - self._last_trajectory.timestamp
        current_expected_location = self._last_trajectory.sample(np.array([time_diff]))
        current_actual_location = np.array([current_ego_state.x, current_ego_state.y])

        # TODO: sides should be switched (this should be in terms of expected coordinate frame)
        errors_in_ego_frame, _ = CartesianFrame.convert_global_to_relative_frame(
            global_pos=np.append(current_expected_location, [0.0]),
            global_yaw=0.0, # irrelevant since we don't care about relative yaw
            frame_position=current_actual_location,
            frame_orientation=current_ego_state.yaw
        )

        distances_in_ego_frame = np.abs(errors_in_ego_frame)

        # TODO: change 0,1 indices to X and Y column constants (when merged with other branches in v1.5.3)
        return distances_in_ego_frame[0] <= NEGLIGIBLE_DISPOSITION_LON and \
               distances_in_ego_frame[1] <= NEGLIGIBLE_DISPOSITION_LAT

    def _get_state_with_expected_ego(self, state: State):
        time_diff = state.ego_state.timestamp - self._last_trajectory.timestamp
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
