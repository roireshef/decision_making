import copy
import time
import traceback
from logging import Logger
from typing import Dict

import numpy as np

from common_data.lcm.config import pubsub_topics
from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.exceptions import MsgDeserializationError, NoValidTrajectoriesFound
from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION, TRAJECTORY_NUM_POINTS, \
    NEGLIGIBLE_DISPOSITION_LON, NEGLIGIBLE_DISPOSITION_LAT, DEFAULT_OBJECT_Z_VALUE, VISUALIZATION_PREDICTION_RESOLUTION, \
    DEFAULT_CURVATURE, DEFAULT_ACCELERATION
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import C_Y, C_X, C_YAW, FP_SX, FP_DX, FrenetPoint, \
    CartesianExtendedState, C_V, C_A, CartesianTrajectories, CartesianPath2D, C_K
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State, EgoState
from mapping.src.transformations.geometry_utils import CartesianFrame


class TrajectoryPlanningFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger,
                 strategy_handlers: Dict[TrajectoryPlanningStrategy, TrajectoryPlanner],
                 last_trajectory: SamplableTrajectory = None):
        """
        The trajectory planning facade handles trajectory planning requests and redirects them to the relevant planner
        :param pubsub: communication layer (DDS/LCM/...) instance
        :param logger: logger
        :param strategy_handlers: a dictionary of trajectory planners as strategy handlers -
        types are {TrajectoryPlanningStrategy: TrajectoryPlanner}
        """
        super().__init__(pubsub=pubsub, logger=logger)

        self._strategy_handlers = strategy_handlers
        self._validate_strategy_handlers()
        self._last_trajectory = last_trajectory

    def _start_impl(self):
        self.pubsub.subscribe(pubsub_topics.TRAJECTORY_PARAMS_TOPIC, None)
        self.pubsub.subscribe(pubsub_topics.STATE_TOPIC, None)

    # TODO: unsubscibe once logic is fixed in LCM
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

            self.logger.debug("input: target_state: %s", params.target_state)
            self.logger.debug("input: reference_route[0]: %s", params.reference_route[0])
            self.logger.debug("input: ego: pos: (x: %f y: %f)", state.ego_state.x, state.ego_state.y)
            self.logger.debug("input: ego: v_x: %f, v_y: %f", state.ego_state.v_x, state.ego_state.v_y)
            self.logger.debug("TrajectoryPlanningFacade is required to plan with time horizon = %s",
                              params.time - state.ego_state.timestamp_in_sec)
            self.logger.info("state: %d objects detected", len(state.dynamic_objects))

            # TODO: this currently applies to location only (not yaw, velocities, accelerations, etc.)
            if self._is_actual_state_close_to_expected_state(state.ego_state):
                updated_state = self._get_state_with_expected_ego(state)
                self.logger.info("TrajectoryPlanningFacade ego localization was overriden to the expected-state "
                                 "according to previous plan")
            else:
                updated_state = state

            # TODO: currently adding default curvature and acceleration values
            extended_target_state = np.append(params.target_state, [DEFAULT_ACCELERATION, DEFAULT_CURVATURE])

            # plan a trajectory according to params (from upper DM level) and most-recent vehicle-state
            samplable_trajectory, ctrajectories, costs = self._strategy_handlers[params.strategy]. \
                plan(updated_state, params.reference_route, extended_target_state, params.time, params.cost_params)

            # TODO: validate that sampling is consistent with controller!
            trajectory_points = samplable_trajectory.sample(
                np.linspace(start=TRAJECTORY_TIME_RESOLUTION,
                            stop=TRAJECTORY_NUM_POINTS * TRAJECTORY_TIME_RESOLUTION,
                            num=TRAJECTORY_NUM_POINTS) + state.ego_state.timestamp_in_sec)

            # TODO: should publish v_x?
            # publish results to the lower DM level (Control)
            self._publish_trajectory(TrajectoryPlanMsg(trajectory=trajectory_points, current_speed=state.ego_state.v_x))
            self._last_trajectory = samplable_trajectory

            # TODO: publish cost to behavioral layer?
            # publish visualization/debug data - based on original state!
            debug_results = TrajectoryPlanningFacade._prepare_visualization_msg(
                state, params.reference_route, ctrajectories, costs, params.time - state.ego_state.timestamp_in_sec,
                self._strategy_handlers[params.strategy].predictor)

            # TODO: DEBUG ONLY
            # debug_results.state = updated_state
            self._publish_debug(debug_results)

            self.logger.info("TrajectoryPlanningFacade._periodic_action_impl time %f", time.time() - start_time)

        except MsgDeserializationError as e:
            self.logger.warn("TrajectoryPlanningFacade: MsgDeserializationError was raised. skipping planning. %s %s",
                             e, traceback.format_exc())
        except NoValidTrajectoriesFound as e:
            # TODO - we need to handle this as an emergency.
            self.logger.warn("TrajectoryPlanningFacade: NoValidTrajectoriesFound was raised. skipping planning. %s %s",
                             e, traceback.format_exc())
        # TODO: remove this handler
        except Exception as e:
            self.logger.critical("TrajectoryPlanningFacade: UNHANDLED EXCEPTION in trajectory planning: %s. %s ",
                                 e, traceback.format_exc())

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
        self.logger.debug('TrajectoryPlanningFacade current localization: [%s, %s, %s, %s]' %
                          (object_state.ego_state.x, object_state.ego_state.y, object_state.ego_state.yaw,
                           object_state.ego_state.v_x))
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

    def _is_actual_state_close_to_expected_state(self, current_ego_state: EgoState) -> bool:
        """
        checks if the actual ego state at time t[current] is close (currently in terms of Euclidean distance of position
        [x,y] only) to the desired state at t[current] according to the plan of the last trajectory.
        :param current_ego_state: the current EgoState object representing the actual state of ego vehicle
        :return: true if actual state is closer than NEGLIGIBLE_LOCATION_DIFF to the planned state. false otherwise
        """
        current_time = current_ego_state.timestamp_in_sec
        if self._last_trajectory is None or current_time > self._last_trajectory.max_sample_time:
            return False

        self.logger.debug("TrajectoryPlanningFacade time-difference from last planned trajectory is %s",
                          current_time - self._last_trajectory.timestamp)

        current_expected_state: CartesianExtendedState = self._last_trajectory.sample(np.array([current_time]))[0]
        current_actual_location = np.array([current_ego_state.x, current_ego_state.y, DEFAULT_OBJECT_Z_VALUE])

        errors_in_expected_frame, _ = CartesianFrame.convert_global_to_relative_frame(
            global_pos=current_actual_location,
            global_yaw=0.0,
            frame_position=np.append(current_expected_state[[C_X, C_Y]], [DEFAULT_OBJECT_Z_VALUE]),
            frame_orientation=current_expected_state[C_YAW]
        )

        distances_in_expected_frame: FrenetPoint = np.abs(errors_in_expected_frame)

        self.logger.info("TrajectoryPlanningFacade localization stats: {desired_localization: %s, actual_localization: "
                         "%s, desired_velocity: %s, actual_velocity: %s, lon_lat_errors: %s, velocity_error: %s}" %
                         (current_expected_state, current_actual_location,
                          current_expected_state[C_V], current_ego_state.v_x,
                          distances_in_expected_frame, current_ego_state.v_x - current_expected_state[C_V]))

        return distances_in_expected_frame[FP_SX] <= NEGLIGIBLE_DISPOSITION_LON and \
               distances_in_expected_frame[FP_DX] <= NEGLIGIBLE_DISPOSITION_LAT

    def _get_state_with_expected_ego(self, state: State):
        """
        takes a state and overrides its ego vehicle's localization to be the localization expected at the state's
        timestamp according to the last trajectory cached in the facade's self._last_trajectory
        :param state: the state to process
        :return: a new state object with a new ego-vehicle localization
        """
        current_time = state.ego_state.timestamp_in_sec
        expected_state: CartesianExtendedState = self._last_trajectory.sample(np.array([current_time]))[0]

        updated_state = copy.deepcopy(state)
        updated_state.ego_state = EgoState(obj_id=state.ego_state.obj_id,
                                           timestamp=state.ego_state.timestamp,
                                           x=expected_state[C_X], y=expected_state[C_Y], z=state.ego_state.z,
                                           yaw=expected_state[C_YAW], size=state.ego_state.size,
                                           confidence=state.ego_state.confidence,
                                           v_x=expected_state[C_V],
                                           v_y=0.0,  # this is ok because we don't PLAN for drift velocity
                                           acceleration_lon=expected_state[C_A],
                                           omega_yaw=state.ego_state.omega_yaw,  # TODO: fill this properly
                                           steering_angle=np.arctan(state.ego_state.size.length * expected_state[C_K]),
                                           )

        return updated_state

    @staticmethod
    def _prepare_visualization_msg(state: State, reference_route: CartesianPath2D,
                                   ctrajectories: CartesianTrajectories, costs: np.ndarray,
                                   planning_horizon: float, predictor: Predictor):
        """
        prepares visualization message for visualization purposes
        :param state: the original (raw, unedited) state got by this facade
        :param reference_route: the reference route got from BP
        :param ctrajectories: alternative trajectories in cartesian-frame
        :param costs: costs computed for each alternative trajectory
        :param planning_horizon: [sec] the (relative) planning-horizon used for planning
        :return:
        """
        # TODO: remove this section and solve timestamps-sync in StateModule
        objects_timestamp_in_sec = [dyn_obj.timestamp_in_sec for dyn_obj in state.dynamic_objects]
        objects_timestamp_in_sec.append(state.ego_state.timestamp_in_sec)
        most_recent_timestamp = np.max(objects_timestamp_in_sec)

        prediction_timestamps = np.arange(most_recent_timestamp, state.ego_state.timestamp_in_sec + planning_horizon,
                                          VISUALIZATION_PREDICTION_RESOLUTION, float)

        # Curently we are predicting the state at ego's timestamp and at the end of the traj execution time.
        # predicted_states[0] is the current state
        # predicted_states[1] is the predicted state in the end of the execution of traj.
        # TODO: move this to visualizer!
        predicted_states = predictor.predict_state(state=state, prediction_timestamps=prediction_timestamps)

        return TrajectoryVisualizationMsg(reference_route,
                                          ctrajectories[:, :, :C_V],
                                          costs,
                                          predicted_states[0],
                                          predicted_states[1:],
                                          planning_horizon)
