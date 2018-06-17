import time
import time
import traceback
from logging import Logger
from typing import Dict

import numpy as np

from common_data.lcm.config import pubsub_topics
from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.exceptions import MsgDeserializationError, NoValidTrajectoriesFound
from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION, TRAJECTORY_NUM_POINTS, \
    VISUALIZATION_PREDICTION_RESOLUTION, MAX_NUM_POINTS_FOR_VIZ, DOWNSAMPLE_STEP_FOR_REF_ROUTE_VISUALIZATION, \
    NUM_ALTERNATIVE_TRAJECTORIES, LOG_MSG_TRAJECTORY_PLANNER_MISSION_PARAMS, LOG_MSG_RECEIVED_STATE, \
    LOG_MSG_TRAJECTORY_PLANNER_TRAJECTORY_MSG, LOG_MSG_TRAJECTORY_PLANNER_IMPL_TIME
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import CartesianExtendedState, C_V, CartesianTrajectories, CartesianPath2D
from decision_making.src.planning.utils.localization_utils import LocalizationUtils
from decision_making.src.planning.utils.transformations import Transformations
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State, NewEgoState
from mapping.src.transformations.geometry_utils import CartesianFrame


class TrajectoryPlanningFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger,
                 strategy_handlers: Dict[TrajectoryPlanningStrategy, TrajectoryPlanner],
                 short_time_predictor: Predictor,
                 last_trajectory: SamplableTrajectory = None):
        """
        The trajectory planning facade handles trajectory planning requests and redirects them to the relevant planner
        :param pubsub: communication layer (DDS/LCM/...) instance
        :param logger: logger
        :param strategy_handlers: a dictionary of trajectory planners as strategy handlers - types are
        {TrajectoryPlanningStrategy: TrajectoryPlanner}
        :param short_time_predictor: predictor used to align all objects in state to ego's timestamp.
        :param last_trajectory: a representation of the last trajectory that was planned during self._periodic_action_impl
        """
        super().__init__(pubsub=pubsub, logger=logger)

        self._short_time_predictor = short_time_predictor
        self._strategy_handlers = strategy_handlers
        self._validate_strategy_handlers()
        self._last_trajectory = last_trajectory

    def _start_impl(self):
        self.pubsub.subscribe(pubsub_topics.TRAJECTORY_PARAMS_TOPIC, None)
        self.pubsub.subscribe(pubsub_topics.STATE_TOPIC, None)

    def _stop_impl(self):
        self.pubsub.unsubscribe(pubsub_topics.TRAJECTORY_PARAMS_TOPIC)
        self.pubsub.unsubscribe(pubsub_topics.STATE_TOPIC)

    def _periodic_action_impl(self):
        """
        will execute planning using the implementation for the desired planning-strategy provided
        :return: no return value. results are published in self.__publish_results()
        """
        try:
            # Monitor execution time of a time-critical component (prints to logging at the end of method)
            start_time = time.time()

            state = self._get_current_state()

            # Update state: align all object to most recent timestamp, based on ego and dynamic objects timestamp
            state_aligned = self._short_time_predictor.align_objects_to_most_recent_timestamp(state=state)

            params = self._get_mission_params()

            # Longitudinal planning horizon (Ts)
            lon_plan_horizon = params.time - state.ego_state.timestamp_in_sec

            self.logger.debug("input: target_state: %s", params.target_state)
            self.logger.debug("input: reference_route[0]: %s", params.reference_route[0])
            self.logger.debug("input: ego: pos: (x: %f y: %f)", state.ego_state.x, state.ego_state.y)
            self.logger.debug("input: ego: v_x: %f, v_y: %f", state.ego_state.v_x, state.ego_state.v_y)
            self.logger.debug("TrajectoryPlanningFacade is required to plan with time horizon = %s", lon_plan_horizon)
            self.logger.debug("state: %d objects detected", len(state.dynamic_objects))

            # Tests if actual localization is close enough to desired localization, and if it is, it starts planning
            # from the DESIRED localization rather than the ACTUAL one. This is due to the nature of planning with
            # Optimal Control and the fact it complies with Bellman principle of optimality.
            # THIS DOES NOT ACCOUNT FOR: yaw, velocities, accelerations, etc. Only to location.
            if LocalizationUtils.is_actual_state_close_to_expected_state(
                    state_aligned.ego_state, self._last_trajectory, self.logger, self.__class__.__name__):
                updated_state = self._get_state_with_expected_ego(state_aligned)
                self.logger.debug("TrajectoryPlanningFacade ego localization was overridden to the expected-state "
                                  "according to previous plan")
            else:
                updated_state = state_aligned

            # plan a trajectory according to specification from upper DM level
            samplable_trajectory, ctrajectories, costs = self._strategy_handlers[params.strategy]. \
                plan(updated_state, params.reference_route, params.target_state, lon_plan_horizon, params.cost_params)

            center_vehicle_trajectory_points = samplable_trajectory.sample(
                np.linspace(start=0,
                            stop=(TRAJECTORY_NUM_POINTS - 1) * TRAJECTORY_TIME_RESOLUTION,
                            num=TRAJECTORY_NUM_POINTS) + state_aligned.ego_state.timestamp_in_sec)
            self._last_trajectory = samplable_trajectory

            vehicle_origin_trajectory_points = Transformations.transform_trajectory_between_ego_center_and_ego_origin(
                center_vehicle_trajectory_points, direction=1)

            # publish results to the lower DM level (Control)
            # TODO: remove ego_state.v_x from the message in version 2.0
            trajectory_msg = TrajectoryPlanMsg(timestamp=state.ego_state.timestamp,
                                               trajectory=vehicle_origin_trajectory_points,
                                               current_speed=state_aligned.ego_state.v_x)
            self._publish_trajectory(trajectory_msg)
            self.logger.debug('%s: %s', LOG_MSG_TRAJECTORY_PLANNER_TRAJECTORY_MSG, trajectory_msg)

            # publish visualization/debug data - based on short term prediction aligned state!
            debug_results = TrajectoryPlanningFacade._prepare_visualization_msg(
                state_aligned, params.reference_route, ctrajectories, costs,
                params.time - state.ego_state.timestamp_in_sec, self._strategy_handlers[params.strategy].predictor)

            self._publish_debug(debug_results)

            self.logger.info("%s %s", LOG_MSG_TRAJECTORY_PLANNER_IMPL_TIME, time.time() - start_time)

        except MsgDeserializationError:
            self.logger.error("TrajectoryPlanningFacade: MsgDeserializationError was raised. skipping planning. %s ",
                             traceback.format_exc())

        # TODO - we need to handle this as an emergency.
        except NoValidTrajectoriesFound:
            self.logger.error("TrajectoryPlanningFacade: MsgDeserializationError was raised. skipping planning. %s",
                             traceback.format_exc())

        except Exception:
            self.logger.critical("TrajectoryPlanningFacade: UNHANDLED EXCEPTION in trajectory planning: %s",
                                 traceback.format_exc())

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
        if input_state is None:
            raise MsgDeserializationError('LCM message queue for %s topic is empty or topic isn\'t subscribed',
                                          pubsub_topics.STATE_TOPIC)
        object_state = State.deserialize(input_state)
        self.logger.debug('%s: %s' % (LOG_MSG_RECEIVED_STATE, object_state))
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
        self.logger.debug('%s: %s', LOG_MSG_TRAJECTORY_PLANNER_MISSION_PARAMS, object_params)
        return object_params

    def _publish_trajectory(self, results: TrajectoryPlanMsg) -> None:
        self.pubsub.publish(pubsub_topics.TRAJECTORY_TOPIC, results.serialize())

    def _publish_debug(self, debug_msg: TrajectoryVisualizationMsg) -> None:
        self.pubsub.publish(pubsub_topics.TRAJECTORY_VISUALIZATION_TOPIC, debug_msg.serialize())

    def _get_state_with_expected_ego(self, state: State) -> State:
        """
        takes a state and overrides its ego vehicle's localization to be the localization expected at the state's
        timestamp according to the last trajectory cached in the facade's self._last_trajectory.
        Note: lateral velocity is zeroed since we don't plan for drifts and lateral components are being reflected in
        yaw and curvature.
        :param state: the state to process
        :return: a new state object with a new ego-vehicle localization
        """
        current_time = state.ego_state.timestamp_in_sec
        expected_state_vec: CartesianExtendedState = self._last_trajectory.sample(np.array([current_time]))[0]

        expected_ego_state = NewEgoState.create_from_cartesian_state(
            obj_id=state.ego_state.obj_id,
            timestamp=state.ego_state.timestamp,
            cartesian_state=expected_state_vec,
            size=state.ego_state.size,
            confidence=state.ego_state.confidence)

        updated_state = state.clone_with(ego_state=expected_ego_state)

        return updated_state

    @staticmethod
    def _prepare_visualization_msg(state: State, reference_route: CartesianPath2D,
                                   ctrajectories: CartesianTrajectories, costs: np.ndarray,
                                   planning_horizon: float, predictor: Predictor):
        """
        prepares visualization message for visualization purposes
        :param state: short-term prediction aligned state
        :param reference_route: the reference route got from BP
        :param ctrajectories: alternative trajectories in cartesian-frame
        :param costs: costs computed for each alternative trajectory
        :param planning_horizon: [sec] the (relative) planning-horizon used for planning
        :return:
        """
        # this assumes the state is already aligned by short time prediction
        most_recent_timestamp = state.ego_state.timestamp_in_sec
        prediction_timestamps = np.arange(most_recent_timestamp, most_recent_timestamp + planning_horizon,
                                          VISUALIZATION_PREDICTION_RESOLUTION, float)

        # TODO: move this to visualizer!
        # Curently we are predicting the state at ego's timestamp and at the end of the traj execution time.
        # predicted_states[0] is the current state
        # predicted_states[1] is the predicted state in the end of the execution of traj.
        predicted_states = predictor.predict_state(state=state, prediction_timestamps=prediction_timestamps)

        _, downsampled_reference_route, _ = CartesianFrame.resample_curve(reference_route,
                                                                          step_size=DOWNSAMPLE_STEP_FOR_REF_ROUTE_VISUALIZATION)

        # slice alternative trajectories by skipping indices - for visualization
        alternative_ids_skip_range = range(0, len(ctrajectories),
                                           max(int(len(ctrajectories) / NUM_ALTERNATIVE_TRAJECTORIES), 1))

        # slice alternative trajectories by skipping indices - for visualization
        sliced_ctrajectories = ctrajectories[alternative_ids_skip_range]
        sliced_costs = costs[alternative_ids_skip_range]

        return TrajectoryVisualizationMsg(downsampled_reference_route,
                                          sliced_ctrajectories[:, :min(MAX_NUM_POINTS_FOR_VIZ, ctrajectories.shape[1]),
                                          :C_V],
                                          sliced_costs,
                                          predicted_states[0],
                                          predicted_states[1:],
                                          planning_horizon)
