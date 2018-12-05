import time

import numpy as np
import traceback
from logging import Logger
from typing import Dict

from common_data.interface.py.pubsub import Rte_Types_pubsub_topics as pubsub_topics
from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.exceptions import MsgDeserializationError, NoValidTrajectoriesFound
from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION, TRAJECTORY_NUM_POINTS, \
    LOG_MSG_TRAJECTORY_PLANNER_MISSION_PARAMS, LOG_MSG_RECEIVED_STATE, \
    LOG_MSG_TRAJECTORY_PLANNER_TRAJECTORY_MSG, LOG_MSG_TRAJECTORY_PLANNER_IMPL_TIME, \
    TRAJECTORY_PLANNING_NAME_FOR_METRICS, MAX_TRAJECTORY_WAYPOINTS, TRAJECTORY_WAYPOINT_SIZE, \
    LOG_MSG_SCENE_STATIC_RECEIVED
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.scene_common_messages import Header, Timestamp, MapOrigin
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlan, DataTrajectoryPlan
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import CartesianExtendedState, CartesianTrajectories, C_V, C_Y, FS_SX, FS_DX, FP_SX
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.localization_utils import LocalizationUtils
from decision_making.src.planning.utils.transformations import Transformations
from decision_making.src.prediction.action_unaware_prediction.ego_unaware_predictor import EgoUnawarePredictor
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.prediction.utils.prediction_utils import PredictionUtils
from decision_making.src.state.state import State
from decision_making.src.utils.metric_logger import MetricLogger
from decision_making.src.scene.scene_static_model import SceneStaticModel


class TrajectoryPlanningFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger,
                 strategy_handlers: Dict[TrajectoryPlanningStrategy, TrajectoryPlanner],
                 short_time_predictor: EgoUnawarePredictor,
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

        MetricLogger.init(TRAJECTORY_PLANNING_NAME_FOR_METRICS)

        self._short_time_predictor = short_time_predictor
        self._strategy_handlers = strategy_handlers
        self._validate_strategy_handlers()
        self._last_trajectory = last_trajectory

    def _start_impl(self):
        self.pubsub.subscribe(pubsub_topics.TRAJECTORY_PARAMS_LCM, None)
        self.pubsub.subscribe(pubsub_topics.STATE_LCM, None)
        self.pubsub.subscribe(pubsub_topics.SCENE_STATIC, None)

    def _stop_impl(self):
        self.pubsub.unsubscribe(pubsub_topics.TRAJECTORY_PARAMS_LCM)
        self.pubsub.unsubscribe(pubsub_topics.STATE_LCM)
        self.pubsub.unsubscribe(pubsub_topics.SCENE_STATIC)

    def _periodic_action_impl(self):
        """
        will execute planning using the implementation for the desired planning-strategy provided
        :return: no return value. results are published in self.__publish_results()
        """
        try:
            # Monitor execution time of a time-critical component (prints to logging at the end of method)
            start_time = time.time()

            state = self._get_current_state()

            scene_static = self._get_current_scene_static()
            SceneStaticModel.get_instance().set_scene_static(scene_static)

            # Update state: align all object to most recent timestamp, based on ego and dynamic objects timestamp
            most_recent_timestamp = PredictionUtils.extract_most_recent_timestamp(state)
            state_aligned = self._short_time_predictor.predict_state(state, np.array([most_recent_timestamp]))[0]

            params = self._get_mission_params()

            # Longitudinal planning horizon (Ts)
            lon_plan_horizon = params.time - state.ego_state.timestamp_in_sec

            self.logger.debug("input: target_state: %s", params.target_state)
            self.logger.debug("input: reference_route[0]: %s", params.reference_route.points[0])
            self.logger.debug("input: ego: pos: (x: %f y: %f)", state.ego_state.x, state.ego_state.y)
            self.logger.debug("input: ego: velocity: %s", state.ego_state.velocity)
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

            MetricLogger.get_logger().bind(bp_time=params.bp_time)

            # plan a trajectory according to specification from upper DM level
            samplable_trajectory, ctrajectories, costs = self._strategy_handlers[params.strategy]. \
                plan(updated_state, params.reference_route, params.target_state, lon_plan_horizon,
                     params.cost_params)

            center_vehicle_trajectory_points = samplable_trajectory.sample(
                np.linspace(start=0,
                            stop=(TRAJECTORY_NUM_POINTS - 1) * TRAJECTORY_TIME_RESOLUTION,
                            num=TRAJECTORY_NUM_POINTS) + state_aligned.ego_state.timestamp_in_sec)
            self._last_trajectory = samplable_trajectory

            vehicle_origin_trajectory_points = Transformations.transform_trajectory_between_ego_center_and_ego_origin(
                center_vehicle_trajectory_points, direction=1)

            # publish results to the lower DM level (Control)
            # TODO: put real values in tolerance and maximal velocity fields
            waypoints = np.vstack((np.hstack((vehicle_origin_trajectory_points,
                                              np.zeros(shape=[TRAJECTORY_NUM_POINTS,
                                                              TRAJECTORY_WAYPOINT_SIZE-vehicle_origin_trajectory_points.shape[1]]))),
                                  np.zeros(shape=[MAX_TRAJECTORY_WAYPOINTS-TRAJECTORY_NUM_POINTS, TRAJECTORY_WAYPOINT_SIZE])))

            timestamp = Timestamp.from_seconds(state.ego_state.timestamp_in_sec)
            map_origin = MapOrigin(e_phi_latitude=0, e_phi_longitude=0, e_l_altitude=0, s_Timestamp=timestamp)

            trajectory_msg = TrajectoryPlan(s_Header=Header(e_Cnt_SeqNum=0, s_Timestamp=timestamp,
                                                            e_Cnt_version=0),
                                            s_Data=DataTrajectoryPlan(s_Timestamp=timestamp, s_MapOrigin=map_origin,
                                                                      a_TrajectoryWaypoints=waypoints,
                                                                      e_Cnt_NumValidTrajectoryWaypoints=TRAJECTORY_NUM_POINTS))

            self._publish_trajectory(trajectory_msg)
            self.logger.debug('%s: %s', LOG_MSG_TRAJECTORY_PLANNER_TRAJECTORY_MSG, trajectory_msg)

            # publish visualization/debug data - based on short term prediction aligned state!
            debug_results = TrajectoryPlanningFacade._prepare_visualization_msg(
                state_aligned, ctrajectories, params.time - state.ego_state.timestamp_in_sec,
                self._strategy_handlers[params.strategy].predictor, params.reference_route)

            # TODO: uncomment this line when proper visualization messages integrate into the code
            # self._publish_debug(debug_results)

            self.logger.info("%s %s", LOG_MSG_TRAJECTORY_PLANNER_IMPL_TIME, time.time() - start_time)
            MetricLogger.get_logger().report()

        except MsgDeserializationError:
            self.logger.error("TrajectoryPlanningFacade: MsgDeserializationError was raised. skipping planning. %s ",
                              traceback.format_exc())

        # TODO - we need to handle this as an emergency.
        except NoValidTrajectoriesFound:
            self.logger.error("TrajectoryPlanningFacade: NoValidTrajectoriesFound was raised. skipping planning. %s",
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
        is_success, input_state = self.pubsub.get_latest_sample(topic=pubsub_topics.STATE_LCM, timeout=1)
        if input_state is None:
            raise MsgDeserializationError('Pubsub message queue for %s topic is empty or topic isn\'t subscribed',
                                          pubsub_topics.STATE_LCM)
        object_state = State.deserialize(input_state)
        self.logger.debug('%s: %s' % (LOG_MSG_RECEIVED_STATE, object_state))
        return object_state

    def _get_current_scene_static(self) -> SceneStatic:
        is_success, serialized_scene_static = self.pubsub.get_latest_sample(topic=pubsub_topics.SCENE_STATIC, timeout=1)
        # TODO Move the raising of the exception to PubSub code. Do the same in trajectory facade
        if serialized_scene_static is None:
            raise MsgDeserializationError('Pubsub message queue for %s topic is empty or topic isn\'t subscribed',
                                          pubsub_topics.SCENE_STATIC)
        scene_static = SceneStatic.deserialize(serialized_scene_static)
        self.logger.debug('%s: %f' % (LOG_MSG_SCENE_STATIC_RECEIVED, scene_static.s_Header.s_Timestamp.timestamp_in_seconds))
        return scene_static

    def _get_mission_params(self) -> TrajectoryParams:
        """
        Returns the last received mission (trajectory) parameters.
        We assume that if no updates have been received since the last call,
        then we will output the last received trajectory parameters.
        :return: deserialized trajectory parameters
        """
        is_success, input_params = self.pubsub.get_latest_sample(topic=pubsub_topics.TRAJECTORY_PARAMS_LCM, timeout=1)
        object_params = TrajectoryParams.deserialize(input_params)
        self.logger.debug('%s: %s', LOG_MSG_TRAJECTORY_PLANNER_MISSION_PARAMS, object_params)
        return object_params

    def _publish_trajectory(self, results: TrajectoryPlan) -> None:
        self.pubsub.publish(pubsub_topics.TRAJECTORY_PLAN, results.serialize())

    def _publish_debug(self, debug_msg: TrajectoryVisualizationMsg) -> None:
        self.pubsub.publish(pubsub_topics.TRAJECTORY_VISUALIZATION_LCM, debug_msg.serialize())

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
        expected_ego_state = state.ego_state.clone_from_cartesian_state(expected_state_vec,
                                                                        state.ego_state.timestamp_in_sec)

        updated_state = state.clone_with(ego_state=expected_ego_state)

        return updated_state

    @staticmethod
    def _prepare_visualization_msg(state: State, ctrajectories: CartesianTrajectories,
                                   planning_horizon: float, predictor: EgoAwarePredictor,
                                   reference_route: GeneralizedFrenetSerretFrame) -> TrajectoryVisualizationMsg:
        """
        prepares visualization message for visualization purposes
        :param state: short-term prediction aligned state
        :param ctrajectories: alternative trajectories in cartesian-frame
        :param planning_horizon: [sec] the (relative) planning-horizon used for planning
        :param predictor: predictor for the actors' predictions
        :return: trajectory visualization message
        """
        pass
