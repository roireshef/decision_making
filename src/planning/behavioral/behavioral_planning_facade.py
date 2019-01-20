import time
import traceback
from logging import Logger

import numpy as np

from common_data.interface.Rte_Types.python import Rte_Types_pubsub as pubsub_topics
from decision_making.src.exceptions import MsgDeserializationError, BehavioralPlanningException
from decision_making.src.global_constants import LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT, LOG_MSG_RECEIVED_STATE, \
    LOG_MSG_BEHAVIORAL_PLANNER_IMPL_TIME, BEHAVIORAL_PLANNING_NAME_FOR_METRICS, LOG_MSG_SCENE_STATIC_RECEIVED
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import \
    CostBasedBehavioralPlanner
from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.types import CartesianExtendedState
from decision_making.src.planning.utils.localization_utils import LocalizationUtils
from decision_making.src.state.state import State
from decision_making.src.utils.metric_logger import MetricLogger
from decision_making.src.scene.scene_static_model import SceneStaticModel


class BehavioralPlanningFacade(DmModule):
    def __init__(self, logger: Logger, behavioral_planner: CostBasedBehavioralPlanner,
                 last_trajectory: SamplableTrajectory = None) -> None:
        """
        :param logger:
        :param behavioral_planner: 
        :param last_trajectory: last trajectory returned from behavioral planner.
        """
        super().__init__(logger=logger)
        self._planner = behavioral_planner
        self.logger.info("Initialized Behavioral Planner Facade.")
        self._last_trajectory = last_trajectory
        MetricLogger.init(BEHAVIORAL_PLANNING_NAME_FOR_METRICS)

    def _start_impl(self):
        pubsub_topics.UC_SYSTEM_STATE_LCM.register_cb(None)
        pubsub_topics.UC_SYSTEM_NAVIGATION_PLAN_LCM.register_cb(None)
        pubsub_topics.UC_SYSTEM_SCENE_STATIC.register_cb(None)

    # TODO: unsubscribe once logic is fixed in LCM
    def _stop_impl(self):
        pass

    def _periodic_action_impl(self) -> None:
        """
        The main function of the behavioral planner. It read the most up-to-date state and navigation plan,
         processes them into the behavioral state, and then performs behavioral planning. The results are then published
          to the trajectory planner and as debug information to the visualizer.
        :return: void
        """

        try:
            start_time = time.time()
            state = self._get_current_state()

            scene_static = self._get_current_scene_static()
            SceneStaticModel.get_instance().set_scene_static(scene_static)

            # Tests if actual localization is close enough to desired localization, and if it is, it starts planning
            # from the DESIRED localization rather than the ACTUAL one. This is due to the nature of planning with
            # Optimal Control and the fact it complies with Bellman principle of optimality.
            # THIS DOES NOT ACCOUNT FOR: yaw, velocities, accelerations, etc. Only to location.
            if LocalizationUtils.is_actual_state_close_to_expected_state(
                    state.ego_state, self._last_trajectory, self.logger, self.__class__.__name__):
                updated_state = self._get_state_with_expected_ego(state)
                self.logger.debug("BehavioralPlanningFacade ego localization was overridden to the expected-state "
                                  "according to previous plan")
            else:
                updated_state = state

            navigation_plan = self._get_current_navigation_plan()

            trajectory_params, samplable_trajectory, behavioral_visualization_message = self._planner.plan(updated_state, navigation_plan)

            self._last_trajectory = samplable_trajectory

            # Send plan to trajectory
            self._publish_results(trajectory_params)

            # Send visualization data
            self._publish_visualization(behavioral_visualization_message)

            self.logger.info("{} {}".format(LOG_MSG_BEHAVIORAL_PLANNER_IMPL_TIME, time.time() - start_time))

            MetricLogger.get_logger().report()

        except MsgDeserializationError as e:
            self.logger.warning("MsgDeserializationError was raised. skipping planning. " +
                                "turn on debug logging level for more details.%s", traceback.format_exc())
            self.logger.debug(str(e))
        except BehavioralPlanningException as e:
            self.logger.warning(e)
        except Exception as e:
            self.logger.critical("UNHANDLED EXCEPTION IN BEHAVIORAL FACADE: %s. Trace: %s",
                                 e, traceback.format_exc())

    def _get_current_state(self) -> State:
        is_success, input_state = self._get_latest_sample(topic=pubsub_topics.UC_SYSTEM_STATE_LCM)
        # TODO Move the raising of the exception to LCM code. Do the same in trajectory facade
        if input_state is None:
            raise MsgDeserializationError('Pubsub message queue for %s topic is empty or topic isn\'t subscribed',
                                          pubsub_topics.UC_SYSTEM_STATE_LCM)
        object_state = State.deserialize(input_state)
        self.logger.debug('{}: {}'.format(LOG_MSG_RECEIVED_STATE, object_state))
        return object_state

    def _get_current_navigation_plan(self) -> NavigationPlanMsg:
        is_success, input_plan = self._get_latest_sample(topic=pubsub_topics.UC_SYSTEM_NAVIGATION_PLAN_LCM)
        object_plan = NavigationPlanMsg.deserialize(input_plan)
        self.logger.debug('Received navigation plan: %s', object_plan)
        return object_plan

    def _get_current_scene_static(self) -> SceneStatic:
        is_success, serialized_scene_static = self._get_latest_sample(topic=pubsub_topics.UC_SYSTEM_SCENE_STATIC)
        # TODO Move the raising of the exception to LCM code. Do the same in trajectory facade
        if serialized_scene_static is None:
            raise MsgDeserializationError('Pubsub message queue for %s topic is empty or topic isn\'t subscribed',
                                          pubsub_topics.UC_SYSTEM_SCENE_STATIC)
        scene_static = SceneStatic.deserialize(serialized_scene_static)
        self.logger.debug('%s: %f' % (LOG_MSG_SCENE_STATIC_RECEIVED, scene_static.s_Header.s_Timestamp.timestamp_in_seconds))
        return scene_static

    def _get_latest_sample(self, topic):
        is_success = True
        while is_success is True:
            is_success, msg = topic.recv_blocking(0)
        if is_success is True and msg is not None:
            self._last_msg[topic] = msg
        return True, self._last_msg[topic]

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
        expected_ego_state = state.ego_state.clone_from_cartesian_state(expected_state_vec, state.ego_state.timestamp_in_sec)

        updated_state = state.clone_with(ego_state=expected_ego_state)

        return updated_state

    def _publish_results(self, trajectory_parameters: TrajectoryParams) -> None:
        pubsub_topics.UC_SYSTEM_TRAJECTORY_PARAMS_LCM.send(trajectory_parameters.serialize())
        self.logger.debug("{} {}".format(LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT, trajectory_parameters))

    def _publish_visualization(self, visualization_message: BehavioralVisualizationMsg) -> None:
        pubsub_topics.UC_SYSTEM_VISUALIZATION_LCM.send(visualization_message.serialize())

    @property
    def planner(self):
        return self._planner

    @property
    def predictor(self):
        return self._predictor
