import time
import traceback
from logging import Logger

import numpy as np

from common_data.lcm.config import pubsub_topics
from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.exceptions import MsgDeserializationError, BehavioralPlanningException
from decision_making.src.global_constants import LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT, LOG_MSG_RECEIVED_STATE, \
    LOG_MSG_BEHAVIORAL_PLANNER_IMPL_TIME
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import \
    CostBasedBehavioralPlanner
from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.types import CartesianExtendedState, C_X, C_Y, C_YAW, C_V, C_A, C_K
from decision_making.src.planning.utils.localization_utils import LocalizationUtils
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State, EgoState


class BehavioralPlanningFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger, behavioral_planner: CostBasedBehavioralPlanner,
                 short_time_predictor: Predictor, last_trajectory: SamplableTrajectory = None) -> None:
        """
        :param pubsub:
        :param logger:
        :param behavioral_planner: 
        :param short_time_predictor: predictor used to align all objects in state to ego's timestamp.
        :param last_trajectory: last trajectory returned from behavioral planner.
        """
        super().__init__(pubsub=pubsub, logger=logger)
        self._planner = behavioral_planner
        self._predictor = short_time_predictor
        self.logger.info("Initialized Behavioral Planner Facade.")
        self._last_trajectory = last_trajectory

    def _start_impl(self):
        self.pubsub.subscribe(pubsub_topics.STATE_TOPIC, None)
        self.pubsub.subscribe(pubsub_topics.NAVIGATION_PLAN_TOPIC, None)

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

            # Update state: align all object to most recent timestamp, based on ego and dynamic objects timestamp
            state_aligned = self._predictor.align_objects_to_most_recent_timestamp(state=state)

            # Tests if actual localization is close enough to desired localization, and if it is, it starts planning
            # from the DESIRED localization rather than the ACTUAL one. This is due to the nature of planning with
            # Optimal Control and the fact it complies with Bellman principle of optimality.
            # THIS DOES NOT ACCOUNT FOR: yaw, velocities, accelerations, etc. Only to location.
            if LocalizationUtils.is_actual_state_close_to_expected_state(
                    state_aligned.ego_state, self._last_trajectory, self.logger, self.__class__.__name__):
                updated_state = self._get_state_with_expected_ego(state_aligned)
                self.logger.debug("BehavioralPlanningFacade ego localization was overridden to the expected-state "
                                  "according to previous plan")
            else:
                updated_state = state_aligned

            navigation_plan = self._get_current_navigation_plan()

            trajectory_params, samplable_trajectory, behavioral_visualization_message = self._planner.plan(updated_state, navigation_plan)

            self._last_trajectory = samplable_trajectory

            # Send plan to trajectory
            self._publish_results(trajectory_params)

            # Send visualization data
            self._publish_visualization(behavioral_visualization_message)

            self.logger.info("{} {}".format(LOG_MSG_BEHAVIORAL_PLANNER_IMPL_TIME, time.time() - start_time))

        except MsgDeserializationError as e:
            self.logger.warning("MsgDeserializationError was raised. skipping planning. " +
                                "turn on debug logging level for more details.")
            self.logger.debug(str(e))
        except BehavioralPlanningException as e:
            self.logger.warning(e)
        except Exception as e:
            self.logger.critical("UNHANDLED EXCEPTION IN BEHAVIORAL FACADE: %s. Trace: %s",
                                 e, traceback.format_exc())

    def _get_current_state(self) -> State:
        input_state = self.pubsub.get_latest_sample(topic=pubsub_topics.STATE_TOPIC, timeout=1)
        # TODO Move the raising of the exception to LCM code. Do the same in trajectory facade
        if input_state is None:
            raise MsgDeserializationError('LCM message queue for %s topic is empty or topic isn\'t subscribed',
                                          pubsub_topics.STATE_TOPIC)
        object_state = State.deserialize(input_state)
        self.logger.debug('{}: {}'.format(LOG_MSG_RECEIVED_STATE, object_state))
        return object_state

    def _get_current_navigation_plan(self) -> NavigationPlanMsg:
        input_plan = self.pubsub.get_latest_sample(topic=pubsub_topics.NAVIGATION_PLAN_TOPIC, timeout=1)
        object_plan = NavigationPlanMsg.deserialize(input_plan)
        self.logger.debug('Received navigation plan: %s', object_plan)
        return object_plan

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

        expected_ego_state = EgoState(
            obj_id=state.ego_state.obj_id,
            timestamp=state.ego_state.timestamp,
            x=expected_state_vec[C_X], y=expected_state_vec[C_Y], z=state.ego_state.z,
            yaw=expected_state_vec[C_YAW], size=state.ego_state.size,
            confidence=state.ego_state.confidence,
            v_x=expected_state_vec[C_V],
            v_y=0.0,  # this is ok because we don't PLAN for drift velocity
            acceleration_lon=expected_state_vec[C_A],
            curvature=expected_state_vec[C_K]
        )

        updated_state = state.clone_with(ego_state=expected_ego_state)

        return updated_state

    def _publish_results(self, trajectory_parameters: TrajectoryParams) -> None:
        self.pubsub.publish(pubsub_topics.TRAJECTORY_PARAMS_TOPIC, trajectory_parameters.serialize())
        self.logger.debug("{} {}".format(LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT, trajectory_parameters))

    def _publish_visualization(self, visualization_message: BehavioralVisualizationMsg) -> None:
        self.pubsub.publish(pubsub_topics.VISUALIZATION_TOPIC, visualization_message.serialize())

