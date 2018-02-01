import traceback
from decision_making.src.exceptions import MsgDeserializationError, BehavioralPlanningException
from decision_making.src.global_constants import LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT, LOG_MSG_RECEIVED_STATE
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.policy import Policy
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State
from logging import Logger

from common_data.src.communication.pubsub.pubsub import PubSub
from common_data.lcm.config import pubsub_topics


import time


class BehavioralFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger, policy: Policy, short_time_predictor: Predictor) -> None:
        """
        :param policy: decision making component
        :param short_time_predictor: predictor used to align all objects in state to ego's timestamp.
        """
        super().__init__(pubsub=pubsub, logger=logger)
        self._policy = policy
        self._predictor = short_time_predictor
        self.logger.info("Initialized Behavioral Planner Facade.")

    def _start_impl(self):
        self.pubsub.subscribe(pubsub_topics.STATE_TOPIC, None)
        self.pubsub.subscribe(pubsub_topics.NAVIGATION_PLAN_TOPIC, None)

    # TODO: unsubscibe once logic is fixed in LCM
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

            navigation_plan = self._get_current_navigation_plan()

            if state_aligned is not None:
                # Plan if the behavioral state has valid timestamp
                trajectory_params, behavioral_visualization_message = self._policy.plan(state_aligned, navigation_plan)

                if trajectory_params is not None:
                    # Send plan to trajectory
                    self._publish_results(trajectory_params)

                    # Send visualization data
                    self._publish_visualization(behavioral_visualization_message)
                else:
                    self.logger.info("No plan was generated.")

            self.logger.info("BehavioralFacade._periodic_action_impl time %s", time.time() - start_time)

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
        object_state = State.deserialize(input_state)
        self.logger.debug('{} {}'.format(LOG_MSG_RECEIVED_STATE, object_state))
        return object_state

    def _get_current_navigation_plan(self) -> NavigationPlanMsg:
        input_plan = self.pubsub.get_latest_sample(topic=pubsub_topics.NAVIGATION_PLAN_TOPIC, timeout=1)
        object_plan = NavigationPlanMsg.deserialize(input_plan)
        self.logger.debug('Received navigation plan: %s', object_plan)
        return object_plan

    def _publish_results(self, trajectory_parameters: TrajectoryParams) -> None:
        self.pubsub.publish(pubsub_topics.TRAJECTORY_PARAMS_TOPIC, trajectory_parameters.serialize())
        self.logger.debug("{} %s", str(LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT, trajectory_parameters))

    def _publish_visualization(self, visualization_message: BehavioralVisualizationMsg) -> None:
        self.pubsub.publish(pubsub_topics.VISUALIZATION_TOPIC, visualization_message.serialize())

