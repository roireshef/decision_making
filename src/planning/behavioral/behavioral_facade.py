import traceback
from decision_making.src.exceptions import MsgDeserializationError, BehavioralPlanningException
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.policy import Policy
from decision_making.src.state.state import State
from logging import Logger

from common_data.src.communication.pubsub.pubsub_factory import create_pubsub
from common_data.src.communication.pubsub.pubsub import PubSub
from common_data.lcm.config import config_defs
from common_data.lcm.config import pubsub_topics

from common_data.lcm.generatedFiles.gm_lcm import LcmState
from common_data.lcm.generatedFiles.gm_lcm import LcmNavigationPlan

import time


class BehavioralFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger, policy: Policy) -> None:
        """
        :param policy: decision making component
        """
        super().__init__(pubsub=pubsub, logger=logger)
        self._policy = policy
        self.logger.info("Initialized Behavioral Planner Facade.")

    # TODO: implement
    def _start_impl(self):
        self.pubsub.subscribe(pubsub_topics.STATE_TOPIC, None, LcmState)
        self.pubsub.subscribe(pubsub_topics.NAVIGATION_PLAN_TOPIC, None, LcmNavigationPlan)

    # TODO: unsubscibe once logic is fixed in LCM
    def _stop_impl(self):
        pass

    # TODO: implement
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
            navigation_plan = self._get_current_navigation_plan()

            if state is not None:
                # Plan if the behavioral state has valid timestamp
                trajectory_params, behavioral_visualization_message = self._policy.plan(state, navigation_plan)

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
        self.logger.debug('Received State: {}'.format(input_state))
        return State.from_lcm(input_state)


    def _get_current_navigation_plan(self) -> NavigationPlanMsg:
        input_plan = self.pubsub.get_latest_sample(topic=pubsub_topics.NAVIGATION_PLAN_TOPIC, timeout=1)
        self.logger.debug('Received navigation plan: %s', input_plan)
        return NavigationPlanMsg.from_lcm(input_plan)

    def _publish_results(self, trajectory_parameters: TrajectoryParams) -> None:
        self.pubsub.publish(pubsub_topics.TRAJECTORY_PARAMS_TOPIC, trajectory_parameters.to_lcm())

    def _publish_visualization(self, visualization_message: BehavioralVisualizationMsg) -> None:
        self.pubsub.publish(pubsub_topics.VISUALIZATION_TOPIC, visualization_message.to_lcm())

