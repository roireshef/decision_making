import traceback
from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.exceptions import MsgDeserializationError, BehavioralPlanningException
from decision_making.src.global_constants import BEHAVIORAL_STATE_READER_TOPIC, \
    BEHAVIORAL_NAV_PLAN_READER_TOPIC, BEHAVIORAL_TRAJECTORY_PARAMS_PUBLISH_TOPIC, BEHAVIORAL_VISUALIZATION_TOPIC
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.policy import Policy
from decision_making.src.state.state import State
from logging import Logger
import rte.python.profiler as prof
import time


class BehavioralFacade(DmModule):
    def __init__(self, dds: DdsPubSub, logger: Logger, policy: Policy) -> None:
        """
        :param policy: decision making component
        """
        super().__init__(dds=dds, logger=logger)
        self._policy = policy
        self.logger.info("Initialized Behavioral Planner Facade.")

    # TODO: implement
    def _start_impl(self):
        pass

    # TODO: implement
    def _stop_impl(self):
        pass

    # TODO: implement
    @prof.ProfileFunction("BehavioralFacade")
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
        input_state = self.dds.get_latest_sample(topic=BEHAVIORAL_STATE_READER_TOPIC, timeout=1)
        self.logger.debug('Received State:  {}'.format(input_state))
        return State.deserialize(input_state)

    def _get_current_navigation_plan(self) -> NavigationPlanMsg:
        input_plan = self.dds.get_latest_sample(topic=BEHAVIORAL_NAV_PLAN_READER_TOPIC, timeout=1)
        self.logger.debug('Received navigation plan: %s', input_plan)
        return NavigationPlanMsg.deserialize(input_plan)

    def _publish_results(self, trajectory_parameters: TrajectoryParams) -> None:
        self.dds.publish(BEHAVIORAL_TRAJECTORY_PARAMS_PUBLISH_TOPIC, trajectory_parameters.serialize())

    def _publish_visualization(self, visualization_message: BehavioralVisualizationMsg) -> None:
        self.dds.publish(BEHAVIORAL_VISUALIZATION_TOPIC, visualization_message.serialize())
