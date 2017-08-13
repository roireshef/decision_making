import time

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, BEHAVIORAL_STATE_READER_TOPIC, \
    BEHAVIORAL_NAV_PLAN_READER_TOPIC, BEHAVIORAL_TRAJECTORY_PARAMS_PUBLISH_TOPIC
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParameters
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.policy import Policy, DefaultPolicy
from decision_making.src.state.enriched_state import EnrichedState
from rte.python.logger.AV_logger import AV_Logger



class BehavioralFacade(DmModule):
    def __init__(self, dds: DdsPubSub, logger: AV_Logger, policy: Policy = None,
                 behavioral_state: BehavioralState = None):
        """
        :param policy: decision making component
        :param behavioral_state: initial state of the system. Can be empty, i.e. initialized with default values.
        """
        super().__init__(dds=dds, logger=logger)
        self._policy = policy
        self._behavioral_state = behavioral_state

    # TODO: implement
    def _start_impl(self):
        pass

    # TODO: implement
    def _stop_impl(self):
        pass

    # TODO: implement
    def _periodic_action_impl(self):
        """
        The main function of the behavioral planner. It read the most up-to-date enriched state and navigation plan,
         processes them into the behavioral state, and then performs behavioral planning. The results are then published
          to the trajectory planner and as debug information to the visualizer.
        :return: void
        """
        state = self.__get_current_state()
        navigation_plan = self.__get_current_navigation_plan()
        self._behavioral_state.update_behavioral_state(state, navigation_plan)
        trajectory_params, behavioral_visualization_message = self._policy.plan(behavioral_state=self._behavioral_state)
        self.__publish_results(trajectory_params)
        self.__publish_visualization(behavioral_visualization_message)

    def __get_current_state(self) -> EnrichedState:
        input_state = self.dds.get_latest_sample(topic=BEHAVIORAL_STATE_READER_TOPIC, timeout=1)
        if input_state is None:
            self.logger.error('Received None State')
            return None
        else:
            self.logger.debug('Received State: ' + str(input_state))
            return EnrichedState.deserialize(input_state)

    def __get_current_navigation_plan(self) -> NavigationPlanMsg:
        input_plan = self.dds.get_latest_sample(topic=BEHAVIORAL_NAV_PLAN_READER_TOPIC, timeout=1)
        self.logger.debug('Received navigation plan: %s', input_plan)
        return input_plan

    def __publish_results(self, trajectory_parameters: TrajectoryParameters) -> None:
        self.dds.publish(BEHAVIORAL_TRAJECTORY_PARAMS_PUBLISH_TOPIC, trajectory_parameters.serialize())

    # TODO: add to DDS XML first
    def __publish_visualization(self, visualization_message: BehavioralVisualizationMsg) -> None:
        pass
