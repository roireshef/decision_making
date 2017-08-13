import time
from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.infra.dm_module import DM_Module
from rte.python.logger.AV_logger import AV_Logger
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.src.messages.trajectory_parameters import TrajectoryParameters
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.policy import Policy, DefaultPolicy
from decision_making.src.planning.navigation.navigation_plan import NavigationPlan
from decision_making.src.state.enriched_state import State
from decision_making.src.state.enriched_state import State, EnrichedState
from rte.python.periodic_timer.periodic_timer import PeriodicTimer


class BehavioralFacade(DM_Module):
    def __init__(self, dds: DdsPubSub, logger: AV_Logger, policy: Policy = None,
                 behavioral_state: BehavioralState = None):
        """
        :param policy: decision making component
        :param behavioral_state: initial state of the system. Can be empty, i.e. initialized with default values.
        """
        super().__init__(DDS=dds, logger=logger)
        state = self.__get_current_state()
        navigation_plan = self.__get_current_navigation_plan()
        self._behavioral_state.update_behavioral_state(state, navigation_plan)
        trajectory_params, behavioral_visualization_message = self._policy.plan(behavioral_state=self._behavioral_state)
        self.__publish_results(trajectory_params)
        self.__publish_visualization(behavioral_visualization_message)

    # TODO : implement message passing
    def __get_current_state(self) -> State:
        input_state = self.DDS.get_latest_sample(topic='BehavioralPlannerSub::StateReader', timeout=1)
        self.logger.info('Received: ' + str(input_state))

        if input_state is None:
            self.logger.info('Received None State')
            return None
        else:
            self.logger.info('Received State: ' + str(input_state))
            return EnrichedState.deserialize(input_state)

    def __get_current_navigation_plan(self) -> NavigationPlan:
        pass

    def __publish_results(self, results: TrajectoryParameters) -> None:
        self.DDS.publish(topic="BehavioralPlannerPub::TrajectoryParametersWriter", data=results.serialize())

    def __publish_results(self, trajectory_parameters: TrajectoryParameters) -> None:
        self.DDS.publish(topic="BehavioralPlannerPub::TrajectoryParametersWriter",
                         data=trajectory_parameters.serialize())

    def __publish_visualization(self, visualization_message: BehavioralVisualizationMsg) -> None:
        pass
