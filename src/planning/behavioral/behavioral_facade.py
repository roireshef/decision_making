import time
from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from rte.python.logger.AV_logger import AV_Logger
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.src.messages.trajectory_parameters import TrajectoryParameters
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMessage
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.policy import Policy
from decision_making.src.planning.navigation.navigation_plan import NavigationPlan
from decision_making.src.state.enriched_state import State


class BehavioralFacade:
    def __init__(self, policy: Policy, behavioral_state: BehavioralState):
        """
        :param policy: decision making component
        :param behavioral_state: initial state of the system. Can be empty, i.e. initialized with default values.
        """
        self._policy = policy
        self._behavioral_state = behavioral_state
        self._logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
        self._logger.info("Initialized Behavioral Planner Facade.")

    def update_state_and_plan(self):
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

    # TODO : implement message passing
    def __get_current_state(self) -> State:
        pass
    
    def __get_current_navigation_plan(self) -> NavigationPlan:
        pass

    def __publish_results(self, results: TrajectoryParameters) -> None:
        pass

    def __publish_visualization(self, visualization_message: BehavioralVisualizationMessage) -> None:
        pass


if __name__ == '__main__':
    dds = DdsPubSub("DecisionMakingParticipantLibrary::BehavioralPlanner",
                    '../../../../common_data/dds/generatedFiles/xml/decisionMakingMain.xml')


    while True:
        input_state = dds.poll(topic='BehavioralPlannerSub::StateReader', timeout=1)

        print (input_state)

        time.sleep(1)