from rte.python.logger.AV_logger import AV_Logger
from src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from src.messages.trajectory_parameters import TrajectoryParamsMsg
from src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from src.planning.behavioral.behavioral_state import BehavioralState
from src.planning.behavioral.policy import Policy
from src.planning.navigation.navigation_plan import NavigationPlan
from src.state.enriched_state import State


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

    def __publish_results(self, results: TrajectoryParamsMsg) -> None:
        pass

    def __publish_visualization(self, visualization_message: BehavioralVisualizationMsg) -> None:
        pass

