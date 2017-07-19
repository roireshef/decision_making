from src.planning.behavioral.behavioral_state import BehavioralState
from src.planning.behavioral.policy import Policy
from src.planning.messages.trajectory_parameters import TrajectoryParameters
from src.state.enriched_state import State


class BehavioralFacade:
    def __init__(self, policy: Policy, behavioral_state: BehavioralState):
        """
        :param policy: decision making component
        :param behavioral_state: initial state of the system. Can be empty, i.e. initialized with default values.
        """
        self._policy = policy
        self._behavioral_state = behavioral_state

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
        trajectory_params = self._policy.plan(self._behavioral_state)
        self.__publish_results(trajectory_params)
        self.__publish_debug(trajectory_params)

    # TODO : implement message passing
    def __get_current_state(self) -> State:
        pass
    
    # TODO add navigation_plan type hint once available
    def __get_current_navigation_plan(self):
        pass

    def __publish_results(self, results: TrajectoryParameters) -> None:
        pass

    def __publish_debug(self, debug_data: TrajectoryParameters) -> None:
        pass
