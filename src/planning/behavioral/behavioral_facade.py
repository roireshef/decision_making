from src.planning.behavioral.behavioral_state import BehavioralState
from src.planning.messages.trajectory_parameters import TrajectoryParameters


class BehavioralFacade:
    def __init__(self, policy, behavioral_state: BehavioralState):
        self._policy = policy
        self._behavioral_state = behavioral_state

    def update_state_and_plan(self):
        state = self.__get_current_state()
        navigation_plan = self.__get_current_navigation_plan()
        self._behavioral_state.update_behavioral_state(state, navigation_plan)
        trajectory_params = self._policy.plan(self._behavioral_state)
        self.__publish_results(trajectory_params)
        self.__publish_debug(trajectory_params)

    # TODO: implement message passing, including type hints.
    def __get_current_state(self):
        pass

    def __get_current_navigation_plan(self):
        pass

    def __publish_results(self, results: TrajectoryParameters):
        pass

    def __publish_debug(self, debug_data: TrajectoryParameters):
        pass



