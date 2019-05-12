from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.state.state import State
from logging import Logger


class BehavioralState:
    @classmethod
    def create_from_state(cls, state: State, nav_plan: NavigationPlanMsg, logger: Logger):
        """
        This method updates the behavioral state according to the new world state and navigation plan.
         It fetches relevant features that will be used for the decision-making process.
        :param state: new world state
        :param nav_plan: navigation plan message
        :param logger
        :return: a new and updated BehavioralState
        """
        pass