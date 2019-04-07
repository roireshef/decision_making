from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.state.state import State
from logging import Logger


class BehavioralState:
    @classmethod
    def create_from_state(cls, state: State, route_plan: RoutePlan, lane_cost_dict: Dict[int, float], logger: Logger):
        """
        This method updates the behavioral state according to the new world state and navigation plan.
         It fetches relevant features that will be used for the decision-making process.
        :param state: new world state
        :param nav_plan: navigation plan message
        :param lane_cost_dict: dictionary of key lane ID with value end cost of traversing lane
        :param logger
        :return: a new and updated BehavioralState
        """
        pass