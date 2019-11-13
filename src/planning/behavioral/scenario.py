from abc import abstractmethod
from logging import Logger

from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.state.state import State


class Scenario:

    @staticmethod
    def identify_scenario(state: State, route_plan: RoutePlan):
        """
        Given the current state, identify the planning scenario (e.g. check existence of the lane merge ahead)
        :param state: current state
        :param route_plan: route plan
        :return: the chosen scenario class
        """
        # TODO: implement the function

        # currently always returns the default scenario
        return DefaultScenario

    @staticmethod
    @abstractmethod
    def choose_planner(state: State, route_plan: RoutePlan, logger: Logger):
        """
        Choose the appropriate planner for the specific scenario, given the current state. Each scenario has its own
        list of planners.
        :param state: current state
        :param route_plan: route plan
        :param logger:
        :return: the chosen planner instance, initialized by the current state
        """
        pass


class DefaultScenario(Scenario):
    @staticmethod
    def choose_planner(state: State, route_plan: RoutePlan, logger: Logger):
        """
        see base class
        """
        return SingleStepBehavioralPlanner
