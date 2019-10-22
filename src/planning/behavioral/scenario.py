from abc import abstractmethod
from logging import Logger

from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.state.state import State


class Planner:
    @staticmethod
    def plan(state: State):
        pass


class Scenario:

    @staticmethod
    def identify_scenario(state: State, route_plan: RoutePlan):
        """
        Given the current state, identify the planning scenario (e.g. check existence of the lane merge ahead)
        :param state: current state
        :param route_plan: route plan
        :return: scenario type
        """
        return DefaultScenario

    @staticmethod
    @abstractmethod
    def choose_planner(state: State, route_plan: RoutePlan, logger: Logger):
        pass


class DefaultScenario(Scenario):
    @staticmethod
    def choose_planner(state: State, route_plan: RoutePlan, logger: Logger):
        # behavioral_state contains road_occupancy_grid and ego_state
        behavioral_state = BehavioralGridState.create_from_state(state=state, route_plan=route_plan, logger=logger)

        return SingleStepBehavioralPlanner(behavioral_state, route_plan, logger)
