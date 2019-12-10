from abc import abstractmethod
from logging import Logger

from decision_making.src.global_constants import LANE_MERGE_STATE_FAR_AWAY_DISTANCE
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.planning.types import FS_SX
from decision_making.src.state.state import State
from decision_making.src.utils.map_utils import MapUtils


class Scenario:

    @staticmethod
    def identify_scenario(state: State, route_plan: RoutePlan, logger: Logger):
        """
        Given the current state, identify the planning scenario (e.g. check existence of the lane merge ahead)
        :param state: current state
        :param route_plan: route plan
        :param logger:
        :return: the chosen scenario class
        """
        # Try to find a lane merge connectivity with forward horizon MERGE_LOOKAHEAD from ego location
        ego_map_state = state.ego_state.map_state

        if MapUtils.get_merge_lane_id(ego_map_state.lane_id, ego_map_state.lane_fstate[FS_SX],
                                      LANE_MERGE_STATE_FAR_AWAY_DISTANCE, route_plan, logger):
            return LaneMergeScenario
        else:
            print('DefaultScenario')
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


class LaneMergeScenario(Scenario):
    @staticmethod
    def choose_planner(state: State, route_plan: RoutePlan, logger: Logger):
        """
        see base class
        """
        return RuleBasedLaneMergePlanner
