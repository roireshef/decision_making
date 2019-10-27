from abc import abstractmethod
from logging import Logger

from decision_making.src.exceptions import LaneMergeNotFound
from decision_making.src.global_constants import MERGE_LOOKAHEAD
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.planner.RL_lane_merge_planner import RL_LaneMergePlanner
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner, \
    ScenarioParams, SimpleLaneMergeState
from decision_making.src.planning.types import FS_SX
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState
from decision_making.src.state.state import State
from decision_making.src.utils.map_utils import MapUtils


class Scenario:

    @staticmethod
    def identify_scenario(state: State, route_plan: RoutePlan):
        """
        Given the current state, identify the planning scenario (e.g. check existence of the lane merge ahead)
        :param state: current state
        :param route_plan: route plan
        :return: the chosen scenario class
        """
        try:
            # Try to find a lane merge connectivity with forward horizon MERGE_LOOKAHEAD from ego location
            ego_map_state = state.ego_state.map_state
            MapUtils.get_closest_lane_merge(ego_map_state.lane_id, ego_map_state.lane_fstate[FS_SX], MERGE_LOOKAHEAD, route_plan)
            return LaneMergeScenario

        except LaneMergeNotFound:
            return DefaultScenario

    @staticmethod
    @abstractmethod
    def choose_planner(state: State, route_plan: RoutePlan, logger: Logger) -> BasePlanner:
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
    def choose_planner(state: State, route_plan: RoutePlan, logger: Logger) -> BasePlanner:
        """
        see base class
        """
        return SingleStepBehavioralPlanner(logger)


class LaneMergeScenario(Scenario):
    @staticmethod
    def choose_planner(state: State, route_plan: RoutePlan, logger: Logger):

        simple_lane_merge_state = SimpleLaneMergeState.create_from_state(state, route_plan, logger)

        # try to find a rule-based lane merge that guarantees a safe merge even in the worst case scenario
        actions = RuleBasedLaneMergePlanner.create_max_vel_quartic_actions(simple_lane_merge_state, ScenarioParams())

        return RuleBasedLaneMergePlanner(actions, logger) if len(actions) > 0 else RL_LaneMergePlanner(logger)
