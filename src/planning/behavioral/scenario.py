from abc import abstractmethod
from logging import Logger

from decision_making.src.global_constants import MERGE_LOOKAHEAD
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.planner.RL_lane_merge_planner import RL_LaneMergePlanner
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner, ScenarioParams
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
        ego_map_state = state.ego_state.map_state

        # Find the lanes before and after the merge point
        merge_lane_id, _, _ = MapUtils.get_closest_lane_merge(ego_map_state.lane_id, ego_map_state.lane_fstate[FS_SX],
                                                              MERGE_LOOKAHEAD, route_plan)
        return LaneMergeScenario if merge_lane_id is not None else DefaultScenario

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
        # behavioral_state contains road_occupancy_grid and ego_state
        behavioral_state = BehavioralGridState.create_from_state(state=state, route_plan=route_plan, logger=logger)

        return SingleStepBehavioralPlanner(behavioral_state, route_plan, logger)


class LaneMergeScenario(Scenario):
    @staticmethod
    def choose_planner(state: State, route_plan: RoutePlan, logger: Logger):

        lane_merge_state = LaneMergeState.create_from_state(state=state, route_plan=route_plan, logger=logger)

        # try to find a rule-based lane merge that guarantees a safe merge even in the worst case scenario
        actions = RuleBasedLaneMergePlanner.create_safe_actions(lane_merge_state, ScenarioParams())
        return RuleBasedLaneMergePlanner(lane_merge_state, actions, logger) if len(actions) > 0 \
            else RL_LaneMergePlanner(lane_merge_state, logger)
