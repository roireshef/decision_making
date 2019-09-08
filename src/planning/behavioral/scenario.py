from abc import abstractmethod, ABCMeta
from logging import Logger

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.behavioral.planner.RL_lane_merge_planner import RL_LaneMergePlanner
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner, ScenarioParams
from decision_making.src.planning.types import FS_SX
from decision_making.src.state.lane_merge_state import LaneMergeState, MERGE_LOOKAHEAD
from decision_making.src.state.state import State
from decision_making.src.utils.map_utils import MapUtils


class Planner:
    @staticmethod
    def plan(state: State):
        pass


class Scenario:

    @staticmethod
    def identify_scenario(behavioral_state: BehavioralGridState):
        """
        Given the current state, find the lane merge ahead and cars on the main road upstream the merge point.
        :param behavioral_state: current behavioral grid state
        :return: lane merge state or None (if no merge), s of merge-point in GFF
        """
        gff = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE]
        ego_fstate_on_gff = behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE]

        # Find the lanes before and after the merge point
        merge_lane_id = MapUtils.get_closest_lane_merge(gff, ego_fstate_on_gff[FS_SX], merge_lookahead=MERGE_LOOKAHEAD)
        return LaneMergeScenario if merge_lane_id is not None else DefaultScenario

    @staticmethod
    @abstractmethod
    def choose_planner(state: State, behavioral_state: BehavioralGridState, logger: Logger):
        pass


class DefaultScenario(Scenario):
    @staticmethod
    def choose_planner(state: State, behavioral_state: BehavioralGridState, logger: Logger):
        return SingleStepBehavioralPlanner(behavioral_state, logger)


class LaneMergeScenario(Scenario):
    @staticmethod
    def choose_planner(state: State, behavioral_state: BehavioralGridState, logger: Logger):
        lane_merge_state = LaneMergeState.build(state, behavioral_state)
        if lane_merge_state is None or len(lane_merge_state.actors) == 0:
            # if there is no lane merge ahead or the main road is empty, then perform the default strategy
            return SingleStepBehavioralPlanner(behavioral_state, logger)

        # there is a merge ahead with cars on the main road
        # try to find a rule-based lane merge that guarantees a safe merge even in the worst case scenario
        actions = RuleBasedLaneMergePlanner.create_safe_actions(lane_merge_state, ScenarioParams())
        return RuleBasedLaneMergePlanner(behavioral_state, lane_merge_state, actions, logger) if len(actions) > 0 \
            else RL_LaneMergePlanner(behavioral_state, lane_merge_state, logger)
