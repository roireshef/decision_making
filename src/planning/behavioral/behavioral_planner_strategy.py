from enum import Enum

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FS_SX
from decision_making.src.state.state import State
from decision_making.src.utils.map_utils import MapUtils


class ActionEvaluationStrategyType(Enum):
    RULE_BASED = 0
    RL_POLICY = 1


class ActionEvaluationStrategy:

    @staticmethod
    def choose_strategy(state: State, behavioral_state: BehavioralGridState) -> ActionEvaluationStrategyType:
        """
        Given the current state, choose the strategy for actions evaluation in BP.
        Currently, perform evaluation by RL policy only if there is a lane merge ahead and there are cars on the
        main lanes upstream the merge point.
        :param state: current state
        :param behavioral_state: current behavioral grid state
        :return: action evaluation strategy
        """
        gff = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE]
        ego_s = behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE][FS_SX]

        # Find the lanes before and after the merge point
        merge_lane_id = MapUtils.get_closest_lane_merge(gff, ego_s, merge_lookahead=300)
        if merge_lane_id is None:
            return ActionEvaluationStrategyType.RULE_BASED
        after_merge_lane_id = MapUtils.get_downstream_lanes(merge_lane_id)[0]
        main_lane_ids = MapUtils.get_straight_upstream_lanes(after_merge_lane_id, max_back_horizon=300)
        if len(main_lane_ids) == 0:
            return ActionEvaluationStrategyType.RULE_BASED

        # check existence of cars on the upstream main road
        for obj in state.dynamic_objects:
            if obj.map_state.lane_id in main_lane_ids:
                return ActionEvaluationStrategyType.RL_POLICY  # a car was found on the main road
        return ActionEvaluationStrategyType.RULE_BASED  # any car was not found on the main road
