import numpy as np
from logging import Logger

from decision_making.src.exceptions import NoActionsLeftForBPError
from decision_making.src.global_constants import FILTER_V_0_GRID, FILTER_V_T_GRID
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, RelativeLane
from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionRecipeEvaluator
from typing import List

from decision_making.src.planning.behavioral.evaluators.action_evaluator_by_policy import LaneMergeRLPolicy
from decision_making.src.planning.types import FS_SX, FS_SV
from decision_making.src.planning.utils.kinematics_utils import BrakingDistances
from decision_making.src.state.state import State
from decision_making.src.utils.map_utils import MapUtils


class LaneMergeEvaluator(ActionRecipeEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate(self, state: State, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                 actions_mask: List[bool], policy) -> np.array:
        """
        Given current state and actions mask, return the actions cost by operating lane merge policy on the state.
        In case of failure it is rule-based 'stop at red-line' policy, in case of success it is RL policy.
        Filtered actions get the maximal cost 1.
        :param state: current state
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_recipes: actions list
        :param actions_mask: actions mask
        :param policy: RL policy
        :return: array of actions costs: the lower the better
        """


        gff = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE]
        ego_s = behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE][FS_SX]
        ego_v = behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE][FS_SV]
        merge_lane_id = MapUtils.get_closest_lane_merge(gff, ego_s, merge_lookahead=300)
        gff_state = gff.convert_from_segment_state(np.zeros(6), merge_lane_id)
        dist_from_red_line = gff_state[FS_SX] - ego_s
        braking_distances = BrakingDistances.create_braking_distances()
        braking_distance = braking_distances[FILTER_V_0_GRID.get_index(ego_v), FILTER_V_T_GRID.get_index(0)]

        if braking_distance >= dist_from_red_line:  # fix it since probably it may be too late to brake
            # choose policy for braking at stop bar
        else:  # apply RL policy
            eval_by_rl_policy = LaneMergeRLPolicy(self.logger)
            return eval_by_rl_policy.evaluate(state, behavioral_state, action_recipes, actions_mask, policy)
