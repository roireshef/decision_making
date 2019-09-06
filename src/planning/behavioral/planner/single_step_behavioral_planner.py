import numpy as np
import rte.python.profiler as prof
from decision_making.src.exceptions import NoActionsLeftForBPError
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.evaluators.action_evaluator_by_policy import LaneMergeRLPolicy
from decision_making.src.planning.behavioral.evaluators.single_lane_action_spec_evaluator import \
    SingleLaneActionsEvaluator
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner, \
    ScenarioParams, LaneMergeState
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, DynamicActionRecipe, \
    ActionSpec, AggressivenessLevel, RelativeLane, ActionType
from decision_making.src.planning.behavioral.planner.base_planner import \
    BasePlanner
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.state import State
from logging import Logger
from typing import Optional, List


class SingleStepBehavioralPlanner(BasePlanner):
    """
    For each received current-state:
     1.A behavioral, semantic state is created and its value is approximated.
     2.The full action-space is enumerated (recipes are generated).
     3.Recipes are filtered according to some pre-defined rules.
     4.Recipes are specified so ActionSpecs are created.
     5.Action Specs are evaluated.
     6.Lowest-Cost ActionSpec is chosen and its parameters are sent to TrajectoryPlanner.
    """
    def __init__(self, action_space: ActionSpace, predictor: EgoAwarePredictor, logger: Logger):
        super().__init__(logger)
        self.default_action_space = action_space
        self.predictor = predictor

    def _choose_action(self, behavioral_state: BehavioralGridState):
        action_specs = self._specify_and_filter_default_actions(behavioral_state)
        costs = self._evaluate(behavioral_state, action_specs)
        return action_specs[np.argmin(costs)]

    def _evaluate(self, behavioral_state: BehavioralGridState, action_specs: List[ActionSpec]) -> np.ndarray:
        """
        Evaluates Action-Specifications based on the following logic:
        * Only takes into account actions on RelativeLane.SAME_LANE
        * If there's a leading vehicle, try following it (ActionType.FOLLOW_VEHICLE, lowest aggressiveness possible)
        * If no action from the previous bullet is found valid, find the ActionType.FOLLOW_ROAD_SIGN action with lowest
        * aggressiveness, and save it.
        * Find the ActionType.FOLLOW_LANE action with maximal allowed velocity and lowest aggressiveness possible,
        * and save it.
        * Compare the saved FOLLOW_ROAD_SIGN and FOLLOW_LANE actions, and choose between them.
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_specs: specifications of action_recipes.
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """
        costs = np.full(len(action_specs), 1)

        # first try to find a valid dynamic action (FOLLOW_VEHICLE) for SAME_LANE
        follow_vehicle_valid_action_idxs = [i for i, spec in enumerate(action_specs)
                                            if spec is not None
                                            and spec.recipe.relative_lane == RelativeLane.SAME_LANE
                                            and spec.recipe.action_type == ActionType.FOLLOW_VEHICLE]
        # The selection is only by aggressiveness, since it relies on the fact that we only follow a vehicle on the
        # SAME lane, which means there is only 1 possible vehicle to follow, so there is only 1 target vehicle speed.
        if len(follow_vehicle_valid_action_idxs) > 0:
            costs[follow_vehicle_valid_action_idxs[0]] = 0  # choose the found dynamic action, which is least aggressive
            return costs

        # next try to find a valid road sign action (FOLLOW_ROAD_SIGN) for SAME_LANE.
        # Selection only needs to consider aggressiveness level, as all the target speeds are ZERO_SPEED.
        # Tentative decision is kept in selected_road_sign_idx, to be compared against STATIC actions
        follow_road_sign_valid_action_idxs = [i for i, spec in enumerate(action_specs)
                                              if spec is not None
                                              and spec.recipe.relative_lane == RelativeLane.SAME_LANE
                                              and spec.recipe.action_type == ActionType.FOLLOW_ROAD_SIGN]
        if len(follow_road_sign_valid_action_idxs) > 0:
            # choose the found action, which is least aggressive.
            selected_road_sign_idx = follow_road_sign_valid_action_idxs[0]
        else:
            selected_road_sign_idx = -1

        # last, look for valid static action
        filtered_follow_lane_idxs = [i for i, spec in enumerate(action_specs)
                                     if spec is not None and isinstance(spec.recipe, StaticActionRecipe)
                                     and spec.recipe.relative_lane == RelativeLane.SAME_LANE]
        if len(filtered_follow_lane_idxs) > 0:
            # find the minimal aggressiveness level among valid static recipes
            min_aggr_level = min([action_specs[idx].recipe.aggressiveness.value for idx in filtered_follow_lane_idxs])

            # among the minimal aggressiveness level, find the fastest action
            follow_lane_valid_action_idxs = [idx for idx in filtered_follow_lane_idxs
                                             if action_specs[idx].recipe.aggressiveness.value == min_aggr_level]

            selected_follow_lane_idx = follow_lane_valid_action_idxs[-1]
        else:
            selected_follow_lane_idx = -1

        # finally decide between the road sign and the static action
        if selected_road_sign_idx < 0 and selected_follow_lane_idx < 0:
            # if no action of either type was found, raise an error
            raise NoActionsLeftForBPError("All actions were filtered in BP. timestamp_in_sec: %f" %
                                          behavioral_state.ego_state.timestamp_in_sec)
        elif selected_road_sign_idx < 0:
            # if no road sign action is found, select the static action
            costs[selected_follow_lane_idx] = 0
            return costs
        elif selected_follow_lane_idx < 0:
            # if no static action is found, select the road sign action
            costs[selected_road_sign_idx] = 0
            return costs
        else:
            # if both road sign and static actions are valid - choose
            if SingleStepBehavioralPlanner._is_static_action_preferred(action_specs, selected_road_sign_idx):
                costs[selected_follow_lane_idx] = 0
                return costs
            else:
                costs[selected_road_sign_idx] = 0
                return costs

    @staticmethod
    def _is_static_action_preferred(action_specs: List[ActionSpec], road_sign_idx: int):
        """
        Selects if a STATIC or ROAD_SIGN action is preferred.
        This can be based on any criteria.
        For example:
            always prefer 1 type of action,
            select action type if aggressiveness is as desired
            Toggle between the 2
        :param action_specs: of all possible actions
        :param road_sign_idx: of calmest road sign action
        :return: True if static action is preferred, False otherwise
        """
        road_sign_action = action_specs[road_sign_idx]
        # Avoid AGGRESSIVE stop. TODO relax the restriction of not selective an aggressive road sign
        return road_sign_action.recipe.aggressiveness != AggressivenessLevel.CALM










    def _choose_strategy(self, state: State, behavioral_state: BehavioralGridState) -> [List[ActionSpec], np.array]:
        """
        Given the current state, choose the strategy for actions evaluation in BP.
        Currently, perform evaluation by RL policy only if there is a lane merge ahead and there are cars on the
        main lanes upstream the merge point.
        :param state: current state
        :param behavioral_state: current behavioral grid state
        :return: action evaluation strategy
        """
        # first check if there is a lane merge ahead
        lane_merge_state = LaneMergeState.build(state, behavioral_state)
        if lane_merge_state is None or len(lane_merge_state.actors) == 0:
            # there is no lane merge ahead, then perform the default strategy with the default action space and
            # the default rule-based policy
            action_specs = self._specify_default_action_space(behavioral_state)
            action_costs = SingleLaneActionsEvaluator.evaluate(action_specs)
            return action_specs, action_costs

        # there is a merge ahead with cars on the main road
        # try to find a rule-based lane merge that guarantees a safe merge even in the worst case scenario
        lane_merge_actions, action_specs = RuleBasedLaneMergePlanner.create_safe_actions(lane_merge_state, ScenarioParams())
        action_costs = RuleBasedLaneMergeEvaluator.evaluate(lane_merge_actions)
        if len(lane_merge_actions) > 0:  # then the safe lane merge was found
            return action_specs, action_costs

        # the safe lane merge can not be guaranteed, so add stop bar at red line and perform RL
        action_specs = self._specify_default_action_space(behavioral_state)
        action_costs = LaneMergeRLPolicy.evaluate(lane_merge_state)
        add_stop_bar()
        return action_specs, action_costs
