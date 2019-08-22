import numpy as np
import rte.python.profiler as prof
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.rule_based_lane_merge import RuleBasedLaneMerge
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, DynamicActionRecipe, \
    ActionSpec
from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionRecipeEvaluator, \
    ActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.value_approximator import ValueApproximator
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import \
    CostBasedBehavioralPlanner
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.state import State
from logging import Logger
from typing import Optional, List
from enum import Enum


class ActionEvaluationStrategyType(Enum):
    DEFAULT_RB = 0
    BRAKING = 1
    MERGE_RB = 2
    LANE_MERGE_RL = 3


class SingleStepBehavioralPlanner(CostBasedBehavioralPlanner):
    """
    For each received current-state:
     1.A behavioral, semantic state is created and its value is approximated.
     2.The full action-space is enumerated (recipes are generated).
     3.Recipes are filtered according to some pre-defined rules.
     4.Recipes are specified so ActionSpecs are created.
     5.Action Specs are evaluated.
     6.Lowest-Cost ActionSpec is chosen and its parameters are sent to TrajectoryPlanner.
    """
    def __init__(self, action_space: ActionSpace, recipe_evaluator: Optional[ActionRecipeEvaluator],
                 action_spec_evaluator: Optional[ActionSpecEvaluator],
                 action_spec_validator: Optional[ActionSpecFiltering],
                 value_approximator: ValueApproximator, predictor: EgoAwarePredictor, logger: Logger):
        super().__init__(action_space, recipe_evaluator, action_spec_evaluator, action_spec_validator,
                         value_approximator, predictor, logger)

    @prof.ProfileFunction()
    def plan(self, state: State, route_plan: RoutePlan):

        # create road semantic grid from the raw State object
        # behavioral_state contains road_occupancy_grid and ego_state
        behavioral_state = BehavioralGridState.create_from_state(state=state, route_plan=route_plan, logger=self.logger)

        # choose action evaluation strategy according to the current state (e.g. rule-based or RL policy)
        eval_strategy, action_specs = self._choose_strategy(state, behavioral_state)

        # State-Action Evaluation
        action_costs = self.actions_evaluator[eval_strategy].evaluate(state, behavioral_state, action_specs)

        # ActionSpec filtering
        action_specs_mask = self.action_spec_validator.filter_action_specs(action_specs, behavioral_state)

        # choose action (argmax)
        valid_idxs = np.where(action_specs_mask)[0]
        selected_action_index = valid_idxs[action_costs[valid_idxs].argmin()]
        selected_action_spec = action_specs[selected_action_index]

        trajectory_parameters = CostBasedBehavioralPlanner._generate_trajectory_specs(
            behavioral_state=behavioral_state, action_spec=selected_action_spec)
        visualization_message = BehavioralVisualizationMsg(reference_route_points=trajectory_parameters.reference_route.points)

        baseline_trajectory = CostBasedBehavioralPlanner.generate_baseline_trajectory(
            state.ego_state.timestamp_in_sec, selected_action_spec, trajectory_parameters,
            behavioral_state.projected_ego_fstates[selected_action_spec.relative_lane])

        self.logger.debug("Chosen behavioral action recipe %s (ego_timestamp: %.2f)",
                          action_recipes[selected_action_index], state.ego_state.timestamp_in_sec)
        self.logger.debug("Chosen behavioral action spec %s (ego_timestamp: %.2f)",
                          selected_action_spec, state.ego_state.timestamp_in_sec)

        self.logger.debug('In timestamp %f, selected action is %s with horizon: %f'
                          % (behavioral_state.ego_state.timestamp_in_sec,
                             action_recipes[selected_action_index],
                             selected_action_spec.t))

        return trajectory_parameters, baseline_trajectory, visualization_message

    def _choose_strategy(self, state: State, behavioral_state: BehavioralGridState) -> \
            [ActionEvaluationStrategyType, List[ActionSpec]]:
        """
        Given the current state, choose the strategy for actions evaluation in BP.
        Currently, perform evaluation by RL policy only if there is a lane merge ahead and there are cars on the
        main lanes upstream the merge point.
        :param state: current state
        :param behavioral_state: current behavioral grid state
        :return: action evaluation strategy
        """
        lane_merge_state, merge_point_in_gff = RuleBasedLaneMerge.create_lane_merge_state(state, behavioral_state)
        if lane_merge_state is None or len(lane_merge_state.actors) == 0:  # no merge ahead or no cars on the main road
            return ActionEvaluationStrategyType.DEFAULT_RULE_BASED, self._specify_default_action_space(behavioral_state)

        merge_specs = RuleBasedLaneMerge.create_safe_actions(lane_merge_state, merge_point_in_gff)
        if len(merge_specs) > 0:
            return ActionEvaluationStrategyType.MERGE_RB, merge_specs
        else:  # not arrived to the yellow line
            add_stop_bar()
            return ActionEvaluationStrategyType.LANE_MERGE_RL, self._specify_default_action_space(behavioral_state)

    def _specify_default_action_space(self, behavioral_state: BehavioralGridState) -> List[ActionSpec]:
        action_recipes = self.default_action_space.recipes

        # Recipe filtering
        recipes_mask = self.default_action_space.filter_recipes(action_recipes, behavioral_state)
        self.logger.debug('Number of actions originally: %d, valid: %d',
                          self.default_action_space.action_space_size, np.sum(recipes_mask))

        action_specs = np.full(len(action_recipes), None)
        valid_action_recipes = [action_recipe for i, action_recipe in enumerate(action_recipes) if recipes_mask[i]]
        action_specs[recipes_mask] = self.default_action_space.specify_goals(valid_action_recipes, behavioral_state)

        # TODO: FOR DEBUG PURPOSES!
        num_of_considered_static_actions = sum(isinstance(x, StaticActionRecipe) for x in valid_action_recipes)
        num_of_considered_dynamic_actions = sum(isinstance(x, DynamicActionRecipe) for x in valid_action_recipes)
        num_of_specified_actions = sum(x is not None for x in action_specs)
        self.logger.debug('Number of actions specified: %d (#%dS,#%dD)',
                          num_of_specified_actions, num_of_considered_static_actions, num_of_considered_dynamic_actions)
        return list(action_specs)
