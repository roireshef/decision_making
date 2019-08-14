import numpy as np
import rte.python.profiler as prof
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, DynamicActionRecipe, ActionRecipe, \
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
                 action_spec_evaluator: Optional[ActionSpecEvaluator], action_spec_validator: Optional[ActionSpecFiltering],
                 value_approximator: ValueApproximator, predictor: EgoAwarePredictor, logger: Logger):
        super().__init__(action_space, recipe_evaluator, action_spec_evaluator, action_spec_validator, value_approximator,
                         predictor, logger)

    def choose_action(self, state: State, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                      recipes_mask: List[bool], route_plan: RoutePlan) -> (int, ActionSpec):
        """
        upon receiving an input state, return an action specification and its respective index in the given list of
        action recipes.
        :param recipes_mask: A list of boolean values, which are True if respective action recipe in
        input argument action_recipes is valid, else False.
        :param state: the current world state
        :param behavioral_state: processed behavioral state
        :param action_recipes: a list of enumerated semantic actions [ActionRecipe].
        :param route_plan:
        :return: a tuple of the selected action index and selected action spec itself (int, ActionSpec).
        """

        # Action specification
        # TODO: replace numpy array with fast sparse-list implementation
        action_specs = np.full(len(action_recipes), None)
        valid_action_recipes = [action_recipe for i, action_recipe in enumerate(action_recipes) if recipes_mask[i]]
        action_specs[recipes_mask] = self.action_space.specify_goals(valid_action_recipes, behavioral_state)
        action_specs = list(action_specs)

        # TODO: FOR DEBUG PURPOSES!
        num_of_considered_static_actions = sum(isinstance(x, StaticActionRecipe) for x in valid_action_recipes)
        num_of_considered_dynamic_actions = sum(isinstance(x, DynamicActionRecipe) for x in valid_action_recipes)
        num_of_specified_actions = sum(x is not None for x in action_specs)
        self.logger.debug('Number of actions specified: %d (#%dS,#%dD)',
                          num_of_specified_actions, num_of_considered_static_actions, num_of_considered_dynamic_actions)

        # ActionSpec filtering
        action_specs_mask = self.action_spec_validator.filter_action_specs(action_specs, behavioral_state)

        # State-Action Evaluation
        action_costs = self.action_spec_evaluator.evaluate(behavioral_state, action_recipes, action_specs, action_specs_mask, route_plan)

        # approximate cost-to-go per terminal state
        terminal_behavioral_states = self._generate_terminal_states(state, behavioral_state, action_specs,
                                                                    action_specs_mask, route_plan)
        # TODO: NavigationPlan is now None and should be meaningful when we have one
        terminal_states_values = np.array([self.value_approximator.approximate(state, None) if action_specs_mask[i] else np.nan
                                           for i, state in enumerate(terminal_behavioral_states)])

        self.logger.debug('terminal states value: %s', np.array_repr(terminal_states_values).replace('\n', ' '))

        # compute "approximated Q-value" (action cost +  cost-to-go) for all actions
        action_q_cost = action_costs + terminal_states_values

        valid_idxs = np.where(action_specs_mask)[0]
        selected_action_index = valid_idxs[action_q_cost[valid_idxs].argmin()]
        selected_action_spec = action_specs[selected_action_index]

        return selected_action_index, selected_action_spec

    @prof.ProfileFunction()
    def plan(self, state: State, route_plan: RoutePlan):

        action_recipes = self.action_space.recipes

        # create road semantic grid from the raw State object
        # behavioral_state contains road_occupancy_grid and ego_state
        behavioral_state = BehavioralGridState.create_from_state(state=state, route_plan=route_plan, logger=self.logger)

        # Recipe filtering
        recipes_mask = self.action_space.filter_recipes(action_recipes, behavioral_state)

        self.logger.debug('Number of actions originally: %d, valid: %d',
                          self.action_space.action_space_size, np.sum(recipes_mask))
        selected_action_index, selected_action_spec = self.choose_action(state, behavioral_state, action_recipes,
                                                                         recipes_mask, route_plan)
        trajectory_parameters = CostBasedBehavioralPlanner._generate_trajectory_specs(
            behavioral_state=behavioral_state, action_spec=selected_action_spec)
        visualization_message = BehavioralVisualizationMsg(
            reference_route_points=trajectory_parameters.reference_route.points)

        # keeping selected actions for next iteration use
        self._last_action = action_recipes[selected_action_index]
        self._last_action_spec = selected_action_spec

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
