import time
from collections import defaultdict
from copy import deepcopy
from logging import Logger
from typing import Optional, List

import numpy as np

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import \
    BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, DynamicActionRecipe, ActionSpec
from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionRecipeEvaluator, \
    ActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.value_approximator import ValueApproximator
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import \
    CostBasedBehavioralPlanner
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State


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
                 value_approximator: ValueApproximator, predictor: Predictor, logger: Logger):
        super().__init__(action_space, recipe_evaluator, action_spec_evaluator, action_spec_validator, value_approximator,
                         predictor, logger)

    def plan(self, state: State, nav_plan: NavigationPlanMsg):
        action_recipes = self.action_space.recipes
        # TODO: FOR DEBUG PURPOSES!
        st = time.time()

        # create road semantic grid from the raw State object
        # behavioral_state contains road_occupancy_grid and ego_state
        behavioral_state = BehavioralGridState.create_from_state(state=state, logger=self.logger)

        # TODO: FOR DEBUG PURPOSES!
        post_grid_creation_time = time.time()
        self.logger.debug('creation of behavioral state took %f seconds', post_grid_creation_time - st)

        # Recipe filtering
        recipes_mask = self.action_space.filter_recipes(action_recipes, behavioral_state)

        # TODO: FOR DEBUG PURPOSES!
        post_recipe_filters_time = time.time()
        self.logger.debug('Number of actions originally: %d, valid: %d, filter processing time: %f',
                          self.action_space.action_space_size, np.sum(recipes_mask),
                          post_recipe_filters_time-post_grid_creation_time)

        # Action specification
        # TODO: replace numpy array with fast sparse-list implementation
        action_specs = np.full(action_recipes.__len__(), None)
        valid_action_recipes = [action_recipe for i, action_recipe in enumerate(action_recipes) if recipes_mask[i]]
        action_specs[recipes_mask] = self.action_space.specify_goals(valid_action_recipes, behavioral_state)
        action_specs = list(action_specs)

        # TODO: FOR DEBUG PURPOSES!
        num_of_considered_static_actions = sum(isinstance(x, StaticActionRecipe) for x in valid_action_recipes)
        num_of_considered_dynamic_actions = sum(isinstance(x, DynamicActionRecipe) for x in valid_action_recipes)
        num_of_specified_actions = sum(x is not None for x in action_specs)
        self.logger.debug('Number of actions specified: %d (#%dS,#%dD), specify processing time: %f',
                          num_of_specified_actions, num_of_considered_static_actions, num_of_considered_dynamic_actions,
                          time.time()-post_recipe_filters_time)

        # ActionSpec filtering
        action_specs_mask = self.action_spec_validator.filter_action_specs(action_specs, behavioral_state)

        # State-Action Evaluation
        action_costs = self.action_spec_evaluator.evaluate(behavioral_state, action_recipes, action_specs, action_specs_mask)

        # approximate cost-to-go per terminal state
        terminal_behavioral_states = self._generate_terminal_states(state, action_specs, action_specs_mask)
        terminal_states_values = np.array([self.value_approximator.approximate(state) if action_specs_mask[i] else np.nan
                                           for i, state in enumerate(terminal_behavioral_states)])

        # compute "approximated Q-value" (action cost +  cost-to-go) for all actions
        action_q_cost = action_costs + terminal_states_values

        valid_idxs = np.where(action_specs_mask)[0]
        selected_action_index = valid_idxs[action_q_cost[valid_idxs].argmin()]
        selected_action_spec = action_specs[selected_action_index]

        trajectory_parameters = CostBasedBehavioralPlanner._generate_trajectory_specs(behavioral_state=behavioral_state,
                                                                                      action_spec=selected_action_spec,
                                                                                      navigation_plan=nav_plan)
        visualization_message = BehavioralVisualizationMsg(reference_route=trajectory_parameters.reference_route)

        # keeping selected actions for next iteration use
        self._last_action = action_recipes[selected_action_index]
        self._last_action_spec = selected_action_spec

        baseline_trajectory = CostBasedBehavioralPlanner.generate_baseline_trajectory(state.ego_state,
                                                                                      selected_action_spec)

        self.logger.debug("Chosen behavioral semantic action is %s, %s",
                          action_recipes[selected_action_index].__dict__, selected_action_spec.__dict__)

        return trajectory_parameters, baseline_trajectory, visualization_message

    def _generate_terminal_states(self, state: State, action_specs: List[ActionSpec], mask: np.ndarray) -> List[State]:
        """
        Given current state and action specifications, generate a corresponding list of future states using the
        predictor. Uses mask over list of action specifications to avoid unnecessary computation
        :param state: the current world state
        :param action_specs: list of action specifications
        :param mask: 1D mask vector (boolean) for filtering valid action specifications
        :return: a list of terminal states
        """
        # TODO: validate time units (all in seconds? global?)
        # generate the simulated terminal states for all actions using predictor
        action_horizons = np.array([action_spec.t
                                    if mask[i] else np.nan
                                    for i, action_spec in enumerate(action_specs)])

        # TODO: replace numpy array with fast sparse-list implementation
        terminal_states = np.full(shape=action_horizons.shape, fill_value=None)
        terminal_states[mask] = deepcopy(state)          # TODO: fix bug in predictor
        # self.predictor.predict_state(state, action_horizons[mask] + state.ego_state.timestamp_in_sec)
        terminal_states = list(terminal_states)

        # transform terminal states into behavioral states
        terminal_behavioral_states = [BehavioralGridState.create_from_state(state=terminal_state, logger=self.logger)
                                      if mask[i] else None
                                      for i, terminal_state in enumerate(terminal_states)]

        return terminal_behavioral_states
