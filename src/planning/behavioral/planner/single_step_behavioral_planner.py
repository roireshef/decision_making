from logging import Logger
from typing import Optional, List

import numpy as np

import rte.python.profiler as prof
from decision_making.src.global_constants import EPS, TRAJECTORY_TIME_RESOLUTION
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import \
    BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, DynamicActionRecipe, ActionRecipe, \
    ActionSpec
from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionRecipeEvaluator, \
    ActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.value_approximator import ValueApproximator
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import \
    CostBasedBehavioralPlanner
from decision_making.src.planning.types import FS_DX, FS_SX
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
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
                 value_approximator: ValueApproximator, predictor: EgoAwarePredictor, logger: Logger):
        super().__init__(action_space, recipe_evaluator, action_spec_evaluator, action_spec_validator, value_approximator,
                         predictor, logger)

    def choose_action(self, state: State, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                      recipes_mask: List[bool]) -> (int, ActionSpec):
        """
        upon receiving an input state, return an action specification and its respective index in the given list of
        action recipes.
        :param recipes_mask: A list of boolean values, which are True if respective action recipe in
        input argument action_recipes is valid, else False.
        :param state: the current world state
        :param behavioral_state: processed behavioral state
        :param action_recipes: a list of enumerated semantic actions [ActionRecipe].
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

        # filter specs by RSS safety
        action_specs_mask_safe = self.check_actions_safety(state, action_specs, action_specs_mask)

        # State-Action Evaluation
        action_costs = self.action_spec_evaluator.evaluate(behavioral_state, action_recipes, action_specs, action_specs_mask_safe)

        # approximate cost-to-go per terminal state
        terminal_behavioral_states = self._generate_terminal_states(state, action_specs, action_specs_mask_safe)
        # TODO: NavigationPlan is now None and should be meaningful when we have one
        terminal_states_values = np.array([self.value_approximator.approximate(state, None) if action_specs_mask_safe[i] else np.nan
                                           for i, state in enumerate(terminal_behavioral_states)])

        self.logger.debug('terminal states value: %s', np.array_repr(terminal_states_values).replace('\n', ' '))

        # compute "approximated Q-value" (action cost +  cost-to-go) for all actions
        action_q_cost = action_costs + terminal_states_values

        valid_idxs = np.where(action_specs_mask_safe)[0]
        selected_action_index = valid_idxs[action_q_cost[valid_idxs].argmin()]
        selected_action_spec = action_specs[selected_action_index]

        return selected_action_index, selected_action_spec

    def check_actions_safety(self, state: State, action_specs: List[ActionSpec], action_specs_mask: np.array) \
            -> List[bool]:
        """
        Check RSS safety for all action specs, for which action_specs_mask is true.
        An action spec is considered safe if it's safe wrt all dynamic objects for all timestamps < spec.t.
        :param state: the current world state
        :param action_specs: list of action specifications
        :param action_specs_mask: 1D mask vector (boolean) for filtering valid action specifications
        :return: boolean list of safe specifications. The list's size is equal to the original action_specs size.
        Specifications filtered by action_specs_mask are considered "unsafe".
        """
        # TODO: in the current version T_d = T_s. Test safety for different values of T_d.
        ego = state.ego_state
        ego_init_fstate = ego.map_state.road_fstate

        spec_arr = np.array([np.array([spec.t, spec.s, spec.v, spec.d])
                             for i, spec in enumerate(action_specs) if action_specs_mask[i]])
        t_arr, s_arr, v_arr, d_arr = np.split(spec_arr, 4, axis=1)
        zeros = np.zeros(t_arr.shape[0])

        init_fstates = np.tile(ego_init_fstate, t_arr.shape[0]).reshape(t_arr.shape[0], 6)
        target_fstates = np.c_[s_arr, v_arr, zeros, d_arr, zeros, zeros]

        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(t_arr))

        constraints_s = np.concatenate((init_fstates[:, :FS_DX], target_fstates[:, :FS_DX]), axis=1)
        constraints_d = np.concatenate((init_fstates[:, FS_DX:], target_fstates[:, FS_DX:]), axis=1)

        poly_coefs_s = QuinticPoly1D.zip_solve(A_inv, constraints_s)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv, constraints_d)

        time_points = np.arange(0, np.max(t_arr) + EPS, TRAJECTORY_TIME_RESOLUTION)
        fstates_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs_s, time_points)
        fstates_d = QuinticPoly1D.polyval_with_derivatives(poly_coefs_d, time_points)  # T_d = T_s
        ftrajectories = np.concatenate((fstates_s, fstates_d), axis=-1)

        # set all points beyond spec.t at infinity, such that they will be safe and will not affect the result
        for i, ftrajectory in enumerate(ftrajectories):
            end_t_idx = int(t_arr[i] / TRAJECTORY_TIME_RESOLUTION)
            ftrajectory[end_t_idx:, FS_SX] = np.inf

        # create objects' trajectories
        obj_fstates = np.array([obj.map_state.road_fstate for obj in state.dynamic_objects])
        obj_sizes = [obj.size for obj in state.dynamic_objects]
        obj_trajectories = np.array(self.predictor.predict_frenet_states(obj_fstates, time_points))

        # calculate safety for each trajectory, each object, each timestamp
        safe_times = SafetyUtils.get_safe_times(ftrajectories, ego.size, obj_trajectories, obj_sizes)
        # trajectory is considered safe if it's safe wrt all dynamic objects for all timestamps
        safe_trajectories = safe_times.all(axis=(1, 2))

        # assign safety to the specs, for which specs_mask is true
        safe_specs = np.copy(np.array(action_specs_mask))
        safe_specs[safe_specs] = safe_trajectories
        return list(safe_specs)  # list's size like the original action_specs size

    @prof.ProfileFunction()
    def plan(self, state: State, nav_plan: NavigationPlanMsg):
        action_recipes = self.action_space.recipes

        # create road semantic grid from the raw State object
        # behavioral_state contains road_occupancy_grid and ego_state
        behavioral_state = BehavioralGridState.create_from_state(state=state, logger=self.logger)

        # Recipe filtering
        recipes_mask = self.action_space.filter_recipes(action_recipes, behavioral_state)

        self.logger.debug('Number of actions originally: %d, valid: %d',
                          self.action_space.action_space_size, np.sum(recipes_mask))

        selected_action_index, selected_action_spec = self.choose_action(state, behavioral_state, action_recipes, recipes_mask)

        trajectory_parameters = CostBasedBehavioralPlanner._generate_trajectory_specs(behavioral_state=behavioral_state,
                                                                                      action_spec=selected_action_spec,
                                                                                      navigation_plan=nav_plan)
        visualization_message = BehavioralVisualizationMsg(reference_route=trajectory_parameters.reference_route)

        # keeping selected actions for next iteration use
        self._last_action = action_recipes[selected_action_index]
        self._last_action_spec = selected_action_spec

        baseline_trajectory = CostBasedBehavioralPlanner.generate_baseline_trajectory(state.ego_state,
                                                                                      selected_action_spec)

        self.logger.debug("Chosen behavioral action recipe %s (ego_timestamp: %.2f)",
                          action_recipes[selected_action_index], state.ego_state.timestamp_in_sec)
        self.logger.debug("Chosen behavioral action spec %s (ego_timestamp: %.2f)",
                          selected_action_spec, state.ego_state.timestamp_in_sec)

        return trajectory_parameters, baseline_trajectory, visualization_message


