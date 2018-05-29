from logging import Logger
from typing import Optional

import numpy as np

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import \
    BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, DynamicActionRecipe, NavigationGoal
from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionRecipeEvaluator, \
    ActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.value_approximator import ValueApproximator
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import \
    CostBasedBehavioralPlanner
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State

import rte.python.profiler as prof
from logging import Logger
from typing import Optional

import numpy as np

import rte.python.profiler as prof
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import \
    BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, DynamicActionRecipe
from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionRecipeEvaluator, \
    ActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.value_approximator import ValueApproximator
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import \
    CostBasedBehavioralPlanner
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State, EgoState
from mapping.src.service.map_service import MapService


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
        self.logger.debug('Number of actions specified: %d (#%dS,#%dD)',
                          num_of_specified_actions, num_of_considered_static_actions, num_of_considered_dynamic_actions)

        # ActionSpec filtering
        action_specs_mask = self.action_spec_validator.filter_action_specs(action_specs, behavioral_state)

        # State-Action Evaluation
        action_costs = self.action_spec_evaluator.evaluate(behavioral_state, action_recipes, action_specs, action_specs_mask)

        # approximate cost-to-go per terminal state
        terminal_behavioral_states = self._generate_terminal_states(state, action_specs, action_recipes, action_specs_mask)
?       terminal_states_values = np.array([self.value_approximator.approximate(state) if action_specs_mask[i] else np.nan
                                           for i, state in enumerate(terminal_behavioral_states)])

        # generate goals for all terminal_behavioral_states, containing all lanes at the same distance
        navigation_goals = SingleStepBehavioralPlanner.generate_goals(
            behavioral_state.ego_state.road_localization.road_id, behavioral_state.ego_state.road_localization.road_lon,
            terminal_behavioral_states)

        terminal_states_values = np.array([self.value_approximator.approximate(state, navigation_goals[i])
                                           if action_specs_mask[i] else np.nan
                                           for i, state in enumerate(terminal_behavioral_states)])

        self.logger.debug('terminal states value: %s', np.array_repr(terminal_states_values).replace('\n', ' '))

        # compute "approximated Q-value" (action cost +  cost-to-go) for all actions
        action_q_cost = action_costs + terminal_states_values

        valid_idxs = np.where(action_specs_mask)[0]
        selected_action_index = valid_idxs[action_q_cost[valid_idxs].argmin()]
        selected_action_spec = action_specs[selected_action_index]

        print('Selected action: %d; rel_lat=%d\n' % (selected_action_index, action_recipes[selected_action_index].relative_lane.value))

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

    @staticmethod
    def generate_terminal_states(state: State, action_specs: List[ActionSpec], action_recipes: List[ActionRecipe],
                                  mask: np.ndarray, logger: Logger) -> List[BehavioralGridState]:
        """
        Given current state and action specifications, generate a corresponding list of future states using the
        predictor. Uses mask over list of action specifications to avoid unnecessary computation
        :param state: the current world state
        :param action_specs: list of action specifications
        :param mask: 1D mask vector (boolean) for filtering valid action specifications
        :return: a list of terminal states
        """
        # create a new behavioral state at the action end
        ego = state.ego_state
        cur_ego_loc = ego.road_localization
        road_id = cur_ego_loc.road_id
        lane_width = MapService.get_instance().get_road(road_id).lane_width

        terminal_behavioral_states = []
        for i, spec in enumerate(action_specs):
            if not mask[i]:
                terminal_behavioral_states.append(None)
                continue
            recipe = action_recipes[i]
            target_lane = cur_ego_loc.lane_num + recipe.relative_lane.value
            cpoint, yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, spec.s, target_lane * lane_width)
            terminal_ego = EgoState(ego.obj_id, ego.timestamp + int(spec.t * 1e9), cpoint[0], cpoint[1], cpoint[2], yaw,
                                    ego.size, 0, spec.v, 0, 0, 0, 0)
            predicted_objects = []  # TODO: use predictor
            terminal_state = State(None, predicted_objects, terminal_ego)
            new_behavioral_state = BehavioralGridState.create_from_state(terminal_state, logger)
            terminal_behavioral_states.append(new_behavioral_state)

        return terminal_behavioral_states

    @staticmethod
    def generate_goals(road_id: int, current_longitude: float, terminal_behavioral_states: List[BehavioralGridState]) \
            -> List[NavigationGoal]:
        num_lanes = MapService.get_instance().get_road(road_id).lanes_num
        des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
        return [NavigationGoal(road_id, current_longitude + 20 * des_vel, list(range(0, num_lanes)))
                if behavioral_states is not None else None
                for i, behavioral_states in enumerate(terminal_behavioral_states)]
