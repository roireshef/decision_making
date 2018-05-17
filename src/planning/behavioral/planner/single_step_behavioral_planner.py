from logging import Logger
import time
from typing import Optional

import numpy as np

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, ActionType, NavigationGoal
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

    def plan(self, state: State, nav_plan: NavigationPlanMsg):
        action_recipes = self.action_space.recipes
        # create road semantic grid from the raw State object
        # behavioral_state contains road_occupancy_grid and ego_state
        behavioral_state = BehavioralGridState.create_from_state(state=state, logger=self.logger)

        # TODO: this should evaluate the terminal states!
        #current_state_value = self.value_approximator.evaluate_state(behavioral_state)

        # TODO: FOR DEBUG PURPOSES!
        time_before_filters = time.time()

        # Recipe filtering
        recipes_mask = self.action_space.filter_recipes(action_recipes, behavioral_state)

        self.logger.debug('Number of actions originally: %d, valid: %d, filter processing time: %f',
                          self.action_space.action_space_size, np.sum(recipes_mask), time.time()-time_before_filters)

        # State-Action Evaluation
        # action_costs = self.recipe_evaluator.evaluate(behavioral_state, action_recipes, recipes_mask)

        # TODO: FOR DEBUG PURPOSES!
        time_before_specify = time.time()

        # Action specification
        action_specs = [self.action_space.specify_goal(recipe, behavioral_state) if recipes_mask[i] else None
                        for i, recipe in enumerate(action_recipes)]

        num_of_specified_actions = sum(x is not None for x in action_specs)
        self.logger.debug('Number of actions specified: %d, specify processing time: %f',
                          num_of_specified_actions, time.time()-time_before_specify)

        # ActionSpec filtering
        action_specs_mask = self.action_spec_validator.filter_action_specs(action_specs, behavioral_state)

        # print('action_specs_mask[76]=%d' % (action_specs_mask[76]))

        # State-Action Evaluation
        action_costs = self.action_spec_evaluator.evaluate(behavioral_state, action_recipes, action_specs, action_specs_mask)

        # Q-values evaluation (action_cost + value_function(next_state))
        #Q_values = [self._approximate_value_function(state, action_recipes[i], spec) + action_costs[i]
        #            if action_specs_mask[i] else np.inf for i, spec in enumerate(action_specs)]

        valid_idx = np.where(action_specs_mask)[0]
        selected_action_index = valid_idx[action_costs[valid_idx].argmin()]
        self.logger.debug('Selected recipe: ', action_recipes[selected_action_index].__dict__)
        selected_action_spec = action_specs[selected_action_index]

        trajectory_parameters = CostBasedBehavioralPlanner._generate_trajectory_specs(behavioral_state=behavioral_state,
                                                                                      action_spec=selected_action_spec,
                                                                                      navigation_plan=nav_plan)
        visualization_message = BehavioralVisualizationMsg(reference_route=trajectory_parameters.reference_route)

        # keeping selected actions for next iteration use
        self._last_action = action_recipes[selected_action_index]
        self._last_action_spec = selected_action_spec

        baseline_trajectory = CostBasedBehavioralPlanner.generate_baseline_trajectory(
            behavioral_state.ego_state, selected_action_spec)

        print('Chosen action: %s\nSpec: %s, dist=%.2f' % (action_recipes[selected_action_index].__dict__,
            selected_action_spec.__dict__,
            selected_action_spec.s - behavioral_state.ego_state.road_localization.road_lon))

        self.logger.debug("Chosen behavioral semantic action is %s, %s",
                          action_recipes[selected_action_index].__dict__, selected_action_spec.__dict__)

        return trajectory_parameters, baseline_trajectory, visualization_message

    def _approximate_value_function(self, state: State, recipe: ActionRecipe, spec: ActionSpec) -> float:

        # create a new behavioral state at the action end
        ego = state.ego_state
        cur_ego_loc = ego.road_localization
        road_id = cur_ego_loc.road_id
        lane_width = MapService.get_instance().get_road(road_id).lane_width
        num_lanes = MapService.get_instance().get_road(road_id).lanes_num
        target_lane = cur_ego_loc.lane_num + recipe.relative_lane.value
        cpoint, yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, spec.s, target_lane * lane_width)
        new_ego = EgoState(ego.obj_id, ego.timestamp + int(spec.t * 1e9), cpoint[0], cpoint[1], cpoint[2], yaw,
                           ego.size, 0, spec.v, 0, 0, 0, 0)
        predicted_objects = []
        for obj in state.dynamic_objects:
            predicted_obj = self.predictor.predict_object_on_road(obj, np.array([ego.timestamp_in_sec + spec.t]))[0]
            predicted_objects.append(predicted_obj)

        new_state = State(None, predicted_objects, new_ego)
        new_behavioral_state = BehavioralGridState.create_from_state(new_state, self.logger)

        goal = NavigationGoal(road_id, spec.s + 300, list(range(0, num_lanes)))
        value = self.value_approximator.evaluate_state(new_behavioral_state, goal)

        return value
