from logging import Logger

import numpy as np
from decision_making.src.exceptions import NoActionsLeftForBPError
from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, AggressivenessLevel, ActionSpec, \
    ActionRecipe
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from decision_making.src.planning.types import ActionSpecArray
from decision_making.src.planning.utils.kinematics_utils import BrakingDistances
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.planning.behavioral.state import LaneMergeState


class RL_LaneMergePlanner(BasePlanner):

    def __init__(self, behavioral_state: BehavioralGridState, lane_merge_state: LaneMergeState, logger: Logger):
        super().__init__(behavioral_state, logger)
        self.lane_merge_state = lane_merge_state
        self.predictor = RoadFollowingPredictor(logger)
        self.action_space = StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING)

    def _create_actions(self) -> np.array:
        action_recipes = self.action_space.recipes

        # Recipe filtering
        recipes_mask = self.action_space.filter_recipes(action_recipes, self.behavioral_state)
        self.logger.debug('Number of actions originally: %d, valid: %d',
                          self.action_space.action_space_size, np.sum(recipes_mask))

        action_specs = np.full(len(action_recipes), None)
        valid_action_recipes = [action_recipe for i, action_recipe in enumerate(action_recipes) if recipes_mask[i]]
        action_specs[recipes_mask] = self.action_space.specify_goals(valid_action_recipes, self.behavioral_state)

        # TODO: FOR DEBUG PURPOSES!
        num_of_considered_static_actions = sum(isinstance(x, StaticActionRecipe) for x in valid_action_recipes)
        num_of_specified_actions = sum(x is not None for x in action_specs)
        self.logger.debug('Number of actions specified: %d (#%dS)',
                          num_of_specified_actions, num_of_considered_static_actions)
        return action_specs

    def _filter_actions(self, action_specs: np.array) -> ActionSpecArray:
        """
        filter out actions that either are filtered by a regular action_spec filter (of the single_lane_planner)
        or don't enable to brake before the red line
        :param action_specs: array of ActionSpec (part of actions may be None)
        :return: array of ActionSpec of the original size, with None for filtered actions
        """
        # filter actions by the regular action_spec filters of the single_lane_planner
        action_specs_mask = self.action_spec_validator.filter_action_specs(action_specs, self.behavioral_state)
        filtered_action_specs = np.full(len(action_specs), None)
        filtered_action_specs[action_specs_mask] = action_specs[action_specs_mask]
        # filter out actions that don't enable to brake before the red line
        filtered_action_specs = self._red_line_filter(filtered_action_specs)
        return filtered_action_specs

    def _red_line_filter(self, action_specs: ActionSpecArray) -> ActionSpecArray:
        """
        filter out actions that don't enable to brake before the red line
        :param action_specs: array of ActionSpec (part of actions may be None)
        :return: array of ActionSpec of the original size, with None for filtered actions
        """
        valid_specs_idxs = np.where(action_specs != None)[0]
        spec_v, spec_s = np.array([[spec.v, spec.s] for spec in action_specs[valid_specs_idxs]]).T
        w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.AGGRESSIVE]
        braking_distances = BrakingDistances.calc_actions_distances_for_given_weights(w_T, w_J, spec_v, np.zeros_like(spec_v))
        action_specs[valid_specs_idxs[spec_s + braking_distances > self.lane_merge_state.red_line_s]] = None
        return action_specs

    def _evaluate(self, action_specs: ActionSpecArray) -> np.ndarray:
        """
        Given RL policy, current state and actions mask, return the actions cost by operating policy on the state.
        Filtered actions get the maximal cost 1.
        :param action_specs: np.array of action specs
        :return: array of actions costs: the lower the better
        """
        encoded_state = encode_state_for_policy(self.lane_merge_state)

        logits, _, _, _ = RL_policy.model({SampleBatch.CUR_OBS: np.array([encoded_state])}, [])
        actions_distribution = RL_policy.dist_class(logits[0])

        actions_distribution[action_specs == None] = 0  # set zero probability for filtered actions
        prob_sum = np.sum(actions_distribution)
        if prob_sum == 0:
            raise NoActionsLeftForBPError()
        costs = 1 - actions_distribution / prob_sum
        return costs

    def _choose_action(self, action_specs: ActionSpecArray, costs: np.array) -> [ActionRecipe, ActionSpec]:
        # choose action with the minimal cost
        selected_action_index = np.argmin(costs)
        best_action_spec = action_specs[selected_action_index]
        # convert spec.s from LaneMergeState to GFF
        best_action_spec.s += self.lane_merge_state.merge_point_in_gff
        return self.action_space.recipes[selected_action_index], best_action_spec
