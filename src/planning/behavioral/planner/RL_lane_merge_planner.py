from logging import Logger

import numpy as np
from decision_making.src.exceptions import NoActionsLeftForBPError
from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, AggressivenessLevel, ActionSpec, \
    ActionRecipe
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING, \
    DEFAULT_ACTION_SPEC_FILTERING
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from decision_making.src.planning.types import ActionSpecArray, FS_SX
from decision_making.src.planning.utils.kinematics_utils import BrakingDistances
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState
from ray.rllib.evaluation import SampleBatch
from pathlib import Path
import torch
from gym.spaces.tuple import Tuple as GymTuple
from gym.spaces.box import Box

from planning_research.src.flow_rl.models.simple_model import SimpleModel  #TODO: remove dependence on planning_research


class RL_LaneMergePlanner(BasePlanner):

    def __init__(self, lane_merge_state: LaneMergeState, logger: Logger):
        super().__init__(lane_merge_state, logger)
        self.predictor = RoadFollowingPredictor(logger)
        self.action_space = StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING)

        # TODO: use global constant for the model path
        checkpoint_path = str(Path.home()) + '/ray_results/checkpoint.torch'
        model_state_dict = torch.load(checkpoint_path)

        # TODO: create global constants for observation space initialization
        ego_box = Box(low=-np.inf, high=np.inf, shape=(1, 3), dtype=np.float32)
        actors_box = Box(low=-np.inf, high=np.inf, shape=(54, 2), dtype=np.float32)
        obs_space = GymTuple((ego_box, actors_box))
        options = {"custom_options": {"hidden_size": 64}}
        self.model = SimpleModel(obs_space=obs_space, num_outputs=6, options=options)
        self.model.load_state_dict(model_state_dict)

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
        action_specs_mask = DEFAULT_ACTION_SPEC_FILTERING.filter_action_specs(action_specs, self.behavioral_state)
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
        valid_specs_idxs = np.where(action_specs.astype(bool))[0]
        spec_v, spec_s = np.array([[spec.v, spec.s] for spec in action_specs[valid_specs_idxs]]).T
        w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.AGGRESSIVE]
        braking_distances = BrakingDistances.calc_actions_distances_for_given_weights(w_T, w_J, spec_v, np.zeros_like(spec_v))
        action_specs[valid_specs_idxs[spec_s + braking_distances > self.lane_merge_state.red_line_s]] = None
        return action_specs

    def _evaluate_actions(self, action_specs: ActionSpecArray) -> np.ndarray:
        """
        Given RL policy, current state and actions mask, return the actions cost by operating policy on the state.
        Filtered actions get the maximal cost 1.
        :param action_specs: np.array of action specs
        :return: array of actions costs: the lower the better
        """
        if not action_specs.astype(bool).any():
            raise NoActionsLeftForBPError("All actions were filtered in BP. timestamp_in_sec: %f" %
                                          self.behavioral_state.ego_state.timestamp_in_sec)

        encoded_state = self.lane_merge_state.encode_state_for_RL()

        logits = self.model._forward({SampleBatch.CUR_OBS: encoded_state}, [])[0]
        logits[~action_specs.astype(bool)] = -np.inf
        chosen_action_idx = np.argmax(logits)

        costs = np.full(len(action_specs), 1)
        costs[chosen_action_idx] = 0
        return costs

    def _choose_action(self, action_specs: ActionSpecArray, costs: np.array) -> [ActionRecipe, ActionSpec]:
        # choose action with the minimal cost
        selected_action_index = int(np.argmin(costs))
        best_action_spec = action_specs[selected_action_index]
        # convert spec.s from LaneMergeState to GFF
        best_action_spec.s += self.lane_merge_state.ego_state.map_state.lane_fstate[FS_SX]
        return self.action_space.recipes[selected_action_index], best_action_spec
