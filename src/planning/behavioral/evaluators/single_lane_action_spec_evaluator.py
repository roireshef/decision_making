from decision_making.src.planning.types import C_V, FS_SX, FS_SA, FS_SV
from logging import Logger
from typing import List

import numpy as np

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, ActionType, RelativeLane, \
    StaticActionRecipe, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.evaluators.action_evaluator import \
    ActionSpecEvaluator


class SingleLaneActionSpecEvaluator(ActionSpecEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate(self, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec], action_specs_mask: List[bool]) -> np.ndarray:
        """
        Evaluates Action-Specifications based on the following logic:
        * Only takes into account actions on RelativeLane.SAME_LANE
        * If there's a leading vehicle, try following it (ActionType.FOLLOW_LANE, lowest aggressiveness possible)
        * If no action from the previous bullet is found valid, find the ActionType.FOLLOW_LANE action with maximal
        allowed velocity and lowest aggressiveness possible.
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_recipes: semantic actions list.
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """
        costs = np.full(len(action_recipes), 1)

        # TODO: remove it
        ego = behavioral_state.ego_state
        ego_fstate = ego.map_state.lane_fstate
        if (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT) in behavioral_state.road_occupancy_grid:
            obj = behavioral_state.road_occupancy_grid[(RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)][0].dynamic_object
            obj_fstate = obj.map_state.lane_fstate
            s_T = np.linalg.norm(obj.cartesian_state[:2] - ego.cartesian_state[:2]) - \
                   0.5 * (ego.size.length + obj.size.length) - LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
            headway = s_T / ego.cartesian_state[C_V]
            print('BP time %.3f: sva=(%.2f, %.2f, %.2f) dvel=%.2f s_T=%.2f headway=%.2f' %
                  (ego.timestamp_in_sec, ego_fstate[FS_SX], ego_fstate[FS_SV], ego_fstate[FS_SA],
                   ego_fstate[FS_SV] - obj_fstate[FS_SV], s_T, headway))
        else:  # no front object
            print('BP time %.3f: sva=(%.2f, %.2f, %.2f)' %
                  (ego.timestamp_in_sec, ego_fstate[FS_SX], ego_fstate[FS_SV], ego_fstate[FS_SA]))

        # first try to find a valid dynamic action for SAME_LANE
        follow_vehicle_valid_action_idxs = np.array([[i, behavioral_state.marginal_safety_per_action[i], recipe]
                                                     for i, recipe in enumerate(action_recipes)
                                                     if action_specs_mask[i]
                                                     and recipe.relative_lane == RelativeLane.SAME_LANE
                                                     and recipe.action_type == ActionType.FOLLOW_VEHICLE])
        if len(follow_vehicle_valid_action_idxs) > 0:
            best_idx = np.argmax(follow_vehicle_valid_action_idxs[:, 1])
            min_safe_dist = follow_vehicle_valid_action_idxs[best_idx, 1]

            # TODO: remove it
            # for line in follow_vehicle_valid_action_idxs:
            #     print('min_safe_dist: %.2f (t=%.2f) %s' % (line[1], action_specs[line[0]].t, line[2].aggressiveness))

            if min_safe_dist > 5:   # safe enough
                chosen_idx = 0      # calm action
            elif len(follow_vehicle_valid_action_idxs) == 3 and min_safe_dist > 3:  # moderate safety
                chosen_idx = 1      # standard action
            else:                   # low safety
                chosen_idx = -1     # aggressive action
            spec_idx = int(follow_vehicle_valid_action_idxs[chosen_idx, 0])
            costs[spec_idx] = 0  # choose the found dynamic action
            # TODO: remove it
            print('evaluator: chosen spec: %s' % (action_specs[spec_idx]))
            return costs

        # if a dynamic action not found, calculate maximal valid existing velocity for same-lane static actions
        terminal_velocities = np.unique([recipe.velocity for i, recipe in enumerate(action_recipes)
                                         if action_specs_mask[i] and isinstance(recipe, StaticActionRecipe)
                                         and recipe.relative_lane == RelativeLane.SAME_LANE])
        maximal_allowed_velocity = max(terminal_velocities[terminal_velocities <= BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED])

        # find the most calm same-lane static action with the maximal existing velocity
        follow_lane_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                         if action_specs_mask[i] and isinstance(recipe, StaticActionRecipe)
                                         and recipe.relative_lane == RelativeLane.SAME_LANE
                                         and recipe.velocity == maximal_allowed_velocity]

        # If there is a front car and all dynamic actions were filtered, choose aggressive action. Else calm action.
        if (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT) in behavioral_state.road_occupancy_grid:
            static_action_idx = -1  # the most aggressive
        else:  # no leading vehicle
            static_action_idx = 0  # the most calm
        costs[follow_lane_valid_action_idxs[static_action_idx]] = 0  # choose the found static action
        return costs
