from typing import List
import numpy as np
from logging import Logger

from decision_making.src.exceptions import BehavioralPlanningException
from decision_making.src.global_constants import SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LON_FRONT, \
    SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LAT_RIGHT, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, MIN_OVERTAKE_VEL, \
    SEMANTIC_CELL_LON_SAME, SEMANTIC_CELL_LON_REAR, SAFE_DIST_TIME_DELAY, LON_ACC_LIMITS
from decision_making.src.planning.behavioral.architecture.components.evaluators.state_action_evaluator import \
    StateActionEvaluator
from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec, ActionRecipe, \
    SemanticGridCell, LAT_CELL
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_state import SemanticActionsGridState
from decision_making.src.planning.types import FrenetPoint, FP_SX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from mapping.src.service.map_service import MapService


class RuleBasedStateActionEvaluator(StateActionEvaluator):

    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate_recipe(self, behavioral_state: SemanticActionsGridState, action_recipe: ActionRecipe) -> float:
        pass

    def evaluate_action_spec(self, behavioral_state: SemanticActionsGridState, action_spec: ActionSpec) -> float:
        pass

    # This method is hre for now just for backward compatibility and will be removed after some content will be
    # inserted into the other interface methods
    def evaluate(self, behavioral_state: SemanticActionsGridState,
                 action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec],
                 action_specs_mask: List[bool]) -> np.ndarray:
        """
        Evaluate the generated actions using the actions' spec and SemanticBehavioralState containing semantic grid.
        Gets a list of actions to evaluate and returns a vector representing their costs.
        A set of actions is provided, enabling us to assess them independently.
        Note: the semantic actions were generated using the behavioral state and don't necessarily capture
         all relevant details in the scene. Therefore the evaluation is done using the behavioral state.
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_recipes: semantic actions list.
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """

        if len(action_recipes) != len(action_specs):
            self.logger.error(
                "The input arrays have different sizes: len(semantic_actions)=%d, len(actions_spec)=%d",
                len(action_recipes), len(action_specs))
            raise BehavioralPlanningException(
                "The input arrays have different sizes: len(semantic_actions)=%d, len(actions_spec)=%d",
                len(action_recipes), len(action_specs))

        # get indices of semantic_actions array for 3 actions: goto-right, straight, goto-left
        current_lane_action_ind = RuleBasedStateActionEvaluator._get_action_ind(
            action_recipes, action_specs_mask, (SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LON_FRONT))
        left_lane_action_ind = RuleBasedStateActionEvaluator._get_action_ind(
            action_recipes, action_specs_mask, (SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LON_FRONT))
        right_lane_action_ind = RuleBasedStateActionEvaluator._get_action_ind(
            action_recipes, action_specs_mask, (SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LON_FRONT))

        # The cost for each action is assigned so that the preferred policy would be:
        # Go to right if right and current lanes are fast enough.
        # Go to left if the current lane is slow and the left lane is faster than the current.
        # Otherwise remain on the current lane.

        # TODO - this needs to come from map
        desired_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED

        # boolean whether the forward-right cell is fast enough (may be empty grid cell)
        is_forward_right_fast = right_lane_action_ind is not None and \
                                desired_vel - action_specs[right_lane_action_ind].v < MIN_OVERTAKE_VEL
        # boolean whether the right cell near ego is occupied
        is_right_occupied = True
        if (SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LON_SAME) in behavioral_state.road_occupancy_grid:
            is_right_occupied = len(behavioral_state.road_occupancy_grid[(SEMANTIC_CELL_LAT_RIGHT,
                                                                          SEMANTIC_CELL_LON_SAME)]) > 0

        # boolean whether the forward cell is fast enough (may be empty grid cell)
        is_forward_fast = current_lane_action_ind is not None and \
                          desired_vel - action_specs[current_lane_action_ind].v < MIN_OVERTAKE_VEL

        # boolean whether the forward-left cell is faster than the forward cell
        is_forward_left_faster = left_lane_action_ind is not None and \
                                 (current_lane_action_ind is None or
                                  action_specs[left_lane_action_ind].v - action_specs[current_lane_action_ind].v >=
                                  MIN_OVERTAKE_VEL)

        ego = behavioral_state.ego_state
        road_id = ego.road_localization.road_id
        road_points = MapService.get_instance()._shift_road_points_to_latitude(road_id, 0.0)
        road_frenet = FrenetSerret2DFrame(road_points)
        ego_fpoint = road_frenet.cpoint_to_fpoint(np.array([ego.x, ego.y]))

        dist_to_backleft, safe_left_dist_behind_ego = RuleBasedStateActionEvaluator._calc_safe_dist_behind_ego(
            behavioral_state, road_frenet, ego_fpoint, SEMANTIC_CELL_LAT_LEFT)
        dist_to_backright, safe_right_dist_behind_ego = RuleBasedStateActionEvaluator._calc_safe_dist_behind_ego(
            behavioral_state, road_frenet, ego_fpoint, SEMANTIC_CELL_LAT_RIGHT)

        self.logger.debug("Distance\safe distance to back left car: %s\%s.", dist_to_backleft,
                          safe_left_dist_behind_ego)
        self.logger.debug("Distance\safe distance to back right car: %s\%s.", dist_to_backright,
                          safe_right_dist_behind_ego)

        # boolean whether the left cell near ego is occupied
        if (SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LON_SAME) in behavioral_state.road_occupancy_grid:
            is_left_occupied = len(behavioral_state.road_occupancy_grid[(SEMANTIC_CELL_LAT_LEFT,
                                                                         SEMANTIC_CELL_LON_SAME)]) > 0
        else:
            is_left_occupied = True

        costs = np.ones(len(action_recipes))

        # move right if both straight and right lanes are fast
        # if is_forward_right_fast and (is_forward_fast or current_lane_action_ind is None) and not is_right_occupied:
        if is_forward_right_fast and not is_right_occupied and dist_to_backright > safe_right_dist_behind_ego:
            costs[right_lane_action_ind] = 0.
        # move left if straight is slow and the left is faster than straight
        elif not is_forward_fast and (
                is_forward_left_faster or current_lane_action_ind is None) and not is_left_occupied and \
                dist_to_backleft > safe_left_dist_behind_ego:
            costs[left_lane_action_ind] = 0.
        else:
            costs[current_lane_action_ind] = 0.
        return costs

    @staticmethod
    def _calc_safe_dist_behind_ego(behavioral_state: SemanticActionsGridState, road_frenet: FrenetSerret2DFrame,
                                   ego_fpoint: FrenetPoint, semantic_cell_lat: int) -> [float, float]:
        """
        Calculate both actual and safe distances between rear object and ego on the left side or right side.
        If there is no object, return actual dist = inf and safe dist = 0.
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :param road_frenet: road Frenet frame for ego's road_id
        :param ego_fpoint: frenet point of ego location
        :param semantic_cell_lat: either SEMANTIC_CELL_LAT_LEFT or SEMANTIC_CELL_LAT_RIGHT
        :return: longitudinal distance between ego and rear object, safe distance between ego and the rear object
        """
        dist_to_back_obj = np.inf
        safe_dist_behind_ego = 0
        back_objects = []
        if (semantic_cell_lat, SEMANTIC_CELL_LON_REAR) in behavioral_state.road_occupancy_grid:
            back_objects = behavioral_state.road_occupancy_grid[(semantic_cell_lat, SEMANTIC_CELL_LON_REAR)]
        if len(back_objects) > 0:
            back_fpoint = road_frenet.cpoint_to_fpoint(np.array([back_objects[0].x, back_objects[0].y]))
            dist_to_back_obj = ego_fpoint[FP_SX] - back_fpoint[FP_SX]
            if behavioral_state.ego_state.v_x > back_objects[0].v_x:
                safe_dist_behind_ego = back_objects[0].v_x * SAFE_DIST_TIME_DELAY
            else:
                safe_dist_behind_ego = back_objects[0].v_x * SAFE_DIST_TIME_DELAY + \
                                       back_objects[0].v_x ** 2 / (2 * abs(LON_ACC_LIMITS[0])) - \
                                       behavioral_state.ego_state.v_x ** 2 / (2 * abs(LON_ACC_LIMITS[0]))
        return dist_to_back_obj, safe_dist_behind_ego

    @staticmethod
    def _get_action_ind(action_recipes: List[ActionRecipe], recipes_mask: List[bool], cell: SemanticGridCell):
        """
        Given semantic actions array and action cell, return index of action matching to the given cell.
        :param action_recipes: array of semantic actions
        :param cell:
        :return: the action index or None if the action does not exist
        """
        action_ind = [i for i, recipe in enumerate(action_recipes) if
                      recipe.relative_lane.value == cell[LAT_CELL] and recipes_mask[i]]
        return action_ind[0] if len(action_ind) > 0 else None
