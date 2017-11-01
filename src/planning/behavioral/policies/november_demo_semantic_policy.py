import numpy as np
from typing import List

from logging import Logger
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.policy import PolicyConfig
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticActionsPolicy, SemanticAction, \
    SemanticActionSpec, SEMANTIC_CELL_LANE
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State
from mapping.src.model.map_api import MapAPI

SEMANTIC_GRID_FRONT, SEMANTIC_GRID_ASIDE, SEMANTIC_GRID_BEHIND = 1, 0, -1
GRID_MID = 10

# The margin that we take from the front/read of the vehicle to define the front/rear partitions
SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO = 1


MIN_OVERTAKE_VEL = 2  # [m/s]
MIN_CHANGE_LANE_TIME = 4  # sec
MAX_CHANGE_LANE_TIME = 6  # sec

class NovDemoPolicy(SemanticActionsPolicy):

    @staticmethod
    def _eval_actions(behavioral_state: NovDemoBehavioralState, semantic_actions: List[SemanticAction],
                      actions_spec: List[SemanticActionSpec], max_velocity: float) -> np.ndarray:
        """
        Evaluate the generated actions using the full state.
        Gets a list of actions to evaluate so and returns a vector representing their costs.
        A set of actions is provided, enabling assessing them dependently.
        Note: the semantic actions were generated using the behavioral state which isn't necessarily captures
         all relevant details in the scene. Therefore the evaluation is done using the full state.
        :param state: world state
        :param semantic_actions: semantic actions list
        :param actions_spec: specifications of semantic actions
        :return: numpy array of costs of semantic actions
        """

        straight_lane_ind = NovDemoPolicy._get_action_ind_by_lane(semantic_actions, actions_spec, 0, state.ego.v_x)
        left_lane_ind = NovDemoPolicy._get_action_ind_by_lane(semantic_actions, actions_spec, 1, state.ego.v_x)
        right_lane_ind = NovDemoPolicy._get_action_ind_by_lane(semantic_actions, actions_spec, -1, state.ego.v_x)

        desired_vel = max_velocity
        if state.ego.v_x < actions_spec[straight_lane_ind].v - MIN_OVERTAKE_VEL:
            desired_vel = state.ego.v_x

        right_is_fast = right_lane_ind is not None and desired_vel - actions_spec[right_lane_ind].v < MIN_OVERTAKE_VEL
        right_is_far = right_lane_ind is not None and actions_spec[right_lane_ind].t > MAX_CHANGE_LANE_TIME
        straight_is_fast = straight_lane_ind is not None and \
                           desired_vel - actions_spec[straight_lane_ind].v < MIN_OVERTAKE_VEL
        straight_is_far = straight_lane_ind is not None and actions_spec[straight_lane_ind].t > MAX_CHANGE_LANE_TIME
        left_is_faster = left_lane_ind is not None and (straight_lane_ind is None or
                            actions_spec[left_lane_ind].v - actions_spec[straight_lane_ind].v >= MIN_OVERTAKE_VEL)
        left_is_far = left_lane_ind is not None and actions_spec[left_lane_ind].t > MAX_CHANGE_LANE_TIME

        costs = np.zeros(len(semantic_actions))
        # move right if both straight and right lanes are not slow
        if (right_is_fast or right_is_far) and (straight_is_fast or straight_is_far or straight_lane_ind is None):
            costs[right_lane_ind] = 1.
            return costs
        # move left if straight is slow and the left is faster than straight
        if not (straight_is_fast or straight_is_far) and \
                (left_is_faster or left_is_far or straight_lane_ind is None):
            costs[left_lane_ind] = 1.
            return costs
        costs[straight_lane_ind] = 1.
        return costs

    @staticmethod
    def _get_action_ind_by_lane(semantic_actions: List[SemanticAction], actions_spec: List[SemanticActionSpec],
                                cell_lane: int, cur_vel: float):
        action_ind = [i for i, action in enumerate(semantic_actions) if action.cell[SEMANTIC_CELL_LANE] == cell_lane]
        if len(action_ind) > 0 and (semantic_actions[action_ind[0]].cell[SEMANTIC_CELL_LANE] == 0 or
                                        actions_spec[action_ind[0]].t > MIN_CHANGE_LANE_TIME):
            return action_ind[0]
        else:
            return None
