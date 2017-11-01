import numpy as np
from typing import List

from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticActionsPolicy, SemanticAction, \
    SemanticActionSpec, SEMANTIC_CELL_LANE, SemanticBehavioralState, RoadSemanticOccupancyGrid
from decision_making.src.state.state import EgoState

SEMANTIC_GRID_FRONT, SEMANTIC_GRID_ASIDE, SEMANTIC_GRID_BEHIND = 1, 0, -1
GRID_MID = 10

# The margin that we take from the front/read of the vehicle to define the front/rear partitions
SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO = 1


MIN_OVERTAKE_VEL = 2  # [m/s]
MIN_CHANGE_LANE_TIME = 4  # sec
LONG_ACHEIVING_TIME = 10  # sec

class NovDemoBehavioralState(SemanticBehavioralState):
    def __init__(self, road_occupancy_grid: RoadSemanticOccupancyGrid, ego_state: EgoState):
        super().__init__(road_occupancy_grid=road_occupancy_grid)
        self.ego_state = ego_state

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

        straight_lane_ind = NovDemoPolicy._get_action_ind_by_lane(semantic_actions, actions_spec, 0)
        left_lane_ind = NovDemoPolicy._get_action_ind_by_lane(semantic_actions, actions_spec, 1)
        right_lane_ind = NovDemoPolicy._get_action_ind_by_lane(semantic_actions, actions_spec, -1)

        desired_vel = max_velocity
        if behavioral_state.ego_state.v_x < actions_spec[straight_lane_ind].v - MIN_OVERTAKE_VEL:
            desired_vel = behavioral_state.ego_state.v_x

        right_is_fast = right_lane_ind is not None and desired_vel - actions_spec[right_lane_ind].v < MIN_OVERTAKE_VEL
        right_is_very_far = right_lane_ind is not None and actions_spec[right_lane_ind].t > LONG_ACHEIVING_TIME
        right_is_occupied = len(behavioral_state.road_occupancy_grid[(-1, 0)]) > 0
        straight_is_fast = straight_lane_ind is not None and \
                           desired_vel - actions_spec[straight_lane_ind].v < MIN_OVERTAKE_VEL
        straight_is_far = straight_lane_ind is not None and actions_spec[straight_lane_ind].t > MIN_CHANGE_LANE_TIME
        left_is_faster = left_lane_ind is not None and (straight_lane_ind is None or
                            actions_spec[left_lane_ind].v - actions_spec[straight_lane_ind].v >= MIN_OVERTAKE_VEL)
        left_is_very_far = left_lane_ind is not None and actions_spec[left_lane_ind].t > LONG_ACHEIVING_TIME
        left_is_occupied = len(behavioral_state.road_occupancy_grid[(1, 0)]) > 0

        costs = np.zeros(len(semantic_actions))
        # move right if both straight and right lanes are fast or far
        if (right_is_fast or right_is_very_far) and \
                (straight_is_fast or straight_is_far or straight_lane_ind is None) and \
                not right_is_occupied:
            costs[right_lane_ind] = 1.
            return costs
        # move left if straight is slow and close and the left is far or faster than straight
        if not (straight_is_fast or straight_is_far) and \
                (left_is_faster or left_is_very_far or straight_lane_ind is None) and \
                not left_is_occupied:
            costs[left_lane_ind] = 1.
            return costs
        costs[straight_lane_ind] = 1.
        return costs

    @staticmethod
    def _get_action_ind_by_lane(semantic_actions: List[SemanticAction], actions_spec: List[SemanticActionSpec],
                                cell_lane: int):
        action_ind = [i for i, action in enumerate(semantic_actions) if action.cell[SEMANTIC_CELL_LANE] == cell_lane]
        if len(action_ind) > 0 and (semantic_actions[action_ind[0]].cell[SEMANTIC_CELL_LANE] == 0 or
                                        actions_spec[action_ind[0]].t > MIN_CHANGE_LANE_TIME):
            return action_ind[0]
        else:
            return None
