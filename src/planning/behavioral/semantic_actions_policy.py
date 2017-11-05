from enum import Enum
from logging import Logger

import numpy as np
from typing import Dict, List, Optional, Tuple

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.policy import Policy, PolicyConfig
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State, DynamicObject
from mapping.src.model.map_api import MapAPI


class SemanticActionType(Enum):
    FOLLOW = 1


# Define semantic cell
SEMANTIC_CELL_LANE, SEMANTIC_CELL_LON = 0, 1
SemanticGridCell = Tuple[int, int]
"""
We assume that the road is partitioned into semantic areas, each area is defined as a cell.
The keys are:
- relative_lane: describes the lane number, relative to ego. For example: {-1, 0, 1}
- relative_lon:  describes the longitudinal partition of the grid.
  For example, the grid can be partitioned in the following way:
  {-1: behind ego, 0: aside, 1: infront of ego}.

"""

# Define semantic occupancy grid
RoadSemanticOccupancyGrid = Dict[SemanticGridCell, List[DynamicObject]]
"""
This type holds a semantic occupancy grid, that maps dynamic objects in the scene
to cells (partitions) of the state space.
"""


class SemanticBehavioralState(BehavioralState):

    def __init__(self, road_occupancy_grid: RoadSemanticOccupancyGrid):
        """
        :param road_occupancy_grid: A dictionary that maps semantic cells to list of dynamic objects.
        """
        self.road_occupancy_grid = road_occupancy_grid

    @classmethod
    def create_from_state(cls, state: State, map_api: MapAPI, logger: Logger):
        """
        :return: a new and updated BehavioralState
        """
        pass


class SemanticAction:
    """
    Enumeration of a semantic action relative to a given cell and object
    """

    def __init__(self, cell: SemanticGridCell, target_obj: Optional[DynamicObject], action_type: SemanticActionType):
        """
        :param cell: which cell in the semantic grid
        :param target_obj: which object is related to this action
        :param action_type: enumerated type
        """
        self.cell = cell
        self.target_obj = target_obj
        self.action_type = action_type


class SemanticActionSpec:
    """
    Holds the actual translation of the semantic action in terms of trajectory specifications.
    """

    def __init__(self, t: float, v: float, s_rel: float, d_rel: float):
        """
        The trajectory specifications are defined by the target ego state
        :param t: time [sec]
        :param v: velocity [m/s]
        :param s_rel: relative longitudinal distance to ego in Frenet frame [m]
        :param d_rel: relative lateral distance to ego in Frenet frame [m]
        """
        self.t = t
        self.v = v
        self.s_rel = s_rel
        self.d_rel = d_rel


class SemanticActionsPolicy(Policy):
    def __init__(self, logger: Logger, policy_config: PolicyConfig, predictor: Predictor, map_api: MapAPI):
        """
        Receives configuration and logger
        :param logger: logger
        :param policy_config: parameters configuration class, loaded from parameter server
        :param predictor: used for predicting ego and other dynamic objects in future states
        :param map_api: Map API
        """
        self._map_api = map_api
        self._policy_config = policy_config
        self._predictor = predictor
        self.logger = logger

    def plan(self, state: State, nav_plan: NavigationPlanMsg) -> (TrajectoryParams, BehavioralVisualizationMsg):
        """
        Generate plan from given state
        :param state:
        :param nav_plan:
        :return:
        """
        pass

    def _enumerate_actions(self, behavioral_state: SemanticBehavioralState) -> List[SemanticAction]:
        """
        Enumerate the list of possible semantic actions to be generated.
        :param behavioral_state:
        :return:
        """
        pass

    def _specify_action(self, behavioral_state: SemanticBehavioralState,
                         semantic_action: SemanticAction) -> SemanticActionSpec:
        """
        For each semantic actions, generate a trajectory specifications that will be passed through to the TP
        :param behavioral_state:
        :param semantic_action:
        :return: semantic action spec
        """
        pass

    def _eval_actions(self, behavioral_state: SemanticBehavioralState, semantic_actions: List[SemanticAction],
                      actions_spec: List[SemanticActionSpec]) -> np.ndarray:
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
        pass

    def _select_best(self, action_specs: List[SemanticActionSpec], costs: np.ndarray) -> int:
        """
        Select the best action out of the possible actions specs considering their cost
        :param action_specs:
        :param costs:
        :return:
        """
        return int(np.argmax(costs)[0])
