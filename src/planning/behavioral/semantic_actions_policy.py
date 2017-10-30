from enum import Enum

import numpy as np
from typing import Dict, List, Optional

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.policy import Policy
from decision_making.src.state.state import State, DynamicObject


class SemanticActionType(Enum):
    FOLLOW = 1


class SemanticGridCell:
    """
    We assume that the road is partitioned into semantic areas, each area is defined as a cell.
    """

    def __init__(self, relative_lane: int, relative_lon: int):
        """
        :param relative_lane: describes the lane number, relative to ego. For example: {-1.0, 0.0, 1.0}
        :param relative_lon:  describes the longitudinal partition of the grid.
            For example, the grid can be partitioned in the following way:
            {-1.0: behind ego, 0.0: aside, 1.0: infront of ego}.
        """
        self.relative_lane = relative_lane
        self.relative_lon = relative_lon


class SemanticBehavioralState(BehavioralState):
    """
    This class holds a semantic occupancy grid, that maps dynamic objects in the scene
    to cells (partitions) of the state space.
    """

    def __init__(self, road_occupancy_grid: Dict[SemanticGridCell, List[DynamicObject]]):
        """
        :param road_occupancy_grid: A dictionary that maps semantic cells to list of dynamic objects.
        """
        self.road_occupancy_grid = road_occupancy_grid

    def update_behavioral_state(self, state: State):
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

    def plan(self, state: State, nav_plan: NavigationPlanMsg):
        self._behavioral_state = self._behavioral_state.update_behavioral_state(state, nav_plan)

    def _enumerate_actions(self, behavioral_state: SemanticBehavioralState) -> List[SemanticAction]:
        pass

    def _specify_actions(self, behavioral_state: SemanticBehavioralState,
                         semantic_actions: SemanticAction) -> SemanticActionSpec:
        pass

    def _eval_actions(self, state: State, action_spec: SemanticActionSpec) -> float:
        """
        Evaluate the generated actions using the full state.
        Note: the semantic actions were generated using the behavioral state which isn't necessarily captures
         all relevant details in the scene. Therefore the evaluation is done using the full state.
        :param state: world state
        :param action_spec: specification of semantic action
        :return: cost of semantic action
        """
        pass
