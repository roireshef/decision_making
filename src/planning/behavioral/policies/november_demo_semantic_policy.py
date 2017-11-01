import numpy as np
from logging import Logger
from typing import List

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT
from decision_making.src.planning.behavioral.constants import BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticActionsPolicy, \
    SemanticBehavioralState, RoadSemanticOccupancyGrid, SemanticAction, SemanticActionType, SemanticActionSpec, \
    SEMANTIC_CELL_LANE
from decision_making.src.state.state import DynamicObject, EgoState, State
from mapping.src.model.map_api import MapAPI

SEMANTIC_GRID_FRONT, SEMANTIC_GRID_ASIDE, SEMANTIC_GRID_BEHIND = 1, 0, -1
GRID_MID = 10

# The margin that we take from the front/read of the vehicle to define the front/rear partitions
SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO = 1


class NovDemoBehavioralState(SemanticBehavioralState):
    def __init__(self, road_occupancy_grid: RoadSemanticOccupancyGrid, ego_state: EgoState):
        super().__init__(road_occupancy_grid=road_occupancy_grid)
        self.ego_state = ego_state


class NovDemoPolicy(SemanticActionsPolicy):

    def _specify_action(self, behavioral_state: NovDemoBehavioralState,
                        semantic_action: SemanticAction) -> SemanticActionSpec:
        """
        For each semantic actions, generate a trajectory specifications that will be passed through to the TP
        :param behavioral_state:
        :param semantic_action:
        :return: semantic action spec
        """

        if semantic_action.target_obj is None:
            return self._specify_action_to_empty_cell(behavioral_state=behavioral_state,
                                                      semantic_action=semantic_action)
        else:
            return self._specify_action_towards_object(behavioral_state=behavioral_state,
                                                       semantic_action=semantic_action)

    def _specify_action_to_empty_cell(self, behavioral_state: NovDemoBehavioralState,
                                      semantic_action: SemanticAction) -> SemanticActionSpec:
        """
        Generate trajectory specification towards a target location in given cell considering ego speed, location.
        :param behavioral_state:
        :param semantic_action:
        :return:
        """

        road_lane_latitudes = self._map_api.get_center_lanes_latitudes(
            road_id=behavioral_state.ego_state.road_localization.road_id)
        target_lane = behavioral_state.ego_state.road_localization.lane_num + semantic_action.cell[SEMANTIC_CELL_LANE]
        target_lane_latitude = road_lane_latitudes[target_lane]

        target_relative_s = target_lane_latitude - behavioral_state.ego_state.road_localization.full_lat
        target_relative_d = BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT * BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON

        return SemanticActionSpec(t=BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON, v=BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT,
                                  s_rel=target_relative_s, d_rel=target_relative_d)

    def _specify_action_towards_object(self, behavioral_state: NovDemoBehavioralState,
                                       semantic_action: SemanticAction) -> SemanticActionSpec:

        # Get object's location
        default_navigation_plan = self._map_api.get_road_based_navigation_plan(
            current_road_id=behavioral_state.ego_state.road_localization.road_id)

        object_relative_road_localization = semantic_action.target_obj.get_relative_road_localization(
            ego_road_localization=behavioral_state.ego_state.road_localization, ego_nav_plan=default_navigation_plan,
            map_api=self._map_api, logger=self.logger)

        road_lane_latitudes = self._map_api.get_center_lanes_latitudes(
            road_id=behavioral_state.ego_state.road_localization.road_id)
        target_lane = behavioral_state.ego_state.road_localization.lane_num + semantic_action.cell[SEMANTIC_CELL_LANE]
        target_lane_latitude = road_lane_latitudes[target_lane]

        target_relative_s = target_lane_latitude - behavioral_state.ego_state.road_localization.full_lat
        target_relative_d = BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT * BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON

        # TODO: Define safe distance from object using ACDA
        # TODO: add prediction
        safe_distance = 10.0

        # Clip target longitudinal location between 0.0 and the safe distance behind the car
        target_relative_d = float(
            np.clip(target_relative_d, 0.0, object_relative_road_localization.rel_lon - safe_distance)[0])

        return SemanticActionSpec(t=BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON, v=BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT,
                                  s_rel=target_relative_s, d_rel=target_relative_d)
