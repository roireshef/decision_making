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

    @classmethod
    def create_from_state(cls, state: State, map_api: MapAPI, logger: Logger):
        """
        :return: a new and updated BehavioralState
        """
        semantic_occupancy_grid = cls._generate_semantic_occupancy_grid(ego_state=state.ego_state,
                                                                        dynamic_objects=state.dynamic_objects,
                                                                        map_api=map_api,
                                                                        logger=logger)

        return NovDemoBehavioralState(road_occupancy_grid=semantic_occupancy_grid, ego_state=state.ego_state)

    @staticmethod
    def _generate_semantic_occupancy_grid(ego_state: EgoState, dynamic_objects: List[
        DynamicObject], map_api: MapAPI, logger: Logger) -> RoadSemanticOccupancyGrid:
        """
        Occupy the occupancy grid.
        This method iterates over all dynamic objects, and fits them into the relevant cell
        in the semantic occupancy grid (semantic_lane, semantic_lon).
        Each cell holds a list of objects that are within the cell borders.
        In this particular implementation, we keep up to one dynamic object per cell, which is the closest ego.
         (e.g. in the cells in front of ego, we keep objects with minimal longitudinal distance
         relative to ego front, while in all other cells we keep the object with the maximal longitudinal distance from
         ego front).
        :param ego_state:
        :param dynamic_objects:
        :return: road semantic occupancy grid
        """

        default_navigation_plan = map_api.get_road_based_navigation_plan(
            current_road_id=ego_state.road_localization.road_id)

        ego_lane = ego_state.road_localization.lane_num

        semantic_occupancy_dict: RoadSemanticOccupancyGrid = dict()
        for dynamic_object in dynamic_objects:

            object_relative_localization = dynamic_object.get_relative_road_localization(
                ego_road_localization=ego_state.road_localization, ego_nav_plan=default_navigation_plan,
                map_api=map_api, logger=logger)
            object_lon_dist = object_relative_localization.rel_lon
            object_dist_from_front = object_lon_dist - ego_state.size.length
            object_relative_lane = int(dynamic_object.road_localization.lane_num - ego_lane)

            # Determine cell index in occupancy grid
            if object_relative_lane == 0:
                # Object is on same lane as ego
                if object_dist_from_front > 0.0:
                    # Object in front of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_FRONT)

                else:
                    # Object behind vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_BEHIND)

            elif object_relative_lane == 1 or object_relative_lane == -1:
                # Object is one lane on the left/right

                if object_dist_from_front > SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO:
                    # Object in front of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_FRONT)

                elif object_lon_dist > -1 * SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO:
                    # Object vehicle aside of ego
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_ASIDE)

                else:
                    # Object behind rear of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_BEHIND)

            # Add object to occupancy grid
            # keeping only a single dynamic object per cell. List is used for future dev.
            if occupancy_index not in semantic_occupancy_dict:
                # add to occupancy grid
                semantic_occupancy_dict[occupancy_index] = [dynamic_object]
            else:
                object_in_cell = semantic_occupancy_dict[occupancy_index][0]
                object_in_grid_lon_dist = object_in_cell.get_relative_road_localization(
                    ego_road_localization=ego_state.road_localization,
                    ego_nav_plan=default_navigation_plan,
                    map_api=map_api, logger=logger).rel_lon
                object_in_grid_dist_from_front = object_in_grid_lon_dist - ego_state.size.length

                if occupancy_index[1] == SEMANTIC_GRID_FRONT:
                    # take the object with least lon
                    if object_lon_dist < object_in_grid_dist_from_front:
                        # replace object the the closer one
                        semantic_occupancy_dict[occupancy_index][0] = dynamic_object
                else:
                    # Assumption - taking the object with the largest long even in the ASIDE cells
                    # take the object with largest lon
                    if object_lon_dist > object_in_grid_dist_from_front:
                        # replace object the the closer one
                        semantic_occupancy_dict[occupancy_index][0] = dynamic_object

        return RoadSemanticOccupancyGrid(semantic_occupancy_dict)


class NovDemoPolicy(SemanticActionsPolicy):
    def _enumerate_actions(self, behavioral_state: SemanticBehavioralState) -> List[SemanticAction]:
        """
        Enumerate the list of possible semantic actions to be generated.
        :param behavioral_state:
        :return:
        """

        semantic_actions: List[SemanticAction] = list()

        # Generate actions towards each of the cells in front of ego
        for relative_lane_key in [-1, 0, 1]:
            for longitudinal_key in [SEMANTIC_GRID_FRONT]:
                semantic_cell = (relative_lane_key, longitudinal_key)
                if semantic_cell in behavioral_state.road_occupancy_grid:
                    # Select first (closest) object in cell
                    target_obj = behavioral_state.road_occupancy_grid[semantic_cell][0]
                else:
                    # There are no objects in cell
                    target_obj = None

                semantic_action = SemanticAction(cell=semantic_cell, target_obj=target_obj,
                                                 action_type=SemanticActionType.FOLLOW)

                semantic_actions.append(semantic_action)

        return semantic_actions

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
