from collections import defaultdict
from enum import Enum
from logging import Logger
from typing import Dict, List, Tuple

import numpy as np

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_LOOKAHEAD_DIST, \
    BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
from decision_making.src.global_constants import LON_MARGIN_FROM_EGO
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.semantic_actions_utils import SemanticActionsUtils
from decision_making.src.planning.types import FP_SX, FS_SX, FrenetState2D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.map_utils import MapUtils
from decision_making.src.state.state import EgoState
from decision_making.src.state.state import State, DynamicObject
from mapping.src.service.map_service import MapService


class SemanticActionType(Enum):
    FOLLOW_VEHICLE = 1
    FOLLOW_LANE = 2


class RelativeLane(Enum):
    """"
    The lane associated with a certain Recipe, relative to ego
    """
    RIGHT_LANE = -1
    SAME_LANE = 0
    LEFT_LANE = 1


class RelativeLongitudinalPosition(Enum):
    """"
    The high-level longitudinal position associated with a certain Recipe, relative to ego
    """
    REAR = -1
    PARALLEL = 0
    FRONT = 1


class DynamicObjectWithRoadSemantics:
    def __init__(self, dynamic_object: DynamicObject, distance: float, center_lane_latitude: float, fstate: FrenetState2D):
        self.dynamic_object = dynamic_object
        self.distance = distance
        self.center_lane_latitude = center_lane_latitude
        self.fstate = fstate

# Define semantic cell
SemanticGridCell = Tuple[RelativeLane, RelativeLongitudinalPosition]

# Define semantic occupancy grid
RoadSemanticOccupancyGrid = Dict[SemanticGridCell, List[DynamicObjectWithRoadSemantics]]


class BehavioralGridState(BehavioralState):
    def __init__(self, road_occupancy_grid: RoadSemanticOccupancyGrid, ego_state: EgoState,
                 right_lane_exists: bool, left_lane_exists: bool):
        self.road_occupancy_grid = road_occupancy_grid
        self.ego_state = ego_state
        self.right_lane_exists = right_lane_exists
        self.left_lane_exists = left_lane_exists

    @classmethod
    def create_from_state(cls, state: State, logger: Logger):
        """
        Occupy the occupancy grid.
        This method iterates over all dynamic objects, and fits them into the relevant cell
        in the semantic occupancy grid (semantic_lane, semantic_lon).
        Each cell holds a list of objects that are within the cell borders.
        In this particular implementation, we keep up to one dynamic object per cell, which is the closest to ego.
         (e.g. in the cells in front of ego, we keep objects with minimal longitudinal distance
         relative to ego front, while in all other cells we keep the object with the maximal longitudinal distance from
         ego front).
        :return: road semantic occupancy grid
        """
        road_id = state.ego_state.road_localization.road_id

        # TODO: the relative localization calculated here assumes that all objects are located on the same road.
        # TODO: Fix after demo and calculate longitudinal difference properly in the general case
        navigation_plan = MapService.get_instance().get_road_based_navigation_plan(current_road_id=road_id)

        road_points = MapService.get_instance()._shift_road_points_to_latitude(road_id, 0.0)
        road_frenet = FrenetSerret2DFrame(road_points)
        lanes_num = MapService.get_instance().get_road(road_id).lanes_num

        # Dict[SemanticGridCell, List[ObjectRelativeToEgo]]
        multi_object_grid = BehavioralGridState.project_objects_on_grid(state.dynamic_objects, state.ego_state,
                                                                        road_frenet)

        # for each grid cell - sort the dynamic objects by proximity to ego
        # Dict[SemanticGridCell, List[ObjectRelativeToEgo]]
        grid_sorted_by_distances = {cell: sorted(obj_dist_list, key=lambda rel_obj: abs(rel_obj.distance))
                                    for cell, obj_dist_list in multi_object_grid.items()}

        ego_lane = state.ego_state.road_localization.lane_num

        return cls(grid_sorted_by_distances, state.ego_state,
                   right_lane_exists=ego_lane > 0, left_lane_exists=ego_lane < lanes_num-1)


    @staticmethod
    def project_objects_on_grid(objects: List[DynamicObject], ego_state: EgoState, road_frenet: FrenetSerret2DFrame) -> \
            Dict[SemanticGridCell, List[DynamicObjectWithRoadSemantics]]:
        """
        Takes a list of objects and projects them unto a semantic grid relative to ego vehicle.
        Determine cell index in occupancy grid (lane and longitudinal location), under the assumption:
          Longitudinal location is defined by the grid structure:
              - front cells: starting from ego front + LON_MARGIN_FROM_EGO [m] and forward
              - back cells: starting from ego back - LON_MARGIN_FROM_EGO[m] and backwards
              - parallel cells: between ego back - LON_MARGIN_FROM_EGO to ego front + LON_MARGIN_FROM_EGO
        :param objects: list of dynamic objects to project
        :param ego_state: the state of ego vehicle
        :param navigation_plan: the frenet object corresponding to the relevant part of the road
        :return:
        """
        grid = defaultdict(list)

        maximal_considered_distance = BEHAVIORAL_PLANNING_LOOKAHEAD_DIST

        ego_lane = ego_state.road_localization.lane_num

        # We consider only object on the adjacent lanes
        adjecent_lanes = [x.value for x in RelativeLane]
        objects_in_adjecent_lanes = [obj for obj in objects if obj.road_localization.lane_num-ego_lane in adjecent_lanes]

        ego_init_fstate = MapUtils.get_ego_road_localization(ego_state, road_frenet)

        for obj in objects_in_adjecent_lanes:
            # Compute relative lane to ego
            object_relative_lane = RelativeLane(obj.road_localization.lane_num - ego_lane)

            # Compute relative longitudinal position to ego (on road)
            obj_init_fstate = MapUtils.get_object_road_localization(obj, road_frenet)

            # compute the relative longitudinal distance between object and ego (positive means object is in front)
            longitudinal_difference = obj_init_fstate[FS_SX] - ego_init_fstate[FS_SX]

            non_overlap_dist_to_ego = MapUtils.nonoverlapping_longitudinal_distance(
                ego_init_fstate, obj_init_fstate, ego_state.size.length, obj.size.length)

            if abs(non_overlap_dist_to_ego) > maximal_considered_distance:
                continue

            # Object is (at least LON_MARGIN_FROM_EGO) ahead of vehicle
            if non_overlap_dist_to_ego > LON_MARGIN_FROM_EGO:
                object_relative_long = RelativeLongitudinalPosition.FRONT
            # Object is (at least LON_MARGIN_FROM_EGO) behind of vehicle
            elif non_overlap_dist_to_ego < -LON_MARGIN_FROM_EGO:
                object_relative_long = RelativeLongitudinalPosition.REAR
            # Object vehicle aside of ego
            else:
                object_relative_long = RelativeLongitudinalPosition.PARALLEL

            obj_on_road = obj.road_localization
            road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(road_id=obj_on_road.road_id)
            obj_center_lane_latitude = road_lane_latitudes[obj_on_road.lane_num]

            grid[(object_relative_lane, object_relative_long)].append(
                DynamicObjectWithRoadSemantics(obj, longitudinal_difference, obj_center_lane_latitude, obj_init_fstate))

        return grid


