from collections import defaultdict
from enum import Enum
from logging import Logger
from typing import Dict, List, Tuple
import numpy as np
import time

import rte.python.profiler as prof
from decision_making.src.global_constants import LON_MARGIN_FROM_EGO
from decision_making.src.global_constants import PLANNING_LOOKAHEAD_DIST
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.types import FS_SX, FrenetState2D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.state import NewDynamicObject, NewEgoState
from decision_making.src.state.state import State
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
    """
    This data object holds together the dynamic_object coupled with the distance from ego, his lane center latitude and
    its frenet state.
    """
    def __init__(self, dynamic_object: NewDynamicObject, longitudinal_distance: float):
        """
        :param dynamic_object:
        :param longitudinal_distance: Distance relative to ego on the road's longitude
        """
        self.dynamic_object = dynamic_object
        self.longitudinal_distance = longitudinal_distance


# Define semantic cell
SemanticGridCell = Tuple[RelativeLane, RelativeLongitudinalPosition]

# Define semantic occupancy grid
RoadSemanticOccupancyGrid = Dict[SemanticGridCell, List[DynamicObjectWithRoadSemantics]]


class BehavioralGridState(BehavioralState):
    def __init__(self, road_occupancy_grid: RoadSemanticOccupancyGrid, ego_state: NewEgoState,
                 right_lane_exists: bool, left_lane_exists: bool):
        self.road_occupancy_grid = road_occupancy_grid
        self.ego_state = ego_state
        self.right_lane_exists = right_lane_exists
        self.left_lane_exists = left_lane_exists

    @classmethod
    @prof.ProfileFunction()
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
        road_id = state.ego_state.map_state.road_id

        # TODO: the relative localization calculated here assumes that all objects are located on the same road.
        # TODO: Fix after demo and calculate longitudinal difference properly in the general case
        navigation_plan = MapService.get_instance().get_road_based_navigation_plan(current_road_id=road_id)

        # Dict[SemanticGridCell, List[DynamicObjectWithRoadSemantics]]
        dynamic_objects_with_road_semantics = BehavioralGridState._add_road_semantics(state.dynamic_objects,
                                                                                      state.ego_state)
        multi_object_grid = BehavioralGridState._project_objects_on_grid(dynamic_objects_with_road_semantics,
                                                                         state.ego_state)

        # for each grid cell - sort the dynamic objects by proximity to ego
        # Dict[SemanticGridCell, List[DynamicObjectWithRoadSemantics]]
        grid_sorted_by_distances = {cell: sorted(obj_dist_list, key=lambda rel_obj: abs(rel_obj.longitudinal_distance))
                                    for cell, obj_dist_list in multi_object_grid.items()}

        ego_lane = state.ego_state.map_state.lane_num
        lanes_num = MapService.get_instance().get_road(road_id).lanes_num

        return cls(grid_sorted_by_distances, state.ego_state,
                   right_lane_exists=ego_lane > 0, left_lane_exists=ego_lane < lanes_num-1)

    @staticmethod
    @prof.ProfileFunction()
    def _add_road_semantics(dynamic_objects: List[NewDynamicObject], ego_state: NewEgoState) -> \
            List[DynamicObjectWithRoadSemantics]:
        """
        Wraps DynamicObjects with "on-road" information (relative progress on road wrt ego, road-localization and more).
        This is a temporary function that caches relevant metrics for re-use. Should be removed after an efficient
        representation of DynamicObject.
        :param dynamic_objects: list of relevant DynamicObjects to calculate "on-road" metrics for.
        :param ego_state:
        :return: list of object of type DynamicObjectWithRoadSemantics
        """
        ego_init_fstate = ego_state.map_state.road_fstate
        # compute the relative longitudinal distance between object and ego (positive means object is in front)
        return [DynamicObjectWithRoadSemantics(obj, obj.map_state.road_fstate[FS_SX] - ego_init_fstate[FS_SX]) for obj in dynamic_objects]

    @staticmethod
    @prof.ProfileFunction()
    def _project_objects_on_grid(objects: List[DynamicObjectWithRoadSemantics], ego_state: NewEgoState) -> \
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
        :return:
        """
        grid = defaultdict(list)

        ego_lane = ego_state.map_state.lane_num

        # We consider only object on the adjacent lanes
        adjacent_lanes = [x.value for x in RelativeLane]

        # ignore vehicles out of pre-defined range
        near_objects = [obj for obj in objects if abs(obj.longitudinal_distance) <= PLANNING_LOOKAHEAD_DIST]
        # ignore vehicles not in adjacent lanes
        nearby_objects_in_adjacent_lanes = [obj for obj in near_objects
                                            if obj.dynamic_object.map_state.lane_num - ego_lane in adjacent_lanes]

        for obj in nearby_objects_in_adjacent_lanes:
            # Compute relative lane to ego
            object_relative_lane = RelativeLane(obj.dynamic_object.map_state.lane_num - ego_lane)

            # compute longitudinal projection on the grid
            object_relative_long = BehavioralGridState._get_longitudinal_grid_cell(obj, ego_state)

            grid[(object_relative_lane, object_relative_long)].append(obj)

        return grid

    @staticmethod
    @prof.ProfileFunction()
    def _get_longitudinal_grid_cell(object: DynamicObjectWithRoadSemantics, ego_state: NewEgoState):
        """
        Given a dynamic object representation and ego state, calculate what is the proper longitudinal
        relative-grid-cell to project it on. An object is set to be in FRONT cell if the distance from its rear to ego's
        front is greater than LON_MARGIN_FROM_EGO.
        An object is set to be in REAR cell if the distance from its front to ego's rear is greater
        than LON_MARGIN_FROM_EGO. Otherwise, the object is set to be parallel to ego.
        (positive if object is in front). difference is computed between objects'-centers
        :param object: the object to project onto the ego-relative grid
        :param ego_state: ego state for localization and size
        :return: RelativeLongitudinalPosition enum's value representing the longitudinal projection on the relative-grid
        """
        obj_length = object.dynamic_object.size.length
        ego_length = ego_state.size.length
        if object.longitudinal_distance > (obj_length / 2 + ego_length / 2 + LON_MARGIN_FROM_EGO):
            return RelativeLongitudinalPosition.FRONT
        elif object.longitudinal_distance < -(obj_length / 2 + ego_length / 2 + LON_MARGIN_FROM_EGO):
            return RelativeLongitudinalPosition.REAR
        else:
            return RelativeLongitudinalPosition.PARALLEL
