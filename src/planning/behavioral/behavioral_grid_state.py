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
from decision_making.src.planning.types import FP_SX, FS_SX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
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


class ObjectRelativeToEgo:
    def __init__(self, dynamic_object: DynamicObject, distance: float):
        self.dynamic_object = dynamic_object
        self.distance = distance

# Define semantic cell
SemanticGridCell = Tuple[RelativeLane, RelativeLongitudinalPosition]

# Define semantic occupancy grid
RoadSemanticOccupancyGrid = Dict[SemanticGridCell, List[DynamicObject]]


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

        # Dict[SemanticGridCell, List[(DynamicObject, NonOverlappingDistance:Float)]]
        multi_object_grid = BehavioralGridState.project_objects_on_grid(state.dynamic_objects, state.ego_state,
                                                                        road_frenet)

        # for each grid cell - sort the dynamic objects by proximity to ego
        # Dict[SemanticGridCell, List[DynamicObject]]
        grid_sorted_by_distances = {cell: sorted(obj_dist_list, key=lambda rel_obj: abs(rel_obj.distance))
                                    for cell, obj_dist_list in multi_object_grid.items()}

        # # for each grid cell - returns the closest object to ego vehicle
        # closest_object_grid = {cell: sorted_obj_dist_list[0].dynamic_object
        #                        for cell, sorted_obj_dist_list in grid_sorted_by_distances.items()}

        ego_lane = state.ego_state.road_localization.lane_num

        return cls(grid_sorted_by_distances, state.ego_state,
                   right_lane_exists=ego_lane > 0, left_lane_exists=ego_lane < lanes_num-1)


    @staticmethod
    def project_objects_on_grid(objects: List[DynamicObject], ego_state: EgoState, road_frenet: FrenetSerret2DFrame) -> \
            Dict[SemanticGridCell, List[ObjectRelativeToEgo]]:
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

        # We treat the object only if its distance is smaller than the distance we
        # would have travelled for the planning horizon in the average speed between current and target vel.
        distance_by_mean_velocity = SemanticActionsUtils.compute_distance_by_mean_velocity(
            current_velocity=ego_state.v_x,
            desired_velocity=BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED  # TODO: change for road-dependant velocity
        )
        maximal_considered_distance = min(BEHAVIORAL_PLANNING_LOOKAHEAD_DIST, distance_by_mean_velocity)

        ego_lane = ego_state.road_localization.lane_num

        # We consider only object on the adjacent lanes
        adjecent_lanes = [x.value for x in RelativeLane]
        objects_in_adjecent_lanes = [obj for obj in objects if obj.road_localization.lane_num-ego_lane in adjecent_lanes]

        for obj in objects_in_adjecent_lanes:
            # Compute relative lane to ego
            object_relative_lane = RelativeLane(obj.road_localization.lane_num - ego_lane)

            # Compute relative longitudinal position to ego (on road)
            non_overlap_dist_to_ego = BehavioralGridState.nonoverlapping_longitudinal_distance(ego_state, road_frenet,
                                                                                               obj)
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

            grid[(object_relative_lane, object_relative_long)].append(ObjectRelativeToEgo(obj, non_overlap_dist_to_ego))

        return grid

    @staticmethod
    def nonoverlapping_longitudinal_distance(ego_state: EgoState,
                                             road_frenet: FrenetSerret2DFrame,
                                             dynamic_object: DynamicObject) -> float:
        """
        Given dynamic object in a cell, calculate the distance from the object's boundaries to ego vehicle boundaries
        :param ego_state: state of ego vehicle
        :param road_frenet: frenet frame on which all objects exist
        :param dynamic_object: dynamic object in some cell
        :return: if object is in front of ego, then the returned value is positive and reflect the longitudinal distance
        between object's rear to ego-front. If the object behind ego, then the returned value is negative and reflect
        the longitudinal distance between object's front to ego-rear. If there's an overlap between ego and object on
        the longitudinal axis, the returned value is 0
        """
        # Object Frenet state
        target_obj_fpoint = road_frenet.cpoint_to_fpoint(np.array([dynamic_object.x, dynamic_object.y]))
        _, _, _, road_curvature_at_obj_location, _ = road_frenet._taylor_interp(target_obj_fpoint[FP_SX])
        obj_init_fstate = road_frenet.cstate_to_fstate(np.array([
            dynamic_object.x, dynamic_object.y,
            dynamic_object.yaw,
            dynamic_object.v_x,
            dynamic_object.acceleration_lon,
            road_curvature_at_obj_location  # We don't care about other agent's curvature, only the road's
        ]))

        # Ego Frenet state
        ego_init_cstate = np.array(
            [ego_state.x, ego_state.y, ego_state.yaw, ego_state.v_x, ego_state.acceleration_lon, ego_state.curvature])
        ego_init_fstate = road_frenet.cstate_to_fstate(ego_init_cstate)

        # Relative longitudinal distance
        object_relative_lon = obj_init_fstate[FS_SX] - ego_init_fstate[FS_SX]

        if object_relative_lon > (dynamic_object.size.length / 2 + ego_state.size.length / 2):
            return object_relative_lon - (dynamic_object.size.length / 2 + ego_state.size.length / 2)
        elif object_relative_lon < -(dynamic_object.size.length / 2 + ego_state.size.length / 2):
            return object_relative_lon + (dynamic_object.size.length / 2 + ego_state.size.length / 2)
        else:
            return 0
