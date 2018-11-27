from collections import defaultdict
from enum import Enum
import numpy as np
from logging import Logger
from typing import Dict, List, Tuple, Optional

import rte.python.profiler as prof
from decision_making.src.global_constants import LON_MARGIN_FROM_EGO, PLANNING_LOOKAHEAD_DIST
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
from decision_making.src.planning.types import FS_SX
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, \
    FrenetSubSegment
from decision_making.src.state.state import DynamicObject, EgoState
from decision_making.src.state.state import State
from decision_making.src.utils.map_utils import MapUtils


class DynamicObjectWithRoadSemantics:
    """
    This data object holds together the dynamic_object coupled with the distance from ego, his lane center latitude and
    its frenet state.
    """

    def __init__(self, dynamic_object: DynamicObject, longitudinal_distance: float, relative_lane: Optional[RelativeLane]):
        """
        :param dynamic_object:
        :param longitudinal_distance: Distance relative to ego on the road's longitude
        :param relative_lane: relative lane w.r.t. ego; None if the object is far laterally (not on adjacent lane)
        """
        self.dynamic_object = dynamic_object
        self.longitudinal_distance = longitudinal_distance
        self.relative_lane = relative_lane


# Define semantic cell
SemanticGridCell = Tuple[RelativeLane, RelativeLongitudinalPosition]

# Define semantic occupancy grid
RoadSemanticOccupancyGrid = Dict[SemanticGridCell, List[DynamicObjectWithRoadSemantics]]


class BehavioralGridState(BehavioralState):
    def __init__(self, road_occupancy_grid: RoadSemanticOccupancyGrid, ego_state: EgoState):
        self.road_occupancy_grid = road_occupancy_grid
        self.ego_state = ego_state

    @classmethod
    @prof.ProfileFunction()
    def create_from_state(cls, state: State, unified_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame], logger: Logger):
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
        # TODO: the relative localization calculated here assumes that all objects are located on the same road.
        # TODO: Fix after demo and calculate longitudinal difference properly in the general case
        # navigation_plan = MapService.get_instance().get_road_based_navigation_plan(current_road_id=road_id)

        # Dict[SemanticGridCell, List[DynamicObjectWithRoadSemantics]]
        dynamic_objects_with_road_semantics = \
            sorted(BehavioralGridState._add_road_semantics(state.dynamic_objects, state.ego_state, unified_frames),
                   key=lambda rel_obj: abs(rel_obj.longitudinal_distance))

        multi_object_grid = BehavioralGridState._project_objects_on_grid(dynamic_objects_with_road_semantics,
                                                                         state.ego_state)
        return cls(multi_object_grid, state.ego_state)

    @staticmethod
    @prof.ProfileFunction()
    def _add_road_semantics(dynamic_objects: List[DynamicObject], ego_state: EgoState,
                            unified_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame]) -> \
            List[DynamicObjectWithRoadSemantics]:
        """
        Wraps DynamicObjects with "on-road" information (relative progress on road wrt ego, road-localization and more).
        This is a temporary function that caches relevant metrics for re-use. Should be removed after an efficient
        representation of DynamicObject.
        :param dynamic_objects: list of relevant DynamicObjects to calculate "on-road" metrics for.
        :param ego_state:
        :return: list of object of type DynamicObjectWithRoadSemantics
        """
        relative_lane_ids = MapUtils.get_relative_lane_ids(ego_state.map_state.lane_id)
        # calculate objects' segment map_states
        objects_segment_ids = np.array([obj.map_state.lane_id for obj in dynamic_objects])
        objects_segment_fstates = np.array([obj.map_state.lane_fstate for obj in dynamic_objects])

        ego_unified_fstates = np.empty((len(objects_segment_ids), 6), dtype=float)
        obj_unified_fstates = np.empty((len(objects_segment_ids), 6), dtype=float)

        # for objects on non-adjacent lane set relative_lanes[i] = None
        rel_lanes_per_obj: List[RelativeLane] = [None]*len(dynamic_objects)

        # calculate ego_unified_fstates and obj_unified_fstates w.r.t. their unified frame
        for rel_lane in relative_lane_ids:
            # find all dynamic objects that belong to the current unified frame
            objects_belong_to_gff = unified_frames[rel_lane].has_segment_ids(objects_segment_ids)

            # project ego on the current unified frame
            ego_unified_fstates[objects_belong_to_gff] = unified_frames[rel_lane].convert_from_segment_state(
                    ego_state.get_adjacent_map_state(rel_lane).lane_fstate, relative_lane_ids[rel_lane])

            # convert relevant dynamic objects to fstate w.r.t. the current unified frame
            obj_unified_fstates[objects_belong_to_gff] = unified_frames[rel_lane].convert_from_segment_states(
                    objects_segment_fstates[objects_belong_to_gff], objects_segment_ids[objects_belong_to_gff])

            rel_lanes_per_obj[objects_belong_to_gff] = rel_lane

        return [DynamicObjectWithRoadSemantics(obj, obj_unified_fstates[i, FS_SX] - ego_unified_fstates[i, FS_SX],
                                               rel_lanes_per_obj[i])
                for i, obj in enumerate(dynamic_objects) if rel_lanes_per_obj[i] is not None]

    @staticmethod
    @prof.ProfileFunction()
    def _project_objects_on_grid(objects: List[DynamicObjectWithRoadSemantics], ego_state: EgoState) -> \
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

        # We consider only object on the adjacent lanes
        for obj in objects:
            # ignore vehicles out of pre-defined range and vehicles not in adjacent lanes
            if abs(obj.longitudinal_distance) <= PLANNING_LOOKAHEAD_DIST and obj.relative_lane is not None:
                # compute longitudinal projection on the grid
                object_relative_long = BehavioralGridState._get_longitudinal_grid_cell(obj, ego_state)

                grid[(obj.relative_lane, object_relative_long)].append(obj)

        return grid

    @staticmethod
    @prof.ProfileFunction()
    def _get_longitudinal_grid_cell(object: DynamicObjectWithRoadSemantics, ego_state: EgoState):
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
