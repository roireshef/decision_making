from collections import defaultdict
from enum import Enum
import numpy as np
from logging import Logger
from typing import Dict, List, Tuple, Optional

import rte.python.profiler as prof
from decision_making.src.global_constants import LON_MARGIN_FROM_EGO, PLANNING_LOOKAHEAD_DIST, MAX_HORIZON_DISTANCE
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
from decision_making.src.planning.types import FS_SX, FrenetState2D
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
    def __init__(self, road_occupancy_grid: RoadSemanticOccupancyGrid, ego_state: EgoState,
                 unified_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame],
                 projected_ego_fstates: Dict[RelativeLane, FrenetState2D]):
        self.road_occupancy_grid = road_occupancy_grid
        self.ego_state = ego_state
        self.unified_frames = unified_frames
        self.projected_ego_fstates = projected_ego_fstates

    @classmethod
    @prof.ProfileFunction()
    def create_from_state(cls, state: State, nav_plan: NavigationPlanMsg, logger: Logger):
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
        # TODO: since this function is called also for all terminal states, consider to make a simplified version of this function
        unified_frames = BehavioralGridState.create_generalized_frenet_frames(state, nav_plan)

        projected_ego_fstates = {rel_lane: unified_frames[rel_lane].cstate_to_fstate(state.ego_state.cartesian_state)
                                 for rel_lane in unified_frames}

        # Dict[SemanticGridCell, List[DynamicObjectWithRoadSemantics]]
        dynamic_objects_with_road_semantics = \
            sorted(BehavioralGridState._add_road_semantics(state.dynamic_objects, state.ego_state, unified_frames, projected_ego_fstates),
                   key=lambda rel_obj: abs(rel_obj.longitudinal_distance))

        multi_object_grid = BehavioralGridState._project_objects_on_grid(dynamic_objects_with_road_semantics,
                                                                         state.ego_state)
        return cls(multi_object_grid, state.ego_state, unified_frames, projected_ego_fstates)

    @staticmethod
    @prof.ProfileFunction()
    def _add_road_semantics(dynamic_objects: List[DynamicObject], ego_state: EgoState,
                            unified_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame],
                            projected_ego_fstates: Dict[RelativeLane, FrenetState2D]) -> \
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

        # for objects on non-adjacent lane set relative_lanes[i] = None
        rel_lanes_per_obj = np.full(len(dynamic_objects), None)
        for rel_lane in relative_lane_ids:
            # find all dynamic objects that belong to the current unified frame
            relevant_objects = unified_frames[rel_lane].has_segment_ids(objects_segment_ids)
            rel_lanes_per_obj[relevant_objects] = rel_lane

        # calculate longitudinal distances between the objects and ego, using unified_frames (GFF's)
        longitudinal_differences = BehavioralGridState._calculate_longitudinal_differences(
            unified_frames, ego_state.map_state.lane_id, projected_ego_fstates,
            objects_segment_ids, objects_segment_fstates, rel_lanes_per_obj)

        return [DynamicObjectWithRoadSemantics(obj, longitudinal_differences[i], rel_lanes_per_obj[i])
                for i, obj in enumerate(dynamic_objects) if rel_lanes_per_obj[i] is not None]

    @staticmethod
    def create_generalized_frenet_frames(state: State, nav_plan: NavigationPlanMsg) -> \
            Dict[RelativeLane, GeneralizedFrenetSerretFrame]:
        """
        For all available relative lanes create a relevant generalized frenet frame
        :param state:
        :param nav_plan:
        :return: dictionary from RelativeLane to GeneralizedFrenetSerretFrame
        """
        # calculate unified generalized frenet frames
        ego_lane_id = state.ego_state.map_state.lane_id
        adjacent_lanes_dict = MapUtils.get_relative_lane_ids(ego_lane_id)  # Dict: RelativeLane -> lane_id
        unified_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame] = {}
        suggested_ref_route_start = state.ego_state.map_state.lane_fstate[FS_SX] - PLANNING_LOOKAHEAD_DIST

        # TODO: remove this hack when using a real map from SP
        # if there is no long enough road behind ego, set ref_route_start = 0
        ref_route_start = suggested_ref_route_start \
            if suggested_ref_route_start >= 0 or MapUtils.does_map_exist_backward(ego_lane_id, -suggested_ref_route_start) \
            else 0

        frame_length = state.ego_state.map_state.lane_fstate[FS_SX] - ref_route_start + MAX_HORIZON_DISTANCE
        for rel_lane in adjacent_lanes_dict:
            unified_frames[rel_lane] = MapUtils.get_lookahead_frenet_frame(
                lane_id=adjacent_lanes_dict[rel_lane], starting_lon=ref_route_start, lookahead_dist=frame_length,
                navigation_plan=nav_plan)
        return unified_frames

    @staticmethod
    def project_ego_on_adjacent_lanes(ego_state: EgoState) -> Dict[RelativeLane, FrenetState2D]:
        """
        project cartesian state on the existing adjacent lanes
        :return: dictionary mapping between existing relative lane (adjacent to lane_id) to Frenet state
                                                                        projected on the adjacent Frenet frame
        """
        projected_fstates: Dict[RelativeLane, FrenetState2D] = {}
        for rel_lane in RelativeLane:
            if rel_lane == RelativeLane.SAME_LANE:
                projected_fstates[rel_lane] = ego_state.map_state.lane_fstate
            else:
                adjacent_lane_ids = MapUtils.get_adjacent_lanes(ego_state.map_state.lane_id, rel_lane)
                if len(adjacent_lane_ids) > 0:
                    adjacent_frenet = MapUtils.get_lane_frenet_frame(adjacent_lane_ids[0])
                    projected_fstates[rel_lane] = adjacent_frenet.cstate_to_fstate(ego_state.cartesian_state)
        return projected_fstates

    def calculate_longitudinal_differences(self, target_segment_ids: np.array, target_segment_fstates: np.array,
                                           rel_lanes_per_target: np.array) -> np.array:
        """
        Given target segment ids and segment fstates, calculate longitudinal differences between the targets and ego
        projected on the target lanes, using the relevant unified frames (GFF).
        :param target_segment_ids: array of original lane ids of the targets
        :param target_segment_fstates: array of target fstates w.r.t. their original lane ids
        :param rel_lanes_per_target: array of relative lanes (LEFT, SAME, RIGHT) for every target
        :return: array of longitudinal differences between the targets and projected ego per target
        """
        return BehavioralGridState._calculate_longitudinal_differences(
            self.unified_frames, self.ego_state.map_state.lane_id, self.projected_ego_fstates,
            target_segment_ids, target_segment_fstates, rel_lanes_per_target)

    @staticmethod
    def _calculate_longitudinal_differences(unified_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame],
                                            ego_lane_id: int, ego_unified_fstates: Dict[RelativeLane, FrenetState2D],
                                            target_segment_ids: np.array, target_segment_fstates: np.array,
                                            rel_lanes_per_target: np.array) -> np.array:
        """
        Given unified frames, ego projected on the unified frames, target segment ids and segment fstates, calculate
        longitudinal differences between the targets and ego.
        projected on the target lanes, using the relevant unified frames (GFF).
        :param target_segment_ids: array of original lane ids of the targets
        :param target_segment_fstates: array of target fstates w.r.t. their original lane ids
        :return: array of longitudinal differences between the targets and projected ego
        """
        tar_unified_fstates = np.empty((len(target_segment_ids), 6), dtype=float)
        relative_lane_ids = MapUtils.get_relative_lane_ids(ego_lane_id)

        # longitudinal difference between object and ego at t=0 (positive if obj in front of ego)
        for rel_lane in relative_lane_ids:  # loop over at most 3 relative lanes (adjacent)
            # find all targets belonging to the current unified frame
            relevant_idxs = unified_frames[rel_lane].has_segment_ids(target_segment_ids)
            if relevant_idxs.any():
                # convert relevant dynamic objects to fstate w.r.t. the current unified frame
                tar_unified_fstates[relevant_idxs] = unified_frames[rel_lane].convert_from_segment_states(
                    target_segment_fstates[relevant_idxs], target_segment_ids[relevant_idxs])

        longitudinal_differences = np.array([tar_unified_fstates[i, FS_SX] - ego_unified_fstates[rel_lane][FS_SX]
                                             for i, rel_lane in enumerate(rel_lanes_per_target)])
        return longitudinal_differences

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
