from collections import defaultdict
from logging import Logger
from typing import Dict, List, Tuple, Optional

import numpy as np
import rte.python.profiler as prof
from decision_making.src.exceptions import MappingException, OutOfSegmentBack, OutOfSegmentFront, LaneNotFound, \
    RoadNotFound, raises
from decision_making.src.global_constants import LON_MARGIN_FROM_EGO, PLANNING_LOOKAHEAD_DIST, MAX_BACKWARD_HORIZON, \
    MAX_FORWARD_HORIZON
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
from decision_making.src.planning.types import FS_SX, FrenetState2D, FP_SX, C_X, C_Y
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, GFFType
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import DynamicObject, EgoState
from decision_making.src.state.state import State
from decision_making.src.utils.map_utils import MapUtils


class DynamicObjectWithRoadSemantics:
    """
    This data object holds together the dynamic_object coupled with the distance from ego, his lane center latitude and
    its frenet state.
    """

    def __init__(self, dynamic_object: DynamicObject, longitudinal_distance: float,
                 relative_lanes: Optional[List[RelativeLane]]):
        """
        :param dynamic_object:
        :param longitudinal_distance: Distance relative to ego on the road's longitude
        :param relative_lane: list of relative lanes w.r.t. ego; None if the object is far laterally (not on adjacent lane)
                              usually just one, but can be multiple in the case of a merge
        """
        self.dynamic_object = dynamic_object
        self.longitudinal_distance = longitudinal_distance
        self.relative_lanes = relative_lanes


# Define semantic cell
SemanticGridCell = Tuple[RelativeLane, RelativeLongitudinalPosition]

# Define semantic occupancy grid
RoadSemanticOccupancyGrid = Dict[SemanticGridCell, List[DynamicObjectWithRoadSemantics]]


class BehavioralGridState:
    def __init__(self, road_occupancy_grid: RoadSemanticOccupancyGrid, ego_state: EgoState,
                 extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame],
                 projected_ego_fstates: Dict[RelativeLane, FrenetState2D]):
        """
        constructor of BehavioralGridState
        :param road_occupancy_grid: dictionary from grid cell to list of dynamic objects with semantics
        :param ego_state:
        :param extended_lane_frames: dictionary from RelativeLane to the corresponding GeneralizedFrenetSerretFrame
        :param projected_ego_fstates: dictionary from RelativeLane to ego Frenet state, which is ego projected on the
                corresponding extended_lane_frame
        """
        self.road_occupancy_grid = road_occupancy_grid
        self.ego_state = ego_state
        self.extended_lane_frames = extended_lane_frames
        self.projected_ego_fstates = projected_ego_fstates

    @classmethod
    @prof.ProfileFunction()
    def create_from_state(cls, state: State, route_plan: RoutePlan, logger: Logger):
        """
        Occupy the occupancy grid.
        This method iterates over all dynamic objects, and fits them into the relevant cell
        in the semantic occupancy grid (semantic_lane, semantic_lon).
        Each cell holds a list of objects that are within the cell borders.
        In this particular implementation, we keep up to one dynamic object per cell, which is the closest to ego.
         (e.g. in the cells in front of ego, we keep objects with minimal longitudinal distance
         relative to ego front, while in all other cells we keep the object with the maximal longitudinal distance from
         ego front).
        :return: created BehavioralGridState
        """
        # TODO: since this function is called also for all terminal states, consider to make a simplified version of this function
        extended_lane_frames = BehavioralGridState._create_generalized_frenet_frames(state, route_plan, logger)

        projected_ego_fstates = {
            rel_lane: extended_lane_frames[rel_lane].cstate_to_fstate(state.ego_state.cartesian_state)
            for rel_lane in extended_lane_frames}

        # TODO: Make sure to account for all relevant actors on all upstream lanes. For example, there may be an actor outside of an
        #  upstream merge area that is moving quickly. If we predict that this actor will come close to the host, we have to consider
        #  it as well.

        # Dict[SemanticGridCell, List[DynamicObjectWithRoadSemantics]]
        dynamic_objects_with_road_semantics = \
            sorted(BehavioralGridState._add_road_semantics(state.dynamic_objects, extended_lane_frames,
                                                           projected_ego_fstates),
                   key=lambda rel_obj: abs(rel_obj.longitudinal_distance))

        multi_object_grid = BehavioralGridState._project_objects_on_grid(dynamic_objects_with_road_semantics,
                                                                         state.ego_state)
        return cls(multi_object_grid, state.ego_state, extended_lane_frames, projected_ego_fstates)

    @staticmethod
    @prof.ProfileFunction()
    def _add_road_semantics(dynamic_objects: List[DynamicObject],
                            extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame],
                            projected_ego_fstates: Dict[RelativeLane, FrenetState2D]) -> \
            List[DynamicObjectWithRoadSemantics]:
        """
        Wraps DynamicObjects with "on-road" information (relative progress on road wrt ego, road-localization and more).
        This is a temporary function that caches relevant metrics for re-use. Should be removed after an efficient
        representation of DynamicObject.
        :param dynamic_objects: list of relevant DynamicObjects to calculate "on-road" metrics for.
        :param extended_lane_frames: dictionary from RelativeLane to the corresponding GeneralizedFrenetSerretFrame
        :param projected_ego_fstates: dictionary from RelativeLane to ego Frenet state, which is ego projected on the
                corresponding extended_lane_frame
        :return: list of object of type DynamicObjectWithRoadSemantics
        """

        # filter out off map dynamic objects
        on_map_dynamic_objects = np.array([obj for obj in dynamic_objects if not obj.off_map])

        # calculate objects' segment map_states
        object_map_states = [obj.map_state for obj in on_map_dynamic_objects]
        objects_segment_ids = np.array([map_state.lane_id for map_state in object_map_states])

        # for objects on non-adjacent lane set relative_lanes[i] = None
        rel_lanes_per_obj = np.full(len(on_map_dynamic_objects), None)
        # calculate relative to ego lane (RIGHT, SAME, LEFT) for every object
        for rel_lane, extended_lane_frame in extended_lane_frames.items():
            # find all dynamic objects that belong to the current unified frame
            relevant_object_mask = extended_lane_frame.has_segment_ids(objects_segment_ids)

            # assign object to lane if it hasn't already been assigned
            unassigned_obj_mask = (rel_lanes_per_obj == None)
            unassigned_relevant_obj_mask = np.logical_and(unassigned_obj_mask, relevant_object_mask)
            for i in range(len(on_map_dynamic_objects)):
                if unassigned_relevant_obj_mask[i]:
                    rel_lanes_per_obj[i] = np.array([rel_lane])

            # add rel_lanes to obj's rel_lanes list if it belongs to more than one rel_lane
            # invert <unassigned_obj_mask> instead of recalculating since rel_lanes_per_obj has been changed
            previously_assigned_obj_indices = np.nonzero(np.logical_and(np.logical_not(unassigned_obj_mask), relevant_object_mask))[0].tolist()
            if len(previously_assigned_obj_indices) > 0:
                # add another rel lane to the rel_lane list
                for idx in previously_assigned_obj_indices:
                    rel_lanes_per_obj[idx] = np.append(rel_lanes_per_obj[idx], [rel_lane])

        # calculate longitudinal distances between the objects and ego, using extended_lane_frames (GFF's)
        longitudinal_differences = BehavioralGridState._calculate_longitudinal_differences(
            extended_lane_frames, projected_ego_fstates, object_map_states)

        return [DynamicObjectWithRoadSemantics(obj, longitudinal_differences[i], rel_lanes_per_obj[i])
                for i, obj in enumerate(on_map_dynamic_objects) if rel_lanes_per_obj[i] is not None]

    def calculate_longitudinal_differences(self, target_map_states: List[MapState]) -> np.array:
        """
        Given target segment ids and segment fstates, calculate longitudinal differences between the targets and ego
        projected on the target lanes, using the relevant unified frames (GFF).
        :param target_map_states: list of original map states of the targets
        :return: array of longitudinal differences between the targets and projected ego per target
        """
        return BehavioralGridState._calculate_longitudinal_differences(
            self.extended_lane_frames, self.projected_ego_fstates, target_map_states)

    @staticmethod
    def _calculate_longitudinal_differences(extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame],
                                            ego_unified_fstates: Dict[RelativeLane, FrenetState2D],
                                            target_map_states: List[MapState]) -> np.array:
        """
        Given unified frames, ego projected on the unified frames, target segment ids and segment fstates, calculate
        longitudinal differences between the targets and ego.
        projected on the target lanes, using the relevant unified frames (GFF).
        :param extended_lane_frames: mapping between 3 lanes relative to the host vehicle (left adjacent, same,
                                                                        right adjacent) to their curve representation
        :param ego_unified_fstates: dictionary from RelativeLane to ego Frenet state, which is ego projected on the
                corresponding extended_lane_frame
        :param target_map_states: list of original map states of the targets
        :return: array of longitudinal differences between the targets and projected ego
        """
        target_segment_ids = np.array([map_state.lane_id for map_state in target_map_states])
        target_segment_fstates = np.array([map_state.lane_fstate for map_state in target_map_states])

        # initialize longitudinal_differences to infinity
        longitudinal_differences = np.full(len(target_segment_ids), np.inf)

        # longitudinal difference between object and ego at t=0 (positive if obj in front of ego)
        for rel_lane, extended_lane_frame in extended_lane_frames.items():  # loop over at most 3 unified frames
            # find all targets belonging to the current unified frame
            relevant_idxs = extended_lane_frame.has_segment_ids(target_segment_ids)
            if relevant_idxs.any():
                # convert relevant dynamic objects to fstate w.r.t. the current unified frame
                target_unified_fstates = extended_lane_frame.convert_from_segment_states(
                    target_segment_fstates[relevant_idxs], target_segment_ids[relevant_idxs])
                # calculate longitudinal distances between the targets from this extended frame and ego projected on it
                longitudinal_differences[relevant_idxs] = \
                    target_unified_fstates[:, FS_SX] - ego_unified_fstates[rel_lane][FS_SX]

        return longitudinal_differences

    @staticmethod
    @prof.ProfileFunction()
    def _create_generalized_frenet_frames(state: State, route_plan: RoutePlan, logger: Logger) -> \
            Dict[RelativeLane, GeneralizedFrenetSerretFrame]:
        """
        For all available nearest lanes create a corresponding generalized frenet frame (long enough) that can
        contain multiple original lane segments. Augmented frenet frames may be created if there are lane splits ahead.
        :param state: the current world state
        :param route_plan: the route plan which contains lane costs
        :return: dictionary from RelativeLane to GeneralizedFrenetSerretFrame
        """
        # calculate unified generalized frenet frames
        ego_lane_id = state.ego_state.map_state.lane_id
        closest_lanes_dict = MapUtils.get_closest_lane_ids(ego_lane_id)  # Dict: RelativeLane -> lane_id

        # Augmented GFFS can be created only if the lanes don't currently exist
        can_augment = {RelativeLane.LEFT_LANE: RelativeLane.LEFT_LANE not in closest_lanes_dict.keys(),
                       RelativeLane.RIGHT_LANE: RelativeLane.RIGHT_LANE not in closest_lanes_dict.keys()}

        extended_lane_frames = {}

        # Create generalized Frenet frame for the host's lane
        try:
            lane_gff_dict = BehavioralGridState.get_generalized_frenet_frames_by_cost(
                lane_id=closest_lanes_dict[RelativeLane.SAME_LANE], station=state.ego_state.map_state.lane_fstate[FS_SX],
                route_plan=route_plan, logger=logger, can_augment=can_augment)
        except MappingException as e:
            # in case of failure to build GFF for SAME_LANE, stop processing this BP frame
            raise AssertionError("Trying to fetch data for %s, but data is unavailable. %s" % (RelativeLane.SAME_LANE, str(e)))

        # set the SAME_LANE first since it cannot be augmented
        extended_lane_frames[RelativeLane.SAME_LANE] = lane_gff_dict[RelativeLane.SAME_LANE]

        host_cartesian_point = np.array([state.ego_state.cartesian_state[C_X],
                                         state.ego_state.cartesian_state[C_Y]])

        # If an adjacent lane exists, create a generalized Frenet frame for it
        for relative_lane in [RelativeLane.LEFT_LANE, RelativeLane.RIGHT_LANE]:
            # can_augment is True only if the adjacent lane does not exist. Therefore, the only thing that can be done is to
            # create an augmented GFF
            if can_augment[relative_lane]:
                # Even though a lane augmentation is possible, it may not exist
                # (e.g. right lane doesn't exist allowing for an augmented GFF, but there are no right splits ahead)
                # Need to check if the augmented GFF was actually created
                if relative_lane in lane_gff_dict:
                    extended_lane_frames[relative_lane] = lane_gff_dict[relative_lane]
            else:
                try:
                    # Find station in the relative lane that is adjacent to the host's station in the lane it is occupying
                    adjacent_lane_frenet_frame = MapUtils.get_lane_frenet_frame(closest_lanes_dict[relative_lane])
                    host_station_in_adjacent_lane = adjacent_lane_frenet_frame.cpoint_to_fpoint(host_cartesian_point)[FP_SX]

                except OutOfSegmentBack:
                    # The host's position on the adjacent lane could not be found because the frame is ahead of the host. This may happen
                    # as the host is transitioning between road segments. The host should be close to the beginning of the road segment
                    # though so the initial station is used here.
                    host_station_in_adjacent_lane = 0.0

                    # TODO: Should we check the actual distance from the host to the first point on the frame and do something if the
                    #  distance is too large?

                except OutOfSegmentFront:
                    # The host's position on the adjacent lane could not be found because the frame is behind the host. This may happen
                    # as the host is transitioning between road segments. The host should be close to the end of the road segment
                    # though so the max station is used here.
                    host_station_in_adjacent_lane = adjacent_lane_frenet_frame.s_max

                    # TODO: Should we check the actual distance from the host to the last point on the frame and do something if the
                    #  distance is too large?

                # If the left or right exists, do a lookahead from that lane instead of using the augmented lanes
                try:
                    lane_gffs = BehavioralGridState.get_generalized_frenet_frames_by_cost(lane_id=closest_lanes_dict[relative_lane],
                                                                              station=host_station_in_adjacent_lane,
                                                                              route_plan=route_plan,
                                                                              logger=logger)

                    # Note that the RelativeLane keys that are in the returned dictionary from get_lookahead_frenet_frame_by_cost are
                    # with respect to the lane ID provided to the function. Therefore, since the lane ID for the left/right lane is
                    # provided to the function above, the RelativeLane.SAME_LANE key in the returned dictionary actually refers to the
                    # left/right lane. That makes the use of this key below correct.
                    extended_lane_frames[relative_lane] = lane_gffs[RelativeLane.SAME_LANE]
                except MappingException as e:
                    logger.warning("Trying to fetch data for %s, but data is unavailable. %s" % (relative_lane, str(e)))

        return extended_lane_frames

    @staticmethod
    @raises(LaneNotFound, RoadNotFound)
    @prof.ProfileFunction()
    def get_generalized_frenet_frames_by_cost(lane_id: int, station: float, route_plan: RoutePlan,
                                              logger: Optional[Logger] = None,
                                              can_augment: Optional[Dict[RelativeLane, bool]] = None) -> \
            Dict[RelativeLane, GeneralizedFrenetSerretFrame]:
        """
        Create Generalized Frenet frame(s) along lane center, starting from given lane and station. If augmented lanes can be created, they will
        be returned in the dictionary under the RelativeLane.LEFT_LANE/RIGHT_LANE keys.
        :param lane_id: starting lane_id
        :param station: starting station [m]
        :param route_plan: the relevant navigation plan to iterate over its road IDs.
        :param logger: Logger object to log warning messages
        :param can_augment: Dict of RelativeLane to bool describing if a search for an augmented LEFT/RIGHT lane
                            starting from the lane_id is needed.
        :return: Dict of generalized Frenet frames with the relative lane as keys
                 The relative lane key is with respect to the provided lane_id. The dictionary will always contain the GFF for the provided
                 lane_id, and the RelativeLane.SAME_LANE key can be used to access it. If possible, augmented GFFs will also be returned,
                 and they can be accessed with the respective relative lane key.
        """
        if station < MAX_BACKWARD_HORIZON:
            # If the given station is not far enough along the lane, then the backward horizon will pass the beginning of the lane, and the
            # upstream lane subsegments need to be found. The starting station for the forward lookahead should be the beginning of the
            # current lane, and the forward lookahead distance should include the maximum forward horizon ahead of the given station and
            # the backward distance to the beginning of the lane (i.e. the station).
            starting_station = 0.0
            lookahead_distance = MAX_FORWARD_HORIZON + station
            upstream_lane_subsegments = MapUtils._get_upstream_lane_subsegments(lane_id, station, MAX_BACKWARD_HORIZON,
                                                                                logger)
        else:
            # If the given station is far enough along the lane, then the backward horizon will not pass the beginning of the lane. In this
            # case, the starting station for the forward lookahead should be the end of the backward horizon, and the forward lookahead
            # distance should include the maximum forward and backward horizons ahead of and behind the given station, respectively. In
            # other words, if we're at station = 150 m on a lane and the maximum forward and backward horizons are 400 m and 100 m,
            # respectively, then starting station = 50 m and forward lookahead distance = 400 + 100 = 500 m. This is the case where the GFF
            # does not include any upstream lanes.
            starting_station = station - MAX_BACKWARD_HORIZON
            lookahead_distance = MAX_FORWARD_HORIZON + MAX_BACKWARD_HORIZON
            upstream_lane_subsegments = []

        lane_subsegments_dict = MapUtils._advance_by_cost(initial_lane_id=lane_id,
                                                          initial_s=starting_station,
                                                          lookahead_distance=lookahead_distance,
                                                          route_plan=route_plan,
                                                          lane_subsegments=upstream_lane_subsegments,
                                                          can_augment=can_augment)

        gffs_dict = {}

        # Build GFFs from the Frenet Subsegments for the lane/augmented lanes that were created.
        for rel_lane in lane_subsegments_dict:
            lane_subsegments, is_partial, is_augmented, _ = lane_subsegments_dict[rel_lane]
            gff_type = GFFType.get(is_partial, is_augmented)

            # Create Frenet frame for each sub segment
            frenet_frames = [MapUtils.get_lane_frenet_frame(lane_subsegment.e_i_SegmentID)
                             for lane_subsegment in upstream_lane_subsegments + lane_subsegments]

            # Create GFF
            gffs_dict[rel_lane] = GeneralizedFrenetSerretFrame.build(frenet_frames, lane_subsegments, gff_type)

        return gffs_dict

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
            if abs(obj.longitudinal_distance) <= PLANNING_LOOKAHEAD_DIST and obj.relative_lanes is not None:
                # compute longitudinal projection on the grid
                object_relative_long = BehavioralGridState._get_longitudinal_grid_cell(obj, ego_state)
                # loop through all the rel_lanes in the case of an obj belonging to more than one (splits/merges)
                for rel_lane in obj.relative_lanes:
                    grid[(rel_lane, object_relative_long)].append(obj)

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
