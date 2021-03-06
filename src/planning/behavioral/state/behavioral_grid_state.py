import itertools
from collections import defaultdict
from logging import Logger
from typing import Dict, List, Tuple, Optional

import numpy as np
import rte.python.profiler as prof
from decision_making.src.exceptions import MappingException, OutOfSegmentBack, OutOfSegmentFront, LaneNotFound, \
    RoadNotFound, raises, StraightConnectionNotFound, UpstreamLaneNotFound
from decision_making.src.global_constants import LON_MARGIN_FROM_EGO, PLANNING_LOOKAHEAD_DIST, MAX_BACKWARD_HORIZON, \
    MAX_FORWARD_HORIZON, LOG_MSG_BEHAVIORAL_GRID, DIM_MARGIN_TO_STOP_BAR
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.scene_static_enums import LaneOverlapType, ManeuverType
from decision_making.src.messages.scene_static_message import TrafficControlBar
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeState
from decision_making.src.planning.types import FS_SX, FS_DX, FrenetState2D, FP_SX, C_X, C_Y, FP_DX
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, GFFType, \
    FrenetSubSegment
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import DynamicObject, EgoState, State
from decision_making.src.utils.map_utils import MapUtils


class DynamicObjectWithRoadSemantics:
    """
    This data object holds together the dynamic_object coupled with the distance from ego, his lane center latitude and
    its frenet state.
    """

    def __init__(self, dynamic_object: DynamicObject, longitudinal_distance: float,
                 relative_lanes: Optional[List[RelativeLane]] = None):
        """
        :param dynamic_object:
        :param longitudinal_distance: Distance relative to ego on the road's longitude
        :param relative_lanes: list of relative lanes w.r.t. ego; None if the object is far laterally (not on adjacent lane)
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
                 projected_ego_fstates: Dict[RelativeLane, FrenetState2D],
                 tcb_in_gff_and_their_distances: Dict[RelativeLane, Tuple[TrafficControlBar, float]],
                 lane_change_state: LaneChangeState,
                 ignored_tcb_distance_in_gff: Dict[RelativeLane, float],
                 logger: Logger):
        """
        constructor of BehavioralGridState
        :param road_occupancy_grid: dictionary from grid cell to list of dynamic objects with semantics
        :param ego_state:
        :param extended_lane_frames: dictionary from RelativeLane to the corresponding GeneralizedFrenetSerretFrame
        :param projected_ego_fstates: dictionary from RelativeLane to ego Frenet state, which is ego projected on the
                corresponding extended_lane_frame
        :param tcb_in_gff_and_their_distances: closest TCB per GFF lane and its distance from ego
        :param lane_change_state: lane change state
        :param ignored_tcb_distance_in_gff: distance to ignored
        :param logger
        """
        self.road_occupancy_grid = road_occupancy_grid
        self.ego_state = ego_state
        self.extended_lane_frames = extended_lane_frames
        self.projected_ego_fstates = projected_ego_fstates
        self.tcb_in_gff_and_their_distances = tcb_in_gff_and_their_distances
        self.lane_change_state = lane_change_state
        self.ignored_tcb_distance_in_gff = ignored_tcb_distance_in_gff
        self.logger = logger
        self.max_speed_limit = self._get_max_speed_limit()

    @property
    def ego_length(self) -> float:
        return self.ego_state.size.length

    @classmethod
    @prof.ProfileFunction()
    def create_from_state(cls, state: State, route_plan: RoutePlan, lane_change_state: LaneChangeState, logger: Logger):
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
        extended_lane_frames = BehavioralGridState._create_generalized_frenet_frames(state.ego_state, route_plan, logger)

        projected_ego_fstates = {rel_lane: extended_lane_frames[rel_lane].cstate_to_fstate(state.ego_state.cartesian_state)
                                 for rel_lane in extended_lane_frames}

        # TODO: Make sure to account for all relevant actors on all upstream lanes. For example, there may be an actor outside of an
        #  upstream merge area that is moving quickly. If we predict that this actor will come close to the host, we have to consider
        #  it as well.

        # Dict[SemanticGridCell, List[DynamicObjectWithRoadSemantics]]
        dynamic_objects_with_road_semantics = \
            sorted(BehavioralGridState._add_road_semantics(state.dynamic_objects, extended_lane_frames, projected_ego_fstates, logger),
                   key=lambda rel_obj: abs(rel_obj.longitudinal_distance))

        multi_object_grid = BehavioralGridState._project_objects_on_grid(dynamic_objects_with_road_semantics,
                                                                         state.ego_state)

        BehavioralGridState._log_grid_data(multi_object_grid, state.ego_state.timestamp_in_sec, logger)

        tcb_and_ignored_tcbs_in_gff_and_their_distances = \
            BehavioralGridState._get_closest_stop_bars(extended_lane_frames, projected_ego_fstates,
                                                       state.ego_state.get_stop_bar_to_ignore(), logger)
        tcb_in_gff_and_their_distances = {k: v[0] for k, v in tcb_and_ignored_tcbs_in_gff_and_their_distances.items()}
        ignored_tcb_distance_in_gff = {k: v[1] for k, v in tcb_and_ignored_tcbs_in_gff_and_their_distances.items()}

        return cls(multi_object_grid, state.ego_state, extended_lane_frames, projected_ego_fstates,
                   tcb_in_gff_and_their_distances, lane_change_state, ignored_tcb_distance_in_gff, logger)

    @staticmethod
    def _create_projected_objects(dynamic_objects: List[DynamicObject],
                                  extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame], logger: Logger) -> List[DynamicObject]:
        """
        Creates projected, "ghost" objects related to actual dynamic objects

        This function finds the dynamic objects that are in an area where it is desirable to create "ghost" objects in other
        lanes and creates those objects. Projected objects all fields unchanged except:
            map_state.lane_fstate: projected fstate on the target (projection on overlap) lane
            map_state.lane_id: the lane ID of the target (projection on overlap) lane
            is_ghost: True

        :param dynamic_objects: list of dynamic objects
        :param logger:
        :return: list of projected dynamic objects
        """
        relevant_overlap_types = [LaneOverlapType.CeSYS_e_LaneOverlapType_Merge, LaneOverlapType.CeSYS_e_LaneOverlapType_Split]

        # create a map between RelativeLane and a list of all LaneOverlaps of the GFF for that RelativeLane
        rel_lane_to_overlaps = {rel: list(itertools.chain(*[MapUtils.get_lane(lane_id).as_lane_overlaps
                                                            for lane_id in gff.segment_ids]))
                                for rel, gff in extended_lane_frames.items()}

        # create a map between RelativeLane and a list of all relevant DynamicObjects to project onto that RelativeLane
        # by crossing all dynamic objects with every RelativeLane's lane-overlaps
        rel_lane_to_overlapping_objs = {rel: list(itertools.chain(*[[dyn_obj for dyn_obj in dynamic_objects
                                                                     if dyn_obj.map_state.lane_id == overlap.e_i_other_lane_segment_id
                                                                     and overlap.e_e_lane_overlap_type in relevant_overlap_types
                                                                     and (not dyn_obj.off_map)
                                                                     and NumpyUtils.is_in_limits(
                                                                         dyn_obj.map_state.lane_fstate[FS_SX],
                                                                         overlap.a_l_other_lane_overlap_stations +
                                                                         [-dyn_obj.size.length/2 , dyn_obj.size.length/2])
                                                                     and BehavioralGridState._is_object_in_lane(
                                                                         dyn_obj, extended_lane_frames[rel], logger)]
                                                                    for overlap in overlaps]))
                                        for rel, overlaps in rel_lane_to_overlaps.items()}

        # Project all overlapping dynamic objects on the GFFs and return a list of projected objects
        projected_dynamic_objects = []

        for rel, dyn_objs in rel_lane_to_overlapping_objs.items():
            if len(dyn_objs) == 0:
                continue

            # A try/catch for OutOfSegment is not needed here since _is_object_in_lane will return False if the centerpoint is OutOfSegment
            dyn_objs_cstates = np.array([dyn_obj.cartesian_state for dyn_obj in dyn_objs])
            projected_dyn_objs_fstates = extended_lane_frames[rel].ctrajectory_to_ftrajectory(dyn_objs_cstates)

            lane_ids, lane_fstates = extended_lane_frames[rel].convert_to_segment_states(projected_dyn_objs_fstates)

            for dynamic_object, lane_id, lane_fstate in zip(dyn_objs, lane_ids, lane_fstates):
                projected_dynamic_objects.append(DynamicObject(obj_id=dynamic_object.obj_id,
                                                               timestamp=dynamic_object.timestamp,
                                                               cartesian_state=dynamic_object.cartesian_state,
                                                               map_state=MapState(lane_fstate, lane_id),
                                                               size=dynamic_object.size,
                                                               confidence=dynamic_object.confidence,
                                                               off_map=dynamic_object.off_map,
                                                               is_ghost=True))

        return projected_dynamic_objects

    @staticmethod
    @prof.ProfileFunction()
    def _add_road_semantics(dynamic_objects: List[DynamicObject],
                            extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame],
                            projected_ego_fstates: Dict[RelativeLane, FrenetState2D], logger: Logger) -> \
            List[DynamicObjectWithRoadSemantics]:
        """
        Wraps DynamicObjects with "on-road" information (relative progress on road wrt ego, road-localization and more).
        This is a temporary function that caches relevant metrics for re-use. Should be removed after an efficient
        representation of DynamicObject.
        :param dynamic_objects: list of relevant DynamicObjects to calculate "on-road" metrics for.
        :param extended_lane_frames: dictionary from RelativeLane to the corresponding GeneralizedFrenetSerretFrame
        :param projected_ego_fstates: dictionary from RelativeLane to ego Frenet state, which is ego projected on the
                corresponding extended_lane_frame
        :param logger:
        :return: list of object of type DynamicObjectWithRoadSemantics
        """
        # filter out off map dynamic objects
        on_map_dynamic_objects = [obj for obj in dynamic_objects if not obj.off_map]

        # Create projected objects as needed
        projected_dynamic_objects = BehavioralGridState._create_projected_objects(on_map_dynamic_objects,
                                                                                  extended_lane_frames, logger)

        # Filter irrelevant objects
        relevant_objects, relevant_objects_relative_lanes = BehavioralGridState._filter_irrelevant_dynamic_objects(
            on_map_dynamic_objects + projected_dynamic_objects, extended_lane_frames)

        relevant_objects_map_states = [obj.map_state for obj in relevant_objects]

        longitudinal_differences = BehavioralGridState._calculate_longitudinal_differences(
            extended_lane_frames, projected_ego_fstates, relevant_objects_map_states)

        return [DynamicObjectWithRoadSemantics(obj, longitudinal_differences[i], relevant_objects_relative_lanes[i])
                for i, obj in enumerate(relevant_objects)]

    @staticmethod
    def _filter_irrelevant_dynamic_objects(dynamic_objects: List[DynamicObject],
                                           extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame]) -> \
            Tuple[List[DynamicObject], List[List[RelativeLane]]]:
        """
        Filters dynamic objects that are on a lane segment that is not contained in any GFF

        :param dynamic_objects: list of dynamic objects
        :param extended_lane_frames: dictionary from RelativeLane to the corresponding GeneralizedFrenetSerretFrame
        :return: two related lists; the first list contains the relevant dynamic objects, and the other contains lists of
                 relative lanes for each relevant dynamic object
        """
        # Get lane segment ID for each dynamic object
        objects_segment_ids = np.array([obj.map_state.lane_id for obj in dynamic_objects])

        # Create boolean matrix that is true when a vehicle is in a relative lane
        objects_lane_matrix = np.array([], dtype=np.bool).reshape(0, len(objects_segment_ids))

        # calculate relative to ego lane (RIGHT, SAME, LEFT) for every object
        for extended_lane_frame in extended_lane_frames.values():
            # find all dynamic objects that belong to the current unified frame
            # add as row to matrix
            relevant_object_mask = extended_lane_frame.has_segment_ids(objects_segment_ids)
            objects_lane_matrix = np.vstack([objects_lane_matrix, relevant_object_mask])

        # Boolean array that is True when an object is in any relative lane
        is_relevant_object = objects_lane_matrix.any(axis=0)

        # Collect relevant object data
        relevant_objects = list(np.array(dynamic_objects)[is_relevant_object])
        relevant_objects_lane_matrix = objects_lane_matrix[:, is_relevant_object]
        relative_lane_keys = np.array(list(extended_lane_frames.keys()))
        relevant_objects_relative_lanes = [list(relative_lane_keys[relevant_objects_lane_matrix[:, i]])
                                           for i in range(len(relevant_objects))]

        return relevant_objects, relevant_objects_relative_lanes

    def calculate_longitudinal_differences(self, target_map_states: List[MapState]) -> np.array:
        """
        Given target segment ids and segment fstates, calculate longitudinal differences between the targets and ego
        projected on the target lanes, using the relevant unified frames (GFF).
        :param target_map_states: list of original map states of the targets
        :return: array of longitudinal differences between the targets and projected ego per target
        """
        return BehavioralGridState._calculate_longitudinal_differences(
            self.extended_lane_frames, self.projected_ego_fstates, target_map_states)

    def _get_max_speed_limit(self) -> float:
        """
        Iterates over all lanes in the state's GFFs and returns the maximal speed limit
        :return: maximal speed limit in [m/s]
        """
        lane_id_by_gff = [gff.segment_ids.tolist() for gff in self.extended_lane_frames.values()]
        unique_lane_ids = set(itertools.chain.from_iterable(lane_id_by_gff))
        return max([MapUtils.get_lane(lane_id).e_v_nominal_speed for lane_id in unique_lane_ids])

    @staticmethod
    def _calculate_longitudinal_differences(extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame],
                                            ego_gff_fstates: Dict[RelativeLane, FrenetState2D],
                                            target_map_states: List[MapState]) -> np.array:
        """
        Given unified frames, ego projected on the unified frames, target segment ids and segment fstates, calculate
        longitudinal differences between the targets and ego.
        projected on the target lanes, using the relevant unified frames (GFF).
        :param extended_lane_frames: mapping between 3 lanes relative to the host vehicle (left adjacent, same,
                                                                        right adjacent) to their curve representation
        :param ego_gff_fstates: dictionary from RelativeLane to ego Frenet state, which is ego projected on the
                corresponding extended_lane_frame
        :param target_map_states: list of original map states of the targets
        :return: array of longitudinal differences between the targets and projected ego
        """
        target_segment_ids = np.array([map_state.lane_id for map_state in target_map_states])
        target_segment_fstates = np.array([map_state.lane_fstate for map_state in target_map_states])

        # initialize longitudinal_differences to infinity
        longitudinal_differences = np.full(len(target_segment_ids), np.inf)

        # longitudinal difference between object and ego at t=0 (positive if obj in front of ego)
        for rel_lane, extended_lane_frame in extended_lane_frames.items():  # loop over at most 3 GFFs
            # find all targets belonging to the current unified frame
            relevant_idxs = extended_lane_frame.has_segment_ids(target_segment_ids)
            if relevant_idxs.any():
                # convert relevant dynamic objects to fstate w.r.t. the current GFF
                target_gff_fstates = extended_lane_frame.convert_from_segment_states(
                    target_segment_fstates[relevant_idxs], target_segment_ids[relevant_idxs])
                # calculate longitudinal distances between the targets from this extended frame and ego projected on it
                longitudinal_differences[relevant_idxs] = \
                    target_gff_fstates[:, FS_SX] - ego_gff_fstates[rel_lane][FS_SX]

        return longitudinal_differences

    @staticmethod
    @prof.ProfileFunction()
    def _create_generalized_frenet_frames(ego_state: EgoState, route_plan: RoutePlan, logger: Logger) -> \
            Dict[RelativeLane, GeneralizedFrenetSerretFrame]:
        """
        For all available nearest lanes create a corresponding generalized frenet frame (long enough) that can
        contain multiple original lane segments. Augmented frenet frames may be created if there are lane splits ahead.
        :param ego_state: ego state from scene_dynamic
        :param route_plan: the route plan which contains lane costs
        :param logger: Logger object to log warning messages
        :return: dictionary from RelativeLane to GeneralizedFrenetSerretFrame
        """
        # calculate unified generalized frenet frames
        ego_lane_id = ego_state.map_state.lane_id
        closest_lanes_dict = MapUtils.get_closest_lane_ids(ego_lane_id)  # Dict: RelativeLane -> lane_id

        # Augmented GFFS can be created only if the lanes don't currently exist
        can_augment = [rel_lane for rel_lane in [RelativeLane.LEFT_LANE, RelativeLane.RIGHT_LANE]
                       if rel_lane not in closest_lanes_dict.keys()]

        extended_lane_frames = {}

        # Create generalized Frenet frame for the host's lane
        try:
            lane_gff_dict = BehavioralGridState._get_generalized_frenet_frames(
                lane_id=closest_lanes_dict[RelativeLane.SAME_LANE], station=ego_state.map_state.lane_fstate[FS_SX],
                route_plan=route_plan, logger=logger, can_augment=can_augment)
        except MappingException as e:
            # in case of failure to build GFF for SAME_LANE, stop processing this BP frame
            raise AssertionError(
                "Trying to fetch data for %s, but data is unavailable. %s" % (RelativeLane.SAME_LANE, str(e)))

        # set the SAME_LANE first since it cannot be augmented
        extended_lane_frames[RelativeLane.SAME_LANE] = lane_gff_dict[RelativeLane.SAME_LANE]

        host_cartesian_point = np.array([ego_state.cartesian_state[C_X],
                                         ego_state.cartesian_state[C_Y]])

        # If an adjacent lane exists, create a generalized Frenet frame for it
        for relative_lane in [RelativeLane.LEFT_LANE, RelativeLane.RIGHT_LANE]:
            # can_augment is True only if the adjacent lane does not exist. Therefore, the only thing that can be done is to
            # create an augmented GFF
            if relative_lane in can_augment:
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
                    lane_gffs = BehavioralGridState._get_generalized_frenet_frames(
                        lane_id=closest_lanes_dict[relative_lane], station=host_station_in_adjacent_lane,
                        route_plan=route_plan, logger=logger)

                    # Note that the RelativeLane keys that are in the returned dictionary from _get_lookahead_frenet_frames are
                    # with respect to the lane ID provided to the function. Therefore, since the lane ID for the left/right lane is
                    # provided to the function above, the RelativeLane.SAME_LANE key in the returned dictionary actually refers to the
                    # left/right lane. That makes the use of this key below correct.
                    extended_lane_frames[relative_lane] = lane_gffs[RelativeLane.SAME_LANE]
                except MappingException as e:
                    logger.warning("Trying to fetch data for %s, but data is unavailable. %s" % (relative_lane, str(e)))

        return extended_lane_frames

    @staticmethod
    @raises(LaneNotFound, RoadNotFound, UpstreamLaneNotFound, StraightConnectionNotFound)
    @prof.ProfileFunction()
    def _get_generalized_frenet_frames(lane_id: int, station: float, route_plan: RoutePlan, logger: Logger,
                                       forward_horizon: float = MAX_FORWARD_HORIZON,
                                       backward_horizon: float = MAX_BACKWARD_HORIZON,
                                       can_augment: Optional[List[RelativeLane]] = None) -> \
            Dict[RelativeLane, GeneralizedFrenetSerretFrame]:
        """
        Create Generalized Frenet frame(s) along lane center, starting from given lane and station. If augmented lanes can be created, they will
        be returned in the dictionary under the RelativeLane.LEFT_LANE/RIGHT_LANE keys.
        :param lane_id: starting lane_id
        :param station: starting station [m]
        :param route_plan: the relevant navigation plan to iterate over its road IDs.
        :param logger: Logger object to log warning messages
        :param can_augment: List of RelativeLane. All relative lanes inside this can be augmented.
        :return: Dict of generalized Frenet frames with the relative lane as keys
                 The relative lane key is with respect to the provided lane_id. The dictionary will always contain the GFF for the provided
                 lane_id, and the RelativeLane.SAME_LANE key can be used to access it. If possible, augmented GFFs will also be returned,
                 and they can be accessed with the respective relative lane key.
        """
        if station < backward_horizon:
            # If the given station is not far enough along the lane, then the backward horizon will pass the beginning of the lane, and the
            # upstream lane subsegments need to be found. The starting station for the forward lookahead should be the beginning of the
            # current lane, and the forward lookahead distance should include the maximum forward horizon ahead of the given station and
            # the backward distance to the beginning of the lane (i.e. the station).
            starting_station = 0.0
            lookahead_distance = forward_horizon + station
            upstream_lane_subsegments = BehavioralGridState._get_upstream_lane_subsegments(lane_id, station,
                                                                                           backward_horizon)
        else:
            # If the given station is far enough along the lane, then the backward horizon will not pass the beginning of the lane. In this
            # case, the starting station for the forward lookahead should be the end of the backward horizon, and the forward lookahead
            # distance should include the maximum forward and backward horizons ahead of and behind the given station, respectively. In
            # other words, if we're at station = 150 m on a lane and the maximum forward and backward horizons are 400 m and 100 m,
            # respectively, then starting station = 50 m and forward lookahead distance = 400 + 100 = 500 m. This is the case where the GFF
            # does not include any upstream lanes.
            starting_station = station - backward_horizon
            lookahead_distance = forward_horizon + backward_horizon
            upstream_lane_subsegments = []

        lane_subsegments_dict = BehavioralGridState._get_downstream_lane_subsegments(initial_lane_id=lane_id,
                                                                                     initial_s=starting_station,
                                                                                     lookahead_distance=lookahead_distance,
                                                                                     route_plan=route_plan,
                                                                                     logger=logger,
                                                                                     can_augment=can_augment)

        gffs_dict = {}

        # Build GFFs from the Frenet Subsegments for the lane/augmented lanes that were created.
        for rel_lane in lane_subsegments_dict:
            lane_subsegments, is_partial, is_augmented, _ = lane_subsegments_dict[rel_lane]
            gff_type = GFFType.get(is_partial, is_augmented)

            concatenated_lane_subsegments = upstream_lane_subsegments + lane_subsegments

            # Create Frenet frame for each sub segment
            frenet_frames = [MapUtils.get_lane_frenet_frame(lane_subsegment.e_i_SegmentID)
                             for lane_subsegment in concatenated_lane_subsegments]

            # Create GFF
            gffs_dict[rel_lane] = GeneralizedFrenetSerretFrame.build(frenet_frames, concatenated_lane_subsegments, gff_type)

        return gffs_dict

    @staticmethod
    @raises(UpstreamLaneNotFound, LaneNotFound)
    def _get_upstream_lane_subsegments(initial_lane_id: int, initial_station: float, backward_distance: float) -> List[FrenetSubSegment]:
        """
        Return a list of lane subsegments that are upstream to the given lane and extending as far back as backward_distance
        :param initial_lane_id: ID of lane to start from
        :param initial_station: Station on given lane
        :param backward_distance: Distance [m] to look backwards
        :return: List of upstream lane subsegments
        """
        lane_id = initial_lane_id
        upstream_distance = initial_station
        upstream_lane_subsegments = []

        while upstream_distance < backward_distance:
            # First, choose an upstream lane
            upstream_lane_ids = MapUtils.get_upstream_lane_ids(lane_id)
            num_upstream_lanes = len(upstream_lane_ids)

            if num_upstream_lanes == 0:
                raise UpstreamLaneNotFound("Upstream lane not found for lane_id=%d" % (lane_id))
            elif num_upstream_lanes == 1:
                chosen_upstream_lane_id = upstream_lane_ids[0]
            elif num_upstream_lanes > 1:
                # If there are multiple upstream lanes and one of those lanes has a STRAIGHT_CONNECTION maneuver type, choose that lane to
                # follow. Otherwise, default to choosing the first upstream lane in the list.
                chosen_upstream_lane_id = upstream_lane_ids[0]
                upstream_lane_maneuver_types = MapUtils.get_upstream_lane_maneuver_types(lane_id)

                for upstream_lane_id in upstream_lane_ids:
                    if upstream_lane_maneuver_types[upstream_lane_id] == ManeuverType.STRAIGHT_CONNECTION:
                        chosen_upstream_lane_id = upstream_lane_id
                        break

            # Second, determine the start and end stations for the subsegment
            end_station = MapUtils.get_lane(chosen_upstream_lane_id).e_l_length
            upstream_distance += end_station
            start_station = max(0.0, upstream_distance - backward_distance)

            # Third, create and append the upstream lane subsegment
            upstream_lane_subsegments.append(FrenetSubSegment(chosen_upstream_lane_id, start_station, end_station))

            # Last, set lane for next loop
            lane_id = chosen_upstream_lane_id

        # Before returning, reverse the order of the subsegments so that they are in the order that they would have been traveled on. In
        # other words, the first subsegment should be the furthest from the host, and the last subsegment should be the closest to the host.
        upstream_lane_subsegments.reverse()

        return upstream_lane_subsegments

    @staticmethod
    @raises(StraightConnectionNotFound, RoadNotFound, LaneNotFound)
    def _get_downstream_lane_subsegments(initial_lane_id: int, initial_s: float, lookahead_distance: float,
                                         route_plan: RoutePlan, logger: Logger, can_augment: Optional[List[RelativeLane]] = None) -> \
            Dict[RelativeLane, Tuple[List[FrenetSubSegment], bool, bool, float]]:
        """
        Given a longitudinal position <initial_s> on lane segment <initial_lane_id>, advance <lookahead_distance>
        further according to costs of each FrenetFrame, and finally return a configuration of lane-subsegments.
        If <desired_lon> is more than the distance to end of the plan, a LongitudeOutOfRoad exception is thrown.
        :param initial_lane_id: the initial lane_id (the vehicle is current on)
        :param initial_s: initial longitude along <initial_lane_id>
        :param lookahead_distance: the desired distance of lookahead in [m].
        :param route_plan: the relevant navigation plan to iterate over its road IDs.
        :param logger: Logger object to log warning messages
        :param can_augment: List of RelativeLane. All relative lanes inside this can be augmented.
        :return: Dictionary with potential keys: [RelativeLane.SAME_LANE, RelativeLane.LEFT_LANE, RelativeLane.RIGHT_LANE]
                 These keys represent the non-augmented, left-augmented, and right-augmented gffs that can be created.
                 The key-value pair for the non-augmented lane (i.e. RelativeLane.SAME_LANE) will always exist, and it refers
                 to the provided initial_lane_id. The left-augmented and right-augmented keys (i.e. RelativeLane.LEFT_LANE
                 and RelativeLane.RIGHT_LANE) will only exist when an augmented GFF can be created. The values are tuples
                 that contain a list of FrenetSubSegments that will be used to create the GFF and two flags that denote the
                 GFF type. The first flag denotes a partial GFF and the second flag denotes an augmented GFF.
                 Lastly, the total length of the GFF is returned
        """
        # initialize default arguments
        can_augment = can_augment or []

        lane_subsegments_dict = {}

        lane_subsegments, cumulative_distance = MapUtils._advance_on_plan(
            initial_lane_id, initial_s, lookahead_distance, route_plan, logger)

        current_lane_id = lane_subsegments[-1].e_i_SegmentID
        valid_downstream_lanes = MapUtils._get_valid_downstream_lanes(current_lane_id, route_plan)

        # traversal reached the end of desired horizon
        if cumulative_distance >= lookahead_distance:
            lane_subsegments_dict[RelativeLane.SAME_LANE] = (lane_subsegments, False, False, cumulative_distance)
            return lane_subsegments_dict

        # a dead-end reached (no downstream lanes at all)
        if len(valid_downstream_lanes) == 0:
            lane_subsegments_dict[RelativeLane.SAME_LANE] = (lane_subsegments, True, False, cumulative_distance)
            return lane_subsegments_dict

        augmented = []  # store all relative lanes that are actually augmented in this code block
        # Deal with "splitting" direction (create augmented lane) in the split #
        for rel_lane, maneuver_type in [(RelativeLane.RIGHT_LANE, ManeuverType.RIGHT_SPLIT),
                                        (RelativeLane.LEFT_LANE, ManeuverType.LEFT_SPLIT)]:
            # Check if augmented lanes can be created
            if (rel_lane in can_augment) and maneuver_type in valid_downstream_lanes:
                # recursive call to construct the augmented ("take split", left/right) sequence of lanes (GFF)
                augmented_lane_dict = BehavioralGridState._get_downstream_lane_subsegments(
                    initial_lane_id=valid_downstream_lanes[maneuver_type], initial_s=0.0,
                    lookahead_distance=lookahead_distance - cumulative_distance, route_plan=route_plan, logger=logger)

                # Get returned information. Note that the use of the RelativeLane.SAME_LANE key here is correct.
                # Read the return value description above for more information.
                augmented_lane_subsegments, is_augmented_partial, _, augmented_cumulative_distance = \
                    augmented_lane_dict[RelativeLane.SAME_LANE]

                # Assign information to dictionary accordingly
                lane_subsegments_dict[rel_lane] = (lane_subsegments + augmented_lane_subsegments, is_augmented_partial,
                                                   True, cumulative_distance + augmented_cumulative_distance)
                augmented.append(rel_lane)

        # remove the already augmented lanes from options to augment after taking the straight connection
        can_still_augment = [rel_lane for rel_lane in can_augment if rel_lane not in augmented]

        # Deal with "straight" direction in the split #
        if ManeuverType.STRAIGHT_CONNECTION in valid_downstream_lanes:
            # recursive call to construct the "keep straight" sequence of lanes (GFF)
            straight_lane_dict = BehavioralGridState._get_downstream_lane_subsegments(
                initial_lane_id=valid_downstream_lanes[ManeuverType.STRAIGHT_CONNECTION], initial_s=0.0,
                lookahead_distance=lookahead_distance - cumulative_distance, route_plan=route_plan, logger=logger,
                can_augment=can_still_augment)

            for rel_lane, _ in straight_lane_dict.items():
                # Get returned information.
                straight_lane_subsegments, is_straight_partial, is_straight_augmented, straight_cumulative_distance = \
                    straight_lane_dict[rel_lane]

                # Concatenate and assign information to dictionary accordingly
                lane_subsegments_dict[rel_lane] = (lane_subsegments + straight_lane_subsegments, is_straight_partial,
                                                   is_straight_augmented,
                                                   cumulative_distance + straight_cumulative_distance)
        else:
            raise StraightConnectionNotFound("Straight downstream connection not found for lane=%d", current_lane_id)

        return lane_subsegments_dict

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
    def _get_longitudinal_grid_cell(obj: DynamicObjectWithRoadSemantics, ego_state: EgoState):
        """
        Given a dynamic object representation and ego state, calculate what is the proper longitudinal
        relative-grid-cell to project it on. An object is set to be in FRONT cell if the distance from its rear to ego's
        front is greater than LON_MARGIN_FROM_EGO.
        An object is set to be in REAR cell if the distance from its front to ego's rear is greater
        than LON_MARGIN_FROM_EGO. Otherwise, the object is set to be parallel to ego.
        (positive if object is in front). difference is computed between objects'-centers
        :param obj: the object to project onto the ego-relative grid
        :param ego_state: ego state for localization and size
        :return: RelativeLongitudinalPosition enum's value representing the longitudinal projection on the relative-grid
        """
        obj_length = obj.dynamic_object.size.length
        ego_length = ego_state.size.length
        if obj.longitudinal_distance > (obj_length / 2 + ego_length / 2 + LON_MARGIN_FROM_EGO):
            return RelativeLongitudinalPosition.FRONT
        elif obj.longitudinal_distance < -(obj_length / 2 + ego_length / 2 + LON_MARGIN_FROM_EGO):
            return RelativeLongitudinalPosition.REAR
        else:
            return RelativeLongitudinalPosition.PARALLEL

    def update_dim_state(self) -> None:
        """
        Update DIM state machine of ego, using reference_route (current-lane GFF)
        """
        ego_lane_fstate = self.ego_state.map_state.lane_fstate
        ego_lane_id = self.ego_state.map_state.lane_id
        ego_s = self.extended_lane_frames[RelativeLane.SAME_LANE].convert_from_segment_state(ego_lane_fstate, ego_lane_id)[FS_SX]

        self.ego_state.update_dim_state(ego_s, self.get_closest_stop_bar(RelativeLane.SAME_LANE),
                                        self.get_ignored_stop_bar_distance(RelativeLane.SAME_LANE))

    @staticmethod
    def _log_grid_data(multi_object_grid: Dict[SemanticGridCell, List[DynamicObjectWithRoadSemantics]],
                       timestamp_in_sec: float, logger: Logger):
        """
        Write to log front object ID, its velocity and the distance from ego
        :param multi_object_grid: dictionary of the behavioral grid: from cell to objects' list
        :param timestamp_in_sec: current state time
        :param logger:
        """
        if logger is None:
            return
        front_cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)
        front_obj = None
        front_obj_dist = 0  # write zeros if there is no front object

        if front_cell in multi_object_grid and len(multi_object_grid[front_cell]) > 0:
            front_obj = multi_object_grid[front_cell][0].dynamic_object
            front_obj_dist = multi_object_grid[front_cell][0].longitudinal_distance
        logger.debug("%s: time %f, dist_from_front_object %f, front_object: %s" %
                     (LOG_MSG_BEHAVIORAL_GRID, timestamp_in_sec, front_obj_dist, front_obj))

    def get_closest_stop_bar(self, relative_lane: RelativeLane) -> Tuple[TrafficControlBar, float]:
        """
        Returns the closest stop bar and its distance.
        :param relative_lane: in the GFF
        :return: tuple of (closest stop bar, its distance.)
        """
        return self.tcb_in_gff_and_their_distances[relative_lane] \
            if relative_lane in self.tcb_in_gff_and_their_distances else None

    def get_ignored_stop_bar_distance(self, relative_lane: RelativeLane) -> float:
        """
        Returns ignored stop bar distance.
        :param relative_lane: in the GFF
        :return: ignored stop bar distance
        """
        return self.ignored_tcb_distance_in_gff[relative_lane]

    @staticmethod
    def _get_closest_stop_bars(extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame],
                               projected_ego_fstates: Dict[RelativeLane, FrenetState2D],
                               stop_bar_id_to_ignore: int = None, logger: Logger = None) \
            -> Dict[RelativeLane, Tuple[Tuple[TrafficControlBar, float], float]]:
        """
        at object construction, find the closest stop bars to the ego per BGS lane.
        Life span is a BP cycle
        :param extended_lane_frames: of the BGS
        :param projected_ego_fstates: ego projection on frenet lanes
        :param stop_bar_to_ignore: id of stop bar that is ignored during DIM
        :param logger:
        :return: dictionary of the closest stop bars with its s-distance to the ego, plus the ignored TCB distance per BGS lane
        """
        bars_per_lane = {}
        for relative_lane, target_lane in extended_lane_frames.items():
            ego_location = projected_ego_fstates[relative_lane][FS_SX]
            if logger is not None:
                logger.debug("Stop bar check for lane %s", relative_lane)  # more logger info inside get_closest_stop_bar
            bars_per_lane[relative_lane] = MapUtils.get_closest_stop_bar(extended_lane_frames[relative_lane],
                                                                         ego_location, DIM_MARGIN_TO_STOP_BAR,
                                                                         stop_bar_id_to_ignore, logger)
        return bars_per_lane

    @staticmethod
    def _is_object_in_lane(dynamic_object: DynamicObject, gff: GeneralizedFrenetSerretFrame, logger: Logger) -> bool:
        """
        Checks if any corner of an object's bounding box or the object's center of mass is inside another gff.
        Checks if the distance from that point to the nominal path point of the lane is less than
        the nominal point's left/right offset.
        :param dynamic_object: object to be checked
        :param gff: gff of interest
        :param logger:
        :return: Bool representing if any bounding box corner or center of mass point is in the lane of interest
        """
        # skip checks if object's assigned lane is in the gff and it is not off map
        obj_lane_id = dynamic_object.map_state.lane_id
        if gff.has_segment_id(obj_lane_id) and not dynamic_object.off_map:
            return True

        # determine if object is to the right or left of GFF
        # (if object's adjacent left lane id is in a gff, the object is to the right of the gff)
        in_right = np.any(gff.has_segment_ids(np.array(MapUtils.get_adjacent_lane_ids(obj_lane_id, RelativeLane.LEFT_LANE))))
        in_left = np.any(gff.has_segment_ids(np.array(MapUtils.get_adjacent_lane_ids(obj_lane_id, RelativeLane.RIGHT_LANE))))
        offset_side = RelativeLane.RIGHT_LANE if in_right else RelativeLane.LEFT_LANE

        # If an object is in both the left and right lane (based only on lane ID's), there may be an error in the map.
        if not (in_left ^ in_right):
            logger.warning(f"Object {dynamic_object.obj_id}'s lane ({dynamic_object.map_state.lane_id}) was found in "
                           f"both the left and right GFFs.")

        bbox = dynamic_object.bounding_box()

        obj_gff_fstates = []
        # Try to project the center of the vehicle. If OutOfSegment, skip checking the rest of the points
        try:
            obj_center_fpoint = gff.cpoint_to_fpoint(np.array([dynamic_object.x, dynamic_object.y]))
            obj_gff_fstates.append(np.array([obj_center_fpoint[FP_SX], 0, 0, obj_center_fpoint[FP_DX], 0, 0]))
        except (OutOfSegmentBack, OutOfSegmentFront):
            logger.debug(f"OutOfSegment for the centerpoint of object {dynamic_object.obj_id} when checking occupancy."
                         f"Projection will not be done for this object.")
            return False


        # A loop is used to convert the cpoints to fpoints instead of the vectorized function to skip the OutOfSegment points
        for point in bbox:
            try:
                obj_gff_fpoint = gff.cpoint_to_fpoint(point)
                obj_gff_fstates.append(np.array([obj_gff_fpoint[FP_SX], 0, 0, obj_gff_fpoint[FP_DX], 0, 0]))
            except OutOfSegmentBack:
                logger.debug(f"OutOfSegmentBack for object {dynamic_object.obj_id} when checking occupancy "
                             f"at point {str(point)}. Point will be skipped")
            except OutOfSegmentFront:
                logger.debug(f"OutOfSegmentFront for object {dynamic_object.obj_id} when checking occupancy "
                             f"at point {str(point)}. Point will be skipped")

        gff_lane_ids, gff_lane_fstates = gff.convert_to_segment_states(np.array(obj_gff_fstates))
        # check if the object spans across multiple lane segments
        if len(set(gff_lane_ids)) > 1:
            # if points belong to multiple lane_ids, group the calls by lane_id
            gff_lane_ids = np.array(gff_lane_ids)
            borders_right = []
            borders_left = []

            for id in set(gff_lane_ids):
                id_indices = np.where(gff_lane_ids == id)
                id_borders_right, id_borders_left = MapUtils.get_dists_to_lane_borders(id, np.squeeze(gff_lane_fstates[id_indices, FS_SX]))
                borders_right.extend(id_borders_right)
                borders_left.extend(id_borders_left)
        else:
            # if all points are in the same lane_id, only one call is needed
            borders_right, borders_left = MapUtils.get_dists_to_lane_borders(gff_lane_ids[0], gff_lane_fstates[:, FS_SX])

        borders_width = borders_right if offset_side == RelativeLane.RIGHT_LANE else borders_left

        return np.any(np.abs(gff_lane_fstates[:, FS_DX]) < borders_width)

