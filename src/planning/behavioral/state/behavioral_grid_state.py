from collections import defaultdict
from logging import Logger
from typing import Dict, List, Tuple, Optional

import numpy as np
import rte.python.profiler as prof
from decision_making.src.exceptions import MappingException, OutOfSegmentBack, OutOfSegmentFront, LaneNotFound, \
    RoadNotFound, raises, StraightConnectionNotFound, UpstreamLaneNotFound, NavigationPlanTooShort
from decision_making.src.global_constants import LON_MARGIN_FROM_EGO, PLANNING_LOOKAHEAD_DIST, MAX_BACKWARD_HORIZON, \
    MAX_FORWARD_HORIZON, LOG_MSG_BEHAVIORAL_GRID
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
from decision_making.src.planning.types import FS_SX, FrenetState2D, FP_SX, C_X, C_Y
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, GFFType, \
    FrenetSubSegment
from decision_making.src.planning.behavioral.state.map_state import MapState
from decision_making.src.planning.behavioral.state.state import DynamicObject, EgoState
from decision_making.src.planning.behavioral.state.state import State
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.messages.scene_static_enums import ManeuverType



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

        projected_ego_fstates = {rel_lane: extended_lane_frames[rel_lane].cstate_to_fstate(state.ego_state.cartesian_state)
                                 for rel_lane in extended_lane_frames}

        # TODO: Make sure to account for all relevant actors on all upstream lanes. For example, there may be an actor outside of an
        #  upstream merge area that is moving quickly. If we predict that this actor will come close to the host, we have to consider
        #  it as well.

        # Dict[SemanticGridCell, List[DynamicObjectWithRoadSemantics]]
        dynamic_objects_with_road_semantics = \
            sorted(BehavioralGridState._add_road_semantics(state.dynamic_objects, extended_lane_frames, projected_ego_fstates),
                   key=lambda rel_obj: abs(rel_obj.longitudinal_distance))

        multi_object_grid = BehavioralGridState._project_objects_on_grid(dynamic_objects_with_road_semantics,
                                                                         state.ego_state)

        BehavioralGridState._log_grid_data(multi_object_grid, state.ego_state.timestamp_in_sec, logger)

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

        # Create boolean matrix that is true when a vehicle is in a relative lane
        vehicle_lane_matrix = np.array([], dtype=np.bool).reshape(0, len(on_map_dynamic_objects))

        # calculate relative to ego lane (RIGHT, SAME, LEFT) for every object
        for _, extended_lane_frame in extended_lane_frames.items():
            # find all dynamic objects that belong to the current unified frame
            # add as row to matrix
            relevant_object_mask = extended_lane_frame.has_segment_ids(objects_segment_ids)
            vehicle_lane_matrix = np.vstack([vehicle_lane_matrix, relevant_object_mask])

        longitudinal_differences = BehavioralGridState._calculate_longitudinal_differences(
            extended_lane_frames, projected_ego_fstates, object_map_states)

        # get the lanes an object is in by looking at the columns of the matrix
        return [DynamicObjectWithRoadSemantics(obj, longitudinal_differences[i],
                                               np.array(list(extended_lane_frames.keys()))[vehicle_lane_matrix[:,i]].tolist())
                for i, obj in enumerate(on_map_dynamic_objects) if vehicle_lane_matrix[:,i].any()]


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
    def _create_generalized_frenet_frames(ego_state: EgoState, route_plan: RoutePlan, logger: Logger) -> \
            Dict[RelativeLane, GeneralizedFrenetSerretFrame]:
        """
        For all available nearest lanes create a corresponding generalized frenet frame (long enough) that can
        contain multiple original lane segments. Augmented frenet frames may be created if there are lane splits ahead.
        :param state: the current world state
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
                lane_id=closest_lanes_dict[RelativeLane.SAME_LANE], station=state.ego_state.map_state.lane_fstate[FS_SX],
                route_plan=route_plan, can_augment=can_augment)
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
                        lane_id=closest_lanes_dict[relative_lane], station=host_station_in_adjacent_lane, route_plan=route_plan)

                    # Note that the RelativeLane keys that are in the returned dictionary from _get_lookahead_frenet_frames are
                    # with respect to the lane ID provided to the function. Therefore, since the lane ID for the left/right lane is
                    # provided to the function above, the RelativeLane.SAME_LANE key in the returned dictionary actually refers to the
                    # left/right lane. That makes the use of this key below correct.
                    extended_lane_frames[relative_lane] = lane_gffs[RelativeLane.SAME_LANE]
                except MappingException as e:
                    logger.warning("Trying to fetch data for %s, but data is unavailable. %s" % (relative_lane, str(e)))

        return extended_lane_frames

    @staticmethod
    @raises(LaneNotFound, RoadNotFound, UpstreamLaneNotFound, StraightConnectionNotFound, NavigationPlanTooShort)
    @prof.ProfileFunction()
    def _get_generalized_frenet_frames(lane_id: int, station: float, route_plan: RoutePlan,
                                       can_augment: Optional[List[RelativeLane]] = None) -> \
            Dict[RelativeLane, GeneralizedFrenetSerretFrame]:
        """
        Create Generalized Frenet frame(s) along lane center, starting from given lane and station. If augmented lanes can be created, they will
        be returned in the dictionary under the RelativeLane.LEFT_LANE/RIGHT_LANE keys.
        :param lane_id: starting lane_id
        :param station: starting station [m]
        :param route_plan: the relevant navigation plan to iterate over its road IDs.
        :param can_augment: List of RelativeLane. All relative lanes inside this can be augmented.
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
            upstream_lane_subsegments = BehavioralGridState._get_upstream_lane_subsegments(lane_id, station, MAX_BACKWARD_HORIZON)
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

        lane_subsegments_dict = BehavioralGridState._get_downstream_lane_subsegments(initial_lane_id=lane_id, initial_s=starting_station,
                                                                                     lookahead_distance=lookahead_distance, route_plan=route_plan,
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
    @raises(StraightConnectionNotFound, RoadNotFound, NavigationPlanTooShort, LaneNotFound)
    def _get_downstream_lane_subsegments(initial_lane_id: int, initial_s: float, lookahead_distance: float,
                                         route_plan: RoutePlan, can_augment: Optional[List[RelativeLane]] = None) -> \
            Dict[RelativeLane, Tuple[List[FrenetSubSegment], bool, bool, float]]:
        """
        Given a longitudinal position <initial_s> on lane segment <initial_lane_id>, advance <lookahead_distance>
        further according to costs of each FrenetFrame, and finally return a configuration of lane-subsegments.
        If <desired_lon> is more than the distance to end of the plan, a LongitudeOutOfRoad exception is thrown.
        :param initial_lane_id: the initial lane_id (the vehicle is current on)
        :param initial_s: initial longitude along <initial_lane_id>
        :param lookahead_distance: the desired distance of lookahead in [m].
        :param route_plan: the relevant navigation plan to iterate over its road IDs.
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
            initial_lane_id, initial_s, lookahead_distance, route_plan)

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
                    lookahead_distance=lookahead_distance - cumulative_distance, route_plan=route_plan)

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
                lookahead_distance=lookahead_distance - cumulative_distance, route_plan=route_plan,
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