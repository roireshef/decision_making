import numpy as np
from logging import Logger
import copy

from decision_making.src.exceptions import raises, RoadNotFound, NavigationPlanTooShort, \
    UpstreamLaneNotFound, LaneNotFound, OutOfSegmentBack, OutOfSegmentFront, EquivalentStationNotFound, \
    IDAppearsMoreThanOnce, StraightConnectionNotFound
from decision_making.src.global_constants import EPS, LANE_END_COST_IND, MAX_BACKWARD_HORIZON, \
    MAX_FORWARD_HORIZON, FLOAT_MAX, MAX_STATION_DIFFERENCE, LANE_OCCUPANCY_COST_IND, SATURATED_COST
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.scene_static_message import SceneLaneSegmentGeometry, \
    SceneLaneSegmentBase, SceneRoadSegment
from decision_making.src.messages.scene_static_enums import NominalPathPoint, RoadObjectType
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import CartesianPoint2D, FS_SX, FP_SX, SIGN_TYPE, SIGN_S
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, \
    FrenetSubSegment
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.scene.scene_static_model import SceneStaticModel
import rte.python.profiler as prof
from typing import List, Dict, Optional, Tuple
from decision_making.src.messages.scene_static_enums import ManeuverType
from decision_making.src.planning.types import LaneSegmentID
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GFF_Type



class MapUtils:

    @staticmethod
    def get_road_segment_ids() -> List[int]:
        """
        :return:road_segment_ids of every road in the static scene
        """
        scene_static = SceneStaticModel.get_instance().get_scene_static()
        road_segments = scene_static.s_Data.s_SceneStaticBase.as_scene_road_segment[:scene_static.s_Data.s_SceneStaticBase.e_Cnt_num_road_segments]
        return [road_segment.e_i_road_segment_id for road_segment in road_segments]

    @staticmethod
    def get_road_segment_id_from_lane_id(lane_id: int) -> int:
        """
        get road_segment_id containing the lane
        :param lane_id:
        :return: road_segment_id
        """
        lane = MapUtils.get_lane(lane_id)
        if lane is None:
            raise LaneNotFound('lane %d cannot be found' % lane_id)
        return lane.e_i_road_segment_id

    @staticmethod
    def get_lane_ordinal(lane_id: int) -> int:
        """
        get lane ordinal of the lane on the road (the rightest lane's ordinal is 0)
        :param lane_id:
        :return: lane's ordinal
        """
        return MapUtils.get_lane(lane_id).e_Cnt_right_adjacent_lane_count

    @staticmethod
    def get_lane_length(lane_id: int) -> float:
        """
        get the whole lane's length
        :param lane_id:
        :return: lane's length
        """
        nominal_points = MapUtils.get_lane_geometry(lane_id).a_nominal_path_points
        # TODO: lane length should be nominal_points[-1, NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]
        ds = np.mean(np.diff(nominal_points[:, NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]))
        return ds * (nominal_points.shape[0] - 1)

    @staticmethod
    @prof.ProfileFunction()
    def get_lane_frenet_frame(lane_id: int) -> FrenetSerret2DFrame:
        """
        get Frenet frame of the whole center-lane for the given lane
        :param lane_id:
        :return: Frenet frame
        """
        nominal_points = MapUtils.get_lane_geometry(lane_id).a_nominal_path_points

        points = nominal_points[:, (NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value,
                                    NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value)]

        yaw = nominal_points[:, NominalPathPoint.CeSYS_NominalPathPoint_e_phi_heading.value]
        T = np.c_[np.cos(yaw), np.sin(yaw)]
        N = NumpyUtils.row_wise_normal(T)
        k = nominal_points[:, NominalPathPoint.CeSYS_NominalPathPoint_e_il_curvature.value][:, np.newaxis]
        k_tag = nominal_points[:, NominalPathPoint.CeSYS_NominalPathPoint_e_il2_curvature_rate.value][:, np.newaxis]
        ds = np.mean(
            np.diff(nominal_points[:, NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]))  # TODO: is this necessary?

        return FrenetSerret2DFrame(points=points, T=T, N=N, k=k, k_tag=k_tag, ds=ds)

    @staticmethod
    def get_adjacent_lane_ids(lane_id: int, relative_lane: RelativeLane) -> List[int]:
        """
        get sorted adjacent (right/left) lanes relative to the given lane segment, or empty list if no adjacent lanes
        :param lane_id:
        :param relative_lane: either right or left
        :return: adjacent lanes ids sorted by their distance from the given lane;
                    if there are no such lanes, return empty list []
        """

        lane = MapUtils.get_lane(lane_id)
        if relative_lane == RelativeLane.RIGHT_LANE:
            adj_lanes = lane.as_right_adjacent_lanes
        elif relative_lane == RelativeLane.LEFT_LANE:
            adj_lanes = lane.as_left_adjacent_lanes
        else:
            raise ValueError('Relative lane must be either right or left: lane_id %d, relative_lane %s'
                             % (lane_id, relative_lane))
        return [adj_lane.e_i_lane_segment_id for adj_lane in adj_lanes]

    @staticmethod
    def get_closest_lane_ids(lane_id: int) -> Dict[RelativeLane, int]:
        """
        get dictionary that given lane_id maps from RelativeLane to lane_id of the immediate neighbor lane
        :param lane_id:
        :return: dictionary from RelativeLane to the immediate neighbor lane ids (or None if the neighbor does not exist)
        """
        right_lanes = MapUtils.get_adjacent_lane_ids(lane_id, RelativeLane.RIGHT_LANE)
        left_lanes = MapUtils.get_adjacent_lane_ids(lane_id, RelativeLane.LEFT_LANE)
        relative_lane_ids: Dict[RelativeLane, int] = {}
        if len(right_lanes) > 0:
            relative_lane_ids[RelativeLane.RIGHT_LANE] = right_lanes[0]
        relative_lane_ids[RelativeLane.SAME_LANE] = lane_id
        if len(left_lanes) > 0:
            relative_lane_ids[RelativeLane.LEFT_LANE] = left_lanes[0]
        return relative_lane_ids

    @staticmethod
    def _get_all_middle_lanes():
        """
        Returns the middle lane of each road segment.
        :return:
        """
        lanes_per_roads = [MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)
                           for road_segment_id in MapUtils.get_road_segment_ids()]
        return [lanes[int(len(lanes) / 2)] for lanes in lanes_per_roads]

    @staticmethod
    def get_closest_lane(cartesian_point: CartesianPoint2D) -> int:
        """
        Given cartesian coordinates, find the closest lane to the point. Note that this function operates only on the lane segments that are
        provided in the smaller, geometry-data horizon (i.e. the lane segments provided in the SceneStaticGeometry structure).
        :param cartesian_point: 2D cartesian coordinates
        :return: closest lane segment id
        """
        x_index = NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value
        y_index = NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value

        map_lane_ids = np.array([lane_segment.e_i_lane_segment_id
                                 for lane_segment in
                                 SceneStaticModel.get_instance().get_scene_static().s_Data.s_SceneStaticGeometry.as_scene_lane_segments])

        num_points_in_map_lanes = np.array([MapUtils.get_lane_geometry(lane_id).a_nominal_path_points.shape[0]
                                            for lane_id in map_lane_ids])

        num_points_in_longest_lane = np.max(num_points_in_map_lanes)
        # create 3D matrix of all lanes' points; pad it by inf according to the largest number of lane points
        map_lanes_xy_points = np.array([np.vstack((MapUtils.get_lane_geometry(lane_id).a_nominal_path_points[:, (x_index, y_index)],
                                        np.full((num_points_in_longest_lane - num_points_in_map_lanes[i], 2), np.inf)))
                                        for i, lane_id in enumerate(map_lane_ids)])
        distances_from_lane_points = np.linalg.norm(map_lanes_xy_points - cartesian_point, axis=2)  # 2D matrix
        closest_points_idx_per_lane = np.argmin(distances_from_lane_points, axis=1)
        # 1D array: the minimal distances to the point per lane
        min_dist_per_lane = distances_from_lane_points[np.arange(distances_from_lane_points.shape[0]),
                                                       closest_points_idx_per_lane]

        # find all lanes having the closest distance to the point
        # TODO: fix map in PG_split.bin such that seam points of connected lanes will overlap,so we can use smaller atol
        closest_lanes_idxs = np.where(np.isclose(min_dist_per_lane, min_dist_per_lane.min(), atol=0.1))[0]

        if closest_lanes_idxs.size == 1:  # a single closest lane
            return map_lane_ids[closest_lanes_idxs[0]]

        # Among the closest lanes, find lanes whose closest point is internal (not start/end point of the lane).
        # In this case (internal point) we are not expecting a numerical issue.
        # If such lanes exist, return an arbitrary one of them.
        lanes_with_internal_closest_point = np.where(np.logical_and(closest_points_idx_per_lane[closest_lanes_idxs] > 0,
                                                                    closest_points_idx_per_lane[closest_lanes_idxs] <
                                                                    num_points_in_map_lanes[closest_lanes_idxs] - 1))[0]
        if len(lanes_with_internal_closest_point) > 0:  # then return arbitrary (first) lane with internal closest point
            return map_lane_ids[closest_lanes_idxs[lanes_with_internal_closest_point[0]]]

        # The rest of the code handles deciding on which lane to project out of two closest lanes, while they share
        # a given mutual closest point.
        # If cartesian_point is near a seam between two (or more) lanes, choose the closest lane according to its
        # local yaw, such that the cartesian_point might be projected on the chosen lane.

        lane_idx = closest_lanes_idxs[0]  # choose arbitrary (first) closest lane
        lane_id = map_lane_ids[lane_idx]
        seam_point_idx = closest_points_idx_per_lane[lane_idx]
        # calculate a vector from the closest point to the input point
        vec_to_input_point = cartesian_point - MapUtils.get_lane_geometry(lane_id).a_nominal_path_points[
            seam_point_idx, (x_index, y_index)]
        yaw_to_input_point = np.arctan2(vec_to_input_point[1], vec_to_input_point[0])
        lane_local_yaw = MapUtils.get_lane_geometry(map_lane_ids[lane_idx]).a_nominal_path_points[
            seam_point_idx, NominalPathPoint.CeSYS_NominalPathPoint_e_phi_heading.value]
        if np.cos(yaw_to_input_point - lane_local_yaw) >= 0:  # local_yaw & yaw_to_input_point create an acute angle
            # take a lane that starts in the closest point
            final_lane_idx = closest_lanes_idxs[closest_points_idx_per_lane[closest_lanes_idxs] == 0][0]
        else:  # local_yaw & yaw_to_input_point create an obtuse angle ( > 90 degrees)
            # take a lane that ends in the closest point
            final_lane_idx = closest_lanes_idxs[closest_points_idx_per_lane[closest_lanes_idxs] > 0][0]
        return map_lane_ids[final_lane_idx]

    @staticmethod
    def get_dist_to_lane_borders(lane_id: int, s: float) -> (float, float):
        """
        get distance from the lane center to the lane borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right lane border, distance from the left lane border
        """
        nominal_points = MapUtils.get_lane_geometry(lane_id).a_nominal_path_points

        closest_s_idx = np.argmin(np.abs(nominal_points[:,
                                         NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] - s))
        return (nominal_points[closest_s_idx, NominalPathPoint.CeSYS_NominalPathPoint_e_l_left_offset.value],
                -nominal_points[closest_s_idx, NominalPathPoint.CeSYS_NominalPathPoint_e_l_right_offset.value])

    @staticmethod
    def get_lane_width(lane_id: int, s: float) -> float:
        """
        get lane width at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: lane width
        """
        border_right, border_left = MapUtils.get_dist_to_lane_borders(lane_id, s)
        return border_right + border_left

    @staticmethod
    def get_upstream_lane_ids(lane_id: int) -> List[LaneSegmentID]:
        """
        Get upstream lane ids (incoming) of the given lane.
        This is referring only to the previous road-segment, and the returned list is there for many-to-1 connection.
        :param lane_id:
        :return: list of upstream lanes ids
        """
        upstream_connectivity = MapUtils.get_lane(lane_id).as_upstream_lanes
        return [connectivity.e_i_lane_segment_id for connectivity in upstream_connectivity]

    @staticmethod
    def get_downstream_lane_ids(lane_id: int) -> List[LaneSegmentID]:
        """
        Get downstream lane ids (outgoing) of the given lane.
        This is referring only to the next road-segment, and the returned list is there for 1-to-many connection.
        :param lane_id:
        :return: list of downstream lanes ids
        """
        downstream_connectivity = MapUtils.get_lane(lane_id).as_downstream_lanes
        return [connectivity.e_i_lane_segment_id for connectivity in downstream_connectivity]

    @staticmethod
    def get_upstream_lane_maneuver_types(lane_id: int) -> Dict[LaneSegmentID, ManeuverType]:
        """
        Get maneuver types of the upstream lanes (incoming) of the given lane as a dictionary with the upstream lane ids as keys.
        This is referring only to the previous road segment.
        :param lane_id: ID for the lane in question
        :return: Maneuver types of the upstream lanes
        """
        upstream_connectivity = MapUtils.get_lane(lane_id).as_upstream_lanes
        return {connectivity.e_i_lane_segment_id: connectivity.e_e_maneuver_type for connectivity in upstream_connectivity}

    @staticmethod
    def get_downstream_lane_maneuver_types(lane_id: int) -> Dict[LaneSegmentID, ManeuverType]:
        """
        Get maneuver types of the downstream lanes (outgoing) of the given lane as a dictionary with the downstream lane ids as keys.
        This is referring only to the next road segment.
        :param lane_id: ID for the lane in question
        :return: Maneuver types of the downstream lanes
        """
        downstream_connectivity = MapUtils.get_lane(lane_id).as_downstream_lanes
        return {connectivity.e_i_lane_segment_id: connectivity.e_e_maneuver_type for connectivity in downstream_connectivity}

    @staticmethod
    def get_lanes_ids_from_road_segment_id(road_segment_id: int) -> List[int]:
        """
        Get sorted list of lanes for given road segment. The output lanes are ordered by the lanes' ordinal,
        i.e. from the rightest lane to the most left.
        :param road_segment_id:
        :return: sorted list of lane segments' IDs
        """
        return list(MapUtils.get_road_segment(road_segment_id).a_i_lane_segment_ids)

    @staticmethod
    @raises(LaneNotFound, RoadNotFound)
    @prof.ProfileFunction()
    def get_lookahead_frenet_frame_by_cost(lane_id: int, station: float, route_plan: RoutePlan, logger: Optional[Logger] = None,
                                           can_augment: Optional[Dict[RelativeLane, bool]] = None) -> \
                                           Dict[RelativeLane, GeneralizedFrenetSerretFrame]:
        """
        Create Generalized Frenet frame along lane center, starting from given lane and station.
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
        # Get the lane subsegments
        upstream_lane_subsegments = MapUtils._get_upstream_lane_subsegments(lane_id, station, MAX_BACKWARD_HORIZON, logger)

        if station < MAX_BACKWARD_HORIZON:
            # If the given station is not far enough along the lane, then the backward horizon will pass the beginning of the lane. In this
            # case, the starting station for the forward lookahead should be the beginning of the current lane, and the forward lookahead
            # distance should include the maximum forward horizon ahead of the given station and the backward distance to the beginning of
            # the lane (i.e. the station).
            starting_station = 0.0
            lookahead_distance = MAX_FORWARD_HORIZON + station
        else:
            # If the given station is far enough along the lane, then the backward horizon will not pass the beginning of the lane. In this
            # case, the starting station for the forward lookahead should be the end of the backward horizon, and the forward lookahead
            # distance should include the maximum forward and backward horizons ahead of and behind the given station, respectively. In
            # other words, if we're at station = 150 m on a lane and the maximum forward and backward horizons are 400 m and 100 m,
            # respectively, then starting station = 50 m and forward lookahead distance = 400 + 100 = 500 m. This is the case where the GFF
            # does not include any upstream lanes.
            starting_station = station - MAX_BACKWARD_HORIZON
            lookahead_distance = MAX_FORWARD_HORIZON + MAX_BACKWARD_HORIZON

        lane_subsegments_dict = MapUtils._advance_by_cost(initial_lane_id=lane_id,
                                                          initial_s=starting_station,
                                                          lookahead_distance=lookahead_distance,
                                                          route_plan=route_plan,
                                                          lane_subsegments=upstream_lane_subsegments,
                                                          can_augment=can_augment)

        gffs_dict = {}

        # Build GFFs from the Frenet Subsegments for the lane/augmented lanes that were created.
        for rel_lane in lane_subsegments_dict:
            lane_subsegments, is_partial, is_augmented = lane_subsegments_dict[rel_lane]

            if is_partial and is_augmented:
                gff_type = GFF_Type.AugmentedPartial
            elif is_partial:
                gff_type = GFF_Type.Partial
            elif is_augmented:
                gff_type = GFF_Type.Augmented
            else:
                gff_type = GFF_Type.Normal

            # Create Frenet frame for each sub segment
            frenet_frames = [MapUtils.get_lane_frenet_frame(lane_subsegment.e_i_SegmentID) for lane_subsegment in lane_subsegments]

            # Create GFF
            gffs_dict[rel_lane] = GeneralizedFrenetSerretFrame.build(frenet_frames, lane_subsegments, gff_type)

        return gffs_dict

    @staticmethod
    def _get_upstream_lane_subsegments(initial_lane_id: int, initial_station: float, backward_distance: float,
                                       logger: Optional[Logger] = None) -> List[FrenetSubSegment]:
        """
        Return a list of lane subsegments that are upstream to the given lane and extending as far back as backward_distance
        :param initial_lane_id: ID of lane to start from
        :param initial_station: Station on given lane
        :param backward_distance: Distance [m] to look backwards
        :param logger: Logger object to log warning messages
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
                if logger is not None:
                    logger.debug(UpstreamLaneNotFound("Upstream lane not found for lane_id=%d" % (lane_id)))

                break
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
    @raises(RoadNotFound, LaneNotFound, NavigationPlanTooShort, StraightConnectionNotFound)
    @prof.ProfileFunction()
    def _advance_by_cost(initial_lane_id: int, initial_s: float, lookahead_distance: float,
                         route_plan: RoutePlan,
                         can_augment: Optional[Dict[RelativeLane, bool]] = None,
                         lane_subsegments: Optional[List[FrenetSubSegment]] = None,
                         cumulative_distance: Optional[float] = None) -> \
                         Dict[RelativeLane, Tuple[List[FrenetSubSegment], bool, bool]]:
        """
        Given a longitudinal position <initial_s> on lane segment <initial_lane_id>, advance <lookahead_distance>
        further according to costs of each FrenetFrame, and finally return a configuration of lane-subsegments.
        If <desired_lon> is more than the distance to end of the plan, a LongitudeOutOfRoad exception is thrown.
        :param initial_lane_id: the initial lane_id (the vehicle is current on)
        :param initial_s: initial longitude along <initial_lane_id>
        :param lookahead_distance: the desired distance of lookahead in [m].
        :param route_plan: the relevant navigation plan to iterate over its road IDs.
        :param can_augment: dict of RelativeLane to bool; if True, try to create an augmented lane for that RelativeLane
        :param lane_subsegments: initial lane segments; further lane segments will be appended to this. Used for recursion.
        :param cumulative_distance: cumulative_distance of lane_subsegments; further distance will be added to this. Used for recursion.
        :return: Dictionary with potential keys: [RelativeLane.SAME_LANE, RelativeLane.LEFT_LANE, RelativeLane.RIGHT_LANE]
                 These keys represent the non-augmented, left-augmented, and right-augmented gffs that can be created.
                 The key-value pair for the non-augmented lane (i.e. RelativeLane.SAME_LANE) will always exist, and it refers
                 to the provided initial_lane_id. The left-augmented and right-augmented keys (i.e. RelativeLane.LEFT_LANE
                 and RelativeLane.RIGHT_LANE) will only exist when an augmented GFF can be created. The values are tuples
                 that contain a list of FrenetSubSegments that will be used to create the GFF and two flags that denote the
                 GFF type. The first flag denotes a partial GFF and the second flag denotes an augmented GFF.
        """
        initial_road_segment_id = MapUtils.get_road_segment_id_from_lane_id(initial_lane_id)

        try:
            current_road_idx_on_plan = np.where(route_plan.s_Data.a_i_road_segment_ids == initial_road_segment_id)[0][0]
        except IndexError:
            raise RoadNotFound("Road ID {} is not in the route plan road segment list".format(initial_road_segment_id))

        # Assign arguments that are default to None
        lane_subsegments = copy.deepcopy(lane_subsegments) or []
        can_augment = copy.deepcopy(can_augment) or {RelativeLane.LEFT_LANE: False, RelativeLane.RIGHT_LANE: False}
        cumulative_distance = cumulative_distance or 0.0

        # Set initial values
        current_lane_id = initial_lane_id
        current_segment_start_s = initial_s  # reference longitudinal position on the lane of current_lane_id
        lane_subsegments_dict = {}
        is_partial = False

        while True:
            current_lane_length = MapUtils.get_lane(current_lane_id).e_l_length  # a lane's s_max

            # distance to travel on current lane: distance to end of lane, or shorter if reached <lookahead distance>
            current_segment_end_s = min(current_lane_length,
                                        current_segment_start_s + lookahead_distance - cumulative_distance)

            # add subsegment to the list and add traveled distance to <cumulative_distance> sum
            lane_subsegments.append(FrenetSubSegment(current_lane_id, current_segment_start_s, current_segment_end_s))
            cumulative_distance += current_segment_end_s - current_segment_start_s

            if cumulative_distance > lookahead_distance - EPS:
                break

            next_road_idx_on_plan = current_road_idx_on_plan + 1

            if next_road_idx_on_plan >= route_plan.s_Data.e_Cnt_num_road_segments:
                raise NavigationPlanTooShort("Cannot progress further on plan %s (leftover: %s [m]); "
                                             "current_segment_end_s=%f lookahead_distance=%f" %
                                             (route_plan.s_Data.a_i_road_segment_ids, lookahead_distance - cumulative_distance,
                                              current_segment_end_s, lookahead_distance))

            valid_downstream_lanes = MapUtils._get_valid_downstream_lanes(current_lane_id, route_plan)
            num_valid_downstream_lanes = len(valid_downstream_lanes.keys())

            if num_valid_downstream_lanes == 0:
                is_partial = True
                break
            elif num_valid_downstream_lanes == 1:
                # Turn the return of values() into list and access the lane segment ID
                current_lane_id = list(valid_downstream_lanes.values())[0]
            else:
                # If there are multiple valid downstream lanes, choose the straight connection to continue on.
                # TODO: This will have to be revisited once more complex road geometries are encountered. At The moment, this should work
                #  for lane splits and forks.
                try:
                    current_lane_id = valid_downstream_lanes[ManeuverType.STRAIGHT_CONNECTION]
                except KeyError:
                    raise StraightConnectionNotFound("Straight downstream connection not found for lane=%d", current_lane_id)

                # Check if augmented lanes can be created
                if can_augment[RelativeLane.RIGHT_LANE] and ManeuverType.RIGHT_SPLIT in valid_downstream_lanes:
                    can_augment[RelativeLane.RIGHT_LANE] = False

                    right_augmented_lane_subsegments_dict = MapUtils._advance_by_cost(initial_lane_id=valid_downstream_lanes[ManeuverType.RIGHT_SPLIT],
                                                                                      initial_s=0.0,
                                                                                      lookahead_distance=lookahead_distance,
                                                                                      route_plan=route_plan,
                                                                                      lane_subsegments=lane_subsegments,
                                                                                      cumulative_distance=cumulative_distance)

                    # Check to make sure that dictionary is not empty
                    if right_augmented_lane_subsegments_dict:
                        # Get returned information. Note that the use of the RelativeLane.SAME_LANE key here is correct. Read the return value
                        # description above for more information.
                        right_augmented_lane_subsegments, is_right_partial, _ = right_augmented_lane_subsegments_dict[RelativeLane.SAME_LANE]

                        # Assign information to dictionary accordingly
                        lane_subsegments_dict[RelativeLane.RIGHT_LANE] = (right_augmented_lane_subsegments, is_right_partial, True)

                if can_augment[RelativeLane.LEFT_LANE] and ManeuverType.LEFT_SPLIT in valid_downstream_lanes:
                    can_augment[RelativeLane.LEFT_LANE] = False

                    left_augmented_lane_subsegments_dict = MapUtils._advance_by_cost(initial_lane_id=valid_downstream_lanes[ManeuverType.LEFT_SPLIT],
                                                                                     initial_s=0.0,
                                                                                     lookahead_distance=lookahead_distance,
                                                                                     route_plan=route_plan,
                                                                                     lane_subsegments=lane_subsegments,
                                                                                     cumulative_distance=cumulative_distance)

                    # Check to make sure that dictionary is not empty
                    if left_augmented_lane_subsegments_dict:
                        # Get returned information. Note that the use of the RelativeLane.SAME_LANE key here is correct. Read the return value
                        # description above for more information.
                        left_augmented_lane_subsegments, is_left_partial, _ = left_augmented_lane_subsegments_dict[RelativeLane.SAME_LANE]

                        # Assign information to dictionary accordingly
                        lane_subsegments_dict[RelativeLane.LEFT_LANE] = (left_augmented_lane_subsegments, is_left_partial, True)

            current_segment_start_s = 0.0
            current_road_idx_on_plan = next_road_idx_on_plan

        # Assign subsegments to dictionary for relative lane
        lane_subsegments_dict[RelativeLane.SAME_LANE] = (lane_subsegments, is_partial, False)

        return lane_subsegments_dict

    @staticmethod
    def _get_valid_downstream_lanes(current_lane_id: int, route_plan: RoutePlan) -> Dict[ManeuverType, LaneSegmentID]:
        """
        Finds the valid downstream lanes from the current_lane_id lane that are on the route_plan.
        Lanes with a saturated occupancy cost are not valid.
        Lanes with a saturated end cost are not valid when there are multiple downstream lanes.
        :param current_lane_id: Lane ID of current lane
        :param route_plan: Route plan that contains desired roads and lane costs
        :return: Dictionary mapping the maneuver type to the downstream lane ID
                 The dictionary is empty when there are no valid downstream lanes.
        """
        route_cost_dict = route_plan.to_costs_dict()

        # Get all downstream lanes in the route plan that do not have a saturated lane occupancy cost
        valid_downstream_lane_ids = [lane_id for lane_id in MapUtils.get_downstream_lane_ids(current_lane_id)
                                     if (lane_id in route_cost_dict
                                         and route_cost_dict[lane_id][LANE_OCCUPANCY_COST_IND] < SATURATED_COST)]

        # If there are multiple valid downstream lanes, then filter the lanes further by lane end cost.
        if len(valid_downstream_lane_ids) > 1:
            valid_downstream_lane_ids = [lane_id for lane_id in valid_downstream_lane_ids
                                         if route_cost_dict[lane_id][LANE_END_COST_IND] < SATURATED_COST]

        downstream_lane_maneuver_types = MapUtils.get_downstream_lane_maneuver_types(current_lane_id)

        return {downstream_lane_maneuver_types[downstream_lane_id]: downstream_lane_id
                for downstream_lane_id in valid_downstream_lane_ids}


    @staticmethod
    @raises(UpstreamLaneNotFound)
    def _get_upstream_lanes_from_distance(starting_lane_id: int, starting_lon: float, backward_dist: float) -> \
            (List[int], float):
        """
        given starting point (lane + starting_lon) on the lane and backward_dist, get list of lanes backward
        until reaching total distance from the starting point at least backward_dist
        :param starting_lane_id:
        :param starting_lon:
        :param backward_dist:
        :return: list of lanes backward and longitude on the last lane
        """
        path = [starting_lane_id]
        prev_lane_id = starting_lane_id
        total_dist = starting_lon
        while total_dist < backward_dist:
            prev_lane_ids = MapUtils.get_upstream_lane_ids(prev_lane_id)
            if len(prev_lane_ids) == 0:
                # TODO: the lane can actually have no upstream; should we continue with the existing path instead of
                #   raising exception, if total_dist > TBD
                raise UpstreamLaneNotFound("_get_upstream_lanes_from_distance: Upstream lane not "
                                           "found for lane_id=%d" % prev_lane_id)
            # TODO: how to choose between multiple upstreams if all of them belong to route plan road segment
            prev_lane_id = prev_lane_ids[0]
            path.append(prev_lane_id)
            total_dist += MapUtils.get_lane_length(prev_lane_id)
        return path, total_dist - backward_dist

    @staticmethod
    @raises(LaneNotFound)
    def get_lane(lane_id: int) -> SceneLaneSegmentBase:
        """
        Retrieves lane by lane_id  according to the last message
        :param lane_id:
        :return:
        """
        scene_static = SceneStaticModel.get_instance().get_scene_static()
        lanes = [lane for lane in scene_static.s_Data.s_SceneStaticBase.as_scene_lane_segments if
                 lane.e_i_lane_segment_id == lane_id]
        if len(lanes) == 0:
            raise LaneNotFound('lane %d not found' % lane_id)
        if len(lanes) > 1:
            raise IDAppearsMoreThanOnce('lane %d appears more than once' % lane_id)
        return lanes[0]

    @staticmethod
    @raises(LaneNotFound)
    def get_lane_geometry(lane_id: int) -> SceneLaneSegmentGeometry:
        """
        Retrieves lane geometry (nom path points/boundary points) by lane_id  according to the last message
        :param lane_id:
        :return:
        """
        scene_static_lane_geo = SceneStaticModel.get_instance().get_scene_static()
        lanes = [lane for lane in scene_static_lane_geo.s_Data.s_SceneStaticGeometry.as_scene_lane_segments if
                 lane.e_i_lane_segment_id == lane_id]
        if len(lanes) == 0:
            raise LaneNotFound('lane %d not found in lane geometry' % lane_id)
        if len(lanes) > 1:
            raise IDAppearsMoreThanOnce('lane %d appears more than once in lane geometry' % lane_id)
        return lanes[0]

    @staticmethod
    @raises(RoadNotFound)
    def get_road_segment(road_id: int) -> SceneRoadSegment:
        """
        Retrieves road by road_id  according to the last message
        :param road_id:
        :return:
        """
        scene_static = SceneStaticModel.get_instance().get_scene_static()
        road_segments = [road_segment for road_segment in scene_static.s_Data.s_SceneStaticBase.as_scene_road_segment if
                         road_segment.e_i_road_segment_id == road_id]
        if len(road_segments) == 0:
            raise RoadNotFound('road %d not found' % road_id)
        if len(road_segments) > 1:
            raise IDAppearsMoreThanOnce('road %d appears more than once' % road_id)
        return road_segments[0]

    @staticmethod
    def get_stop_bar_and_stop_sign(lane_frenet: GeneralizedFrenetSerretFrame) -> []:
        """
        Returns a list of the locations (s coordinates) of stop signs and stop bars on the GFF, with their type
        The list is ordered from closest traffic flow control to farthest.
        :param lane_frenet: The GFF on which to retrieve the static flow controls.
        :return: A list of distances to stop signs and stop bars on the the GFF, ordered from closest traffic flow
        control to farthest, along with the type of the control.
        """
        road_signs = MapUtils.get_static_traffic_flow_controls_s(lane_frenet)
        stop_bars_and_signs = []
        for road_sign in road_signs:
            # TODO verify these are the correct stop bar enums
            if road_sign[SIGN_TYPE] in [RoadObjectType.StopSign, RoadObjectType.StopBar_Left, RoadObjectType.StopBar_Right]:
                stop_bars_and_signs.append(road_sign)
        return stop_bars_and_signs

    @staticmethod
    def get_static_traffic_flow_controls_s(lane_frenet: GeneralizedFrenetSerretFrame) -> []:
        """
        Returns a list of the locations (s coordinates) of Static_Traffic_flow_controls on the GFF, with their type
        The list is ordered from closest traffic flow control to farthest.
        :param lane_frenet: The GFF on which to retrieve the static flow controls.
        :return: A list of distances to static flow controls on the the GFF, ordered from closest traffic flow control
        to farthest, along with the type of the control.
        """
        lane_ids = []
        # s coordinates
        road_signs_s_on_lane_segments = []
        road_sign_types = []
        for lane_id in lane_frenet.segment_ids:
            lane_segment = MapUtils.get_lane(lane_id)
            for static_traffic_flow_control in lane_segment.as_static_traffic_flow_control:
                lane_ids.append(lane_id)
                road_sign_types.append(static_traffic_flow_control.e_e_road_object_type)
                road_signs_s_on_lane_segments.append(static_traffic_flow_control.e_l_station)
        frenet_states = np.zeros((len(road_signs_s_on_lane_segments), 6))
        frenet_states[:, FS_SX] = np.asarray(road_signs_s_on_lane_segments)
        road_sign_s_on_gff = lane_frenet.convert_from_segment_states(frenet_states, np.asarray(lane_ids))[:, FS_SX]
        road_sign_info_on_gff = list(zip(road_sign_types, road_sign_s_on_gff))  # order of elements in zip must match types.py SIGN_TYPE, SIGN_DISTANCE
        road_sign_info_on_gff.sort(key=lambda x: x[SIGN_S])  # sort by distance after the conversion to real distance
        return road_sign_info_on_gff

    @staticmethod
    @raises(EquivalentStationNotFound)
    def _find_equivalent_station(lane_segment_id: int, gff_station: float, gff: GeneralizedFrenetSerretFrame) -> float:
        """
        Find the station on a lane that is equivalent to a given station in a GFF
        :param lane_segment_id: ID for the lane segment in question
        :param gff_station: Station in given GFF
        :param gff: Generalized Frenet frame
        :return: Equivalent station in given lane to given GFF station
        """
        previous_station_difference = FLOAT_MAX

        nominal_path_points = MapUtils.get_lane_geometry(lane_segment_id).a_nominal_path_points
        last_nominal_path_point_index = len(nominal_path_points) - 1

        for i, nominal_path_point in enumerate(nominal_path_points):
            cartesian_point = np.array([nominal_path_point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value],
                                        nominal_path_point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value]])

            try:
                station = gff.cpoint_to_fpoint(cartesian_point)[FP_SX]
            except (OutOfSegmentBack, OutOfSegmentFront, AssertionError):
                # If any of these errors were raised, then the nominal path point is not close to the desired station.
                continue

            station_difference = abs(gff_station - station)

            if station_difference < previous_station_difference:
                # If station_difference is less than previous_station_difference, there is potentially a closer station to the desired
                # station. Unless the last nominal path point has been reached, continue iterating until station_difference begins to
                # increase. If the last nominal path point has been reached and it's "close enough" to the desired station, return its
                # station.
                if i == last_nominal_path_point_index and station_difference < MAX_STATION_DIFFERENCE:
                    return nominal_path_point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]
                else:
                    previous_station_difference = station_difference
            else:
                # If station_difference is greater than or equal to previous_station_difference, then the closest station in the given lane
                # has been reached. Return the station of the previous nominal path point.
                return nominal_path_points[i - 1][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]

        # If this point is reached, then an equivalent station could not be found
        raise EquivalentStationNotFound("Could not determine the host's equivalent station in lane segment {}".format(lane_segment_id))
