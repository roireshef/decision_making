from logging import Logger
from typing import List, Dict, Tuple, Optional

import numpy as np
import rte.python.profiler as prof
from decision_making.src.exceptions import raises, RoadNotFound, \
    UpstreamLaneNotFound, LaneNotFound, IDAppearsMoreThanOnce
from decision_making.src.global_constants import EPS, LANE_END_COST_IND, LANE_OCCUPANCY_COST_IND, SATURATED_COST
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.scene_static_enums import ManeuverType, TrafficLightState, \
    StaticTrafficControlDeviceType, DynamicTrafficControlDeviceType
from decision_making.src.messages.scene_static_enums import NominalPathPoint
from decision_making.src.messages.scene_static_message import SceneLaneSegmentGeometry, \
    SceneLaneSegmentBase, SceneRoadSegment, TrafficControlBar, StaticTrafficControlDevice, DynamicTrafficControlDevice
from decision_making.src.messages.scene_tcd_message import DynamicTrafficControlDeviceStatus
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RoadSignRestriction
from decision_making.src.planning.types import CartesianPoint2D, FS_SX
from decision_making.src.planning.types import LaneSegmentID
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, \
    FrenetSubSegment
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.scene.scene_traffic_control_devices_status_model import SceneTrafficControlDevicesStatusModel


class MapUtils:
    @staticmethod
    def get_road_segment_ids() -> List[int]:
        """
        :return:road_segment_ids of every road in the static scene
        """
        scene_static = SceneStaticModel.get_instance().get_scene_static()
        road_segments = scene_static.s_Data.s_SceneStaticBase.as_scene_road_segment[
                        :scene_static.s_Data.s_SceneStaticBase.e_Cnt_num_road_segments]
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
        map_lanes_xy_points = np.array(
            [np.vstack((MapUtils.get_lane_geometry(lane_id).a_nominal_path_points[:, (x_index, y_index)],
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
        right_borders, left_borders =  MapUtils.get_dists_to_lane_borders(lane_id, np.array([s]))
        return right_borders[0], left_borders[0]

    @staticmethod
    def get_dists_to_lane_borders(lane_id: int, ss: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        get distances from the lane center to the lane borders at given longitudes from the lane's origin
        :param lane_id: id of lane of interest
        :param ss: longitudes of the lane center points (w.r.t. the lane Frenet frame) (must be 1D array)
        :return: distances from the right lane border, distances from the left lane border
                 (both 1D arrays, length corresponds to number of points in ss)
        """
        nominal_points = MapUtils.get_lane_geometry(lane_id).a_nominal_path_points

        # if ss was array of size 1, it is squeezed to zero-dimensional array by the calling function
        if ss.ndim == 0:
            ss = np.array([ss])
        closest_s_idxs = np.array([np.argmin(np.abs(nominal_points[:,
                                         NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] - s)) for s in ss])

        return (nominal_points[closest_s_idxs, NominalPathPoint.CeSYS_NominalPathPoint_e_l_left_offset.value],
               - nominal_points[closest_s_idxs, NominalPathPoint.CeSYS_NominalPathPoint_e_l_right_offset.value])



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
        return {connectivity.e_i_lane_segment_id: connectivity.e_e_maneuver_type for connectivity in
                upstream_connectivity}

    @staticmethod
    def get_downstream_lane_maneuver_types(lane_id: int) -> Dict[LaneSegmentID, ManeuverType]:
        """
        Get maneuver types of the downstream lanes (outgoing) of the given lane as a dictionary with the downstream lane ids as keys.
        This is referring only to the next road segment.
        :param lane_id: ID for the lane in question
        :return: Maneuver types of the downstream lanes
        """
        downstream_connectivity = MapUtils.get_lane(lane_id).as_downstream_lanes
        return {connectivity.e_i_lane_segment_id: connectivity.e_e_maneuver_type for connectivity in
                downstream_connectivity}

    @staticmethod
    def get_lane_maneuver_type(lane_id: int) -> ManeuverType:
        """
        Get maneuver type for the given lane id
        :param lane_id: ID for the lane in question
        :return: Maneuver type of the lane
        """
        try:
            upstream_lane = MapUtils.get_upstream_lane_ids(lane_id)[0]
            downstream_connectivity = MapUtils.get_lane(upstream_lane).as_downstream_lanes
        # returns unknown type if there is no upstream lane for current lane or the upstream lane can not be found in scene static data
        except (IndexError, LaneNotFound):
            return ManeuverType.UNKNOWN
        connectivity = [connectivity.e_e_maneuver_type for connectivity in downstream_connectivity if
                        connectivity.e_i_lane_segment_id == lane_id]
        assert len(connectivity) == 1, f"connectivity from upstream lane {upstream_lane} to lane {lane_id} was supposed to be 1-to-1" + \
                                       f"connection but got {len(connectivity)} instances"
        return connectivity[0]


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
    @raises(RoadNotFound, LaneNotFound)
    def _advance_on_plan(initial_lane_id: int, initial_s: float, lookahead_distance: float, route_plan: RoutePlan, logger: Logger) \
            -> Tuple[List[FrenetSubSegment], float]:
        """
        Advances downstream according to plan as long as there is a single valid (according to navigation plan)
        downstream lane only - this can be in case there is a single downstream lane, or in a case of a split where only
        one of the downstream lanes are in the navigation plan (the others are not)
        :param initial_lane_id: the initial lane_id (the vehicle is current on)
        :param initial_s: initial longitude along <initial_lane_id>
        :param lookahead_distance: the desired distance of lookahead in [m].
        :param route_plan: the relevant navigation plan to iterate over its road IDs.
        :param logger: Logger object to log warning messages
        :return: Tuple(List of FrenetSubSegment traversed downstream, accumulated traveled distance on that sequence)
        """

        initial_road_segment_id = MapUtils.get_road_segment_id_from_lane_id(initial_lane_id)

        try:
            current_road_idx_on_plan = np.where(route_plan.s_Data.a_i_road_segment_ids == initial_road_segment_id)[0][0]
        except IndexError:
            raise RoadNotFound("Road ID {} is not in the route plan road segment list".format(initial_road_segment_id))

        # Assign arguments that are default to None
        lane_subsegments = []
        cumulative_distance = 0.0

        # Set initial values
        current_lane_id = initial_lane_id
        current_segment_start_s = initial_s  # reference longitudinal position on the lane of current_lane_id

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
                logger.debug(f"NavigationPlanTooShort: Cannot progress further on plan {route_plan.s_Data.a_i_road_segment_ids}"
                             f" (leftover: {lookahead_distance - cumulative_distance} [m]); current_segment_end_s={current_segment_end_s}"
                             f" lookahead_distance={lookahead_distance}")

            valid_downstream_lanes = MapUtils._get_valid_downstream_lanes(current_lane_id, route_plan)
            num_valid_downstream_lanes = len(valid_downstream_lanes.keys())

            if num_valid_downstream_lanes == 0:     # no valid downstream lanes on the navigation plan (dead-end)
                break
            elif num_valid_downstream_lanes > 1:    # more than 1 downstream lane on the navigation plan
                break

            # If the downstream lane exist and is single, use it
            current_lane_id = list(valid_downstream_lanes.values())[0]

            current_segment_start_s = 0.0
            current_road_idx_on_plan = next_road_idx_on_plan

        return lane_subsegments, cumulative_distance


    @staticmethod
    @raises(LaneNotFound)
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

        valid_downstream_lane_ids = MapUtils._fix_closed_splits(
            current_lane_id=current_lane_id,
            valid_downstream_lane_ids=valid_downstream_lane_ids,
            downstream_lane_maneuver_types=downstream_lane_maneuver_types
        )

        return {downstream_lane_maneuver_types[downstream_lane_id]: downstream_lane_id
                for downstream_lane_id in valid_downstream_lane_ids}

    @staticmethod
    def _fix_closed_splits(current_lane_id: int,
                           valid_downstream_lane_ids: List[int],
                           downstream_lane_maneuver_types: Dict[LaneSegmentID, ManeuverType]) -> List[int]:
        # Handle closed splits
        straight_connections = [downstream_lane_id
                                for downstream_lane_id in valid_downstream_lane_ids
                                if downstream_lane_maneuver_types[downstream_lane_id] == ManeuverType.STRAIGHT_CONNECTION]

        if len(straight_connections) == 0:
            return valid_downstream_lane_ids

        chosen_straight_connection = straight_connections[0]
        if len(straight_connections) > 1:
            # In this case we have a closed split. Until there is a specific marking for it on the map, choose the one
            # that corresponds to driving straight
            current_right_adjacent_lanes = MapUtils.get_adjacent_lane_ids(current_lane_id, RelativeLane.RIGHT_LANE)
            current_left_adjacent_lanes = MapUtils.get_adjacent_lane_ids(current_lane_id, RelativeLane.LEFT_LANE)

            # Get a sorted list of the downstream straight connections, from left to right.
            sorted_straight_connections = sorted(
                straight_connections,
                key=lambda lane_id: len(MapUtils.get_adjacent_lane_ids(lane_id, RelativeLane.LEFT_LANE))
            )

            if len(current_right_adjacent_lanes) == 0 and len(current_left_adjacent_lanes) > 0:
                # We're driving on the rightmost lane currently, we choose the leftmost straight connection (we assume
                # the split is to the right
                chosen_straight_connection = sorted_straight_connections[0]
            elif len(current_right_adjacent_lanes) > 0 and len(current_left_adjacent_lanes) == 0:
                # We're driving on the leftmost lane currently, we choose the rightmost straight connection (we assume
                # the split is to the right
                chosen_straight_connection = sorted_straight_connections[-1]
            else:
                # We're driving on a single lane road, in this case, if there is a split to an odd number of lanes, we
                # pick the center one, if there is an even number of lane we pick the left most as we assume the splits
                # occur to the right
                if len(sorted_straight_connections) % 2 == 1:
                    chosen_straight_connection = sorted_straight_connections[len(sorted_straight_connections) // 2]
                else:
                    chosen_straight_connection = sorted_straight_connections[0]

        # Remove all straight connnections and then re-add only the chosen straight connection
        for lane_id in straight_connections:
            valid_downstream_lane_ids.remove(lane_id)

        valid_downstream_lane_ids.append(chosen_straight_connection)

        return valid_downstream_lane_ids

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
    def get_closest_stop_bar(target_lane_frenet: GeneralizedFrenetSerretFrame, ego_location: float,
                             offset_to_ego: float, stop_bar_id_to_ignore: int = None, logger: Logger = None) -> \
            Tuple[Optional[Tuple[TrafficControlBar, float]], float]:
        """
        Returns the closest stop bar and its distance.
        No existence checks necessary, as it was already tested by FilterActionsTowardsCellsWithoutRoadSigns
        :param target_lane_frenet:
        :param ego_location: on GFF
        :param offset_to_ego the offset relative to the ego location from which to start looking for a stop bar
        :param stop_bar_id_to_ignore: id of stop sign/bar to ignore (used by driver initiated motion)
        :param logger:
        :return: tuple of (closest stop bar, distance to closest stop bar, distance to ignored) or None if not found
        and distance to ignored stop bar (or None)
        """
        # TODO Possibly apply the DIM_MARGIN_TO_STOP_BAR only if there is no other stop bar close in front,
        #  to handle case of 2 close stop bars say DIM_MARGIN_TO_STOP_BAR-1 apart
        stop_bars_and_distances = MapUtils.get_traffic_control_bars_s(target_lane_frenet, ego_location - offset_to_ego)
        static_tcds, dynamic_tcds = MapUtils.get_traffic_control_devices()

        ignored_distance = None
        # check for active stop bar from the closest to the farthest (ignored should be first if it exists)
        for stop_bar, distance in stop_bars_and_distances:
            if stop_bar.e_i_traffic_control_bar_id != stop_bar_id_to_ignore:
                # Only considers TCB is in front of (ego_location - DIM_MARGIN_TO_STOP_BAR)
                active_static_tcds, active_dynamic_tcds = MapUtils.get_TCDs_for_bar(stop_bar, static_tcds, dynamic_tcds)
                road_signs_restriction = MapUtils.resolve_restriction_of_road_sign(active_static_tcds, active_dynamic_tcds,
                                                                                   logger)
                should_stop = MapUtils.should_stop_at_stop_bar(road_signs_restriction)
                if logger is not None:
                    logger.debug("Stop bar check id %d at distance %f active S-TCD %s active D-TCD %s, ego at %f, stop? %s",
                                 stop_bar.e_i_traffic_control_bar_id, distance,
                                 [(active_static_tcd.object_id, active_static_tcd.e_e_traffic_control_device_type)
                                  for active_static_tcd in active_static_tcds],
                                 [(active_dynamic_tcd[0].object_id, active_dynamic_tcd[0].e_e_traffic_control_device_type,
                                   MapUtils._get_confident_status(active_dynamic_tcd[1]))
                                  for active_dynamic_tcd in active_dynamic_tcds],
                                 ego_location, should_stop)
                if should_stop:
                    return (stop_bar, distance), ignored_distance
            else:
                ignored_distance = distance
                logger.debug("Stop bar ignored id %d distance %f", stop_bar_id_to_ignore, ignored_distance)
        return None, ignored_distance

    @staticmethod
    def get_traffic_control_bars_s(lane_frenet: GeneralizedFrenetSerretFrame, start_offset: float) -> \
            List[Tuple[TrafficControlBar, float]]:
        """
        Returns a list of the TrafficControlBars and their locations (s coordinates) on the GFF
        The list is ordered from closest traffic flow control to farthest.
        :param lane_frenet: The GFF on which to retrieve the TCBs.
        :param start_offset: An offset relative to GFF start from which TCBs are returned
        :return: List of TrafficControlBars and their distances on the the GFF, ordered from closest to farthest.
        """
        offset = max(start_offset, 0)
        lane_ids = []
        tcbs = []
        # go over the lanes and aggregate the TCBs
        for lane_id in lane_frenet.segment_ids:
            lane_segment = MapUtils.get_lane(lane_id)
            lane_ids += len(lane_segment.as_traffic_control_bar) * [lane_id]  # add same value multiple times
            tcbs.extend(lane_segment.as_traffic_control_bar)
        # calculate the distance on the GFF
        frenet_states = np.zeros((len(tcbs), 6))
        frenet_states[:, FS_SX] = np.asarray([tcb.e_l_station for tcb in tcbs])
        tcb_s_on_gff = lane_frenet.convert_from_segment_states(frenet_states, np.asarray(lane_ids))[:, FS_SX]
        # ensures we only return control bars that are within the GFF
        tcb_info_on_gff = [(tcb, s) for tcb, s in zip(tcbs, tcb_s_on_gff) if offset < s < lane_frenet.s_max]
        # sort by distance after the conversion to real distance
        tcb_info_on_gff.sort(key=lambda x: x[1])

        return tcb_info_on_gff

    @staticmethod
    def get_traffic_control_devices() -> \
            (Dict[int, StaticTrafficControlDevice],
             Dict[int, Tuple[DynamicTrafficControlDevice, DynamicTrafficControlDeviceStatus]]):
        """
        Returns a Dictionary of id:StaticTrafficControlDevice
        and id:(DynamicTrafficControlDevice, DynamicTrafficControlDeviceStatus)
        :return: Dict of static / dynamic TCDs on the the GFF. TCD id -> TCD.
        """
        scene_static = SceneStaticModel.get_instance().get_scene_static()
        tcds_status = SceneTrafficControlDevicesStatusModel.get_instance().get_traffic_control_devices_status()

        # STATIC signs
        static_tcds = {static_tcd.object_id: static_tcd
                       for static_tcd in scene_static.s_Data.s_SceneStaticBase.as_static_traffic_control_device}

        # DYNAMIC signs
        dynamic_tcds_and_status = {dynamic_tcd.object_id: (dynamic_tcd, tcds_status[dynamic_tcd.object_id]
                                   if dynamic_tcd.object_id in tcds_status else None) for dynamic_tcd in
                                   scene_static.s_Data.s_SceneStaticBase.as_dynamic_traffic_control_device}

        return static_tcds, dynamic_tcds_and_status

    @staticmethod
    def get_TCDs_for_bar(stop_bar: TrafficControlBar, static_tcds: Dict[int, StaticTrafficControlDevice],
                         dynamic_tcds: Dict[int, Tuple[DynamicTrafficControlDevice, DynamicTrafficControlDeviceStatus]]) -> \
            (List[StaticTrafficControlDevice], List[Tuple[DynamicTrafficControlDevice, DynamicTrafficControlDeviceStatus]]):
        """
        Find the TCDs that govern the desired behavior at the stop bar.
        :param stop_bar: for which the TCDs are retrieved.
        :param static_tcds: Dict of static TCDs on the GFF
        :param dynamic_tcds: Dict of dynamic TCDs on the GFF
        :return: Lists of static / dynamic TCDs for the given TCB
        """
        static_tcds_in_effect = [static_tcds[tcd_id] for tcd_id in stop_bar.e_i_static_traffic_control_device_id]
        dynamic_tcds_in_effect = [dynamic_tcds[tcd_id] for tcd_id in stop_bar.e_i_dynamic_traffic_control_device_id]
        return static_tcds_in_effect, dynamic_tcds_in_effect

    @staticmethod
    def resolve_restriction_of_road_sign(active_static_tcds: List[StaticTrafficControlDevice], active_dynamic_tcds:
                                         List[Tuple[DynamicTrafficControlDevice, DynamicTrafficControlDeviceStatus]],
                                         logger: Logger = None) \
            -> RoadSignRestriction:
        """
        Find the proper restriction given all the relevant TCDs and their statuses
        :param active_static_tcds: List of active static TCDs to consider 
        :param active_dynamic_tcds: List of (active dynamic TCDs, their status) to consider
        :param logger: logger
        :return: A single restriction imposed by the TCDs
        """
        # find the restriction imposed by each TCD. Keep it in a list of TCD type to its restriction
        static_restrictions = []
        for active_static_tcd in active_static_tcds:
            if active_static_tcd.e_e_traffic_control_device_type in [StaticTrafficControlDeviceType.YIELD,
                                                                     StaticTrafficControlDeviceType.STOP,
                                                                     StaticTrafficControlDeviceType.CROSSWALK,
                                                                     StaticTrafficControlDeviceType.MOVABLE_BARRIER]:
                static_restrictions.append((active_static_tcd.e_e_traffic_control_device_type, RoadSignRestriction.STOP))
            else:
                logger.warning("Static TCD for id " + active_static_tcd.object_id + ", type " +
                               active_static_tcd.e_e_traffic_control_device_type + " not expected. Ignoring")
        dynamic_restrictions = []
        for active_dynamic_tcd, dynamic_tcd_status in active_dynamic_tcds:
            status = MapUtils._get_confident_status(dynamic_tcd_status)
            if (status == TrafficLightState.UNKNOWN) and logger is not None:
                logger.warning("Dynamic TCD status for id " + active_dynamic_tcd.object_id + ", type " +
                               active_dynamic_tcd.e_e_traffic_control_device_type + " is UNKNOWN. Will restrict to STOP")
            if active_dynamic_tcd.e_e_traffic_control_device_type == DynamicTrafficControlDeviceType.TRAFFIC_LIGHT:
                dynamic_restrictions.append((active_dynamic_tcd.e_e_traffic_control_device_type,
                                             RoadSignRestriction.STOP
                                             if status != TrafficLightState.GREEN and status != TrafficLightState.GREEN_OLD
                                             else RoadSignRestriction.NONE))
            elif active_dynamic_tcd.e_e_traffic_control_device_type == DynamicTrafficControlDeviceType.RAILROAD_CROSSING:
                dynamic_restrictions.append((active_dynamic_tcd.e_e_traffic_control_device_type,
                                             RoadSignRestriction.STOP))
            elif active_dynamic_tcd.e_e_traffic_control_device_type == DynamicTrafficControlDeviceType.SCHOOL_ZONE:
                # TODO: Need to understand how to get the information on days + hours at which this is in effect.
                #   Currently, always stop in this situation
                dynamic_restrictions.append((active_dynamic_tcd.e_e_traffic_control_device_type,
                                            RoadSignRestriction.STOP))

        # combine restrictions
        return MapUtils.combine_restrictions(static_restrictions, dynamic_restrictions)

    @staticmethod
    def combine_restrictions(static_restrictions: List[Tuple[StaticTrafficControlDeviceType, RoadSignRestriction]],
                             dynamic_restrictions: List[Tuple[DynamicTrafficControlDeviceType, RoadSignRestriction]]) \
            -> RoadSignRestriction:
        """
        Combine the restrictions of all the TCDs into a single restriction.
        If there are dynamic TCDs, they currently override static TCDs
        Select the most restricting dynamic restriction. This allows a TRAFFIC LIGHT to override a STOP SIGN
        Otherwise select the most restricting static restriction.
        Assumes restriction with higher ordinal number are more restrictive
        # TODO break cases below into finer cases when relevant,
        #  e.g when there is a traffic light with flashing yellow + stop sign, we should stop
        :param static_restrictions: List of tuples(StaticTrafficControlDeviceType, RoadSignRestriction) for static TCD.
        :param dynamic_restrictions: List of tuples(DynamicTrafficControlDeviceType, RoadSignRestriction) for dynamic TCD.
        :return: combined restriction
        """
        # if there are dynamic TCDs, they currently override static TCDs
        # select the most restricting dynamic restriction. This allows a TRAFFIC LIGHT to override a STOP SIGN
        if len(dynamic_restrictions) > 0:
            return max([restriction for tcd_type, restriction in dynamic_restrictions])
        # otherwise select the most restricting static restriction.
        elif len(static_restrictions):
            return max([restriction for tcd_type, restriction in static_restrictions])
        else:
            return RoadSignRestriction.NONE

    @staticmethod
    def should_stop_at_stop_bar(restriction: RoadSignRestriction) -> bool:
        """
        Decide if need to stop at stop bar according to the restriction.
        Currently, decides to stop unless the restriction is RoadSignRestriction.NONE
        :param restriction: to evaluate
        :return: whether it is required to stop at stop bar
        """
        return restriction != RoadSignRestriction.NONE

    @staticmethod
    def _get_confident_status(status: DynamicTrafficControlDeviceStatus) -> TrafficLightState:
        """
        Given a list of statuses and their confidence for a dynamic TCD, decide which status to use.
        Currently uses the status with the highest probability
        :param status: of a dynamic TCD (2 lists of possible status and associated confidence)
        :return: A single status to be used as a probability 1 status
        """
        # TODO handle case where argmax is not good enough, and need to choose the worst case that is confident enough
        if status is None or len(status.a_e_status) == 0:
            return TrafficLightState.UNKNOWN
        else:
            confidence = np.array(status.a_Pct_status_confidence)
            status = np.array(status.a_e_status)
            highest_confidence_idx = np.argmax(confidence)
            return status[highest_confidence_idx]

    @staticmethod
    def get_merge_lane_id(initial_lane_id: int, initial_s: float, lookahead_distance: float, route_plan: RoutePlan,
                          logger: Logger) -> Optional[int]:
        """
        Given GFF for the current lane, find the closest merge connection into main road.
        Red line is s coordinate, from which host starts to interference laterally with the main road actors.
        We assume that there is a host's road segment starting from the red line and ending at the merge point.
        If initial_lane_id == segment.e_i_SegmentID, then we already crossed the red line.
        If there is a stop bar/sign before the red line, then return None.
        :param initial_lane_id: current lane id of ego
        :param initial_s: s of ego on initial_lane_id
        :param lookahead_distance: maximal lookahead for the lane merge from ego location
        :param route_plan: the relevant navigation plan to iterate over its road IDs
        :param logger:
        :return: lane_id of the segment between the red line and the merge point itself
        """

        lane_subsegments, _ = MapUtils._advance_on_plan(initial_lane_id, initial_s, lookahead_distance, route_plan, logger)

        # Find the merge point ahead
        cumulative_length = 0
        for segment in lane_subsegments:
            # TODO should this if be after the check for merge in current segment?
            cumulative_length += segment.e_i_SEnd - segment.e_i_SStart
            if cumulative_length > lookahead_distance:
                break
            current_lane_segment = MapUtils.get_lane(segment.e_i_SegmentID)

            # if there is a stop bar/sign before the red line, then return None
            # TODO is this a good restriction ???
            stop_bars = current_lane_segment.as_traffic_control_bar
            if len(stop_bars) > 0:
                break

            downstream_connectivity = current_lane_segment.as_downstream_lanes

            # Red line is s coordinate, from which host starts to interference laterally with the main road actors.
            # We assume that there is a host's road segment starting from the red line and ending at the merge point.
            # If initial_lane_id == segment.e_i_SegmentID, then we already crossed the red line.
            lane_merge_ahead = len(downstream_connectivity) == 1 and \
                               (downstream_connectivity[0].e_e_maneuver_type == ManeuverType.LEFT_MERGE_CONNECTION or
                                downstream_connectivity[0].e_e_maneuver_type == ManeuverType.RIGHT_MERGE_CONNECTION)
            # if segment.e_i_SegmentID == initial_lane_id then host already passed the red line and the merge completed
            if lane_merge_ahead and segment.e_i_SegmentID != initial_lane_id:
                    return segment.e_i_SegmentID

        # no merge connection was found
        return None
