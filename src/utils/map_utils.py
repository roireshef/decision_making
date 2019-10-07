from logging import Logger
from typing import List, Dict, Tuple

import numpy as np
import rte.python.profiler as prof
from decision_making.src.exceptions import raises, RoadNotFound, NavigationPlanTooShort, \
    UpstreamLaneNotFound, LaneNotFound, IDAppearsMoreThanOnce
from decision_making.src.global_constants import EPS, LANE_END_COST_IND, LANE_OCCUPANCY_COST_IND, SATURATED_COST
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.scene_static_enums import ManeuverType
from decision_making.src.messages.scene_static_enums import NominalPathPoint, RoadObjectType
from decision_making.src.messages.scene_static_message import SceneLaneSegmentGeometry, \
    SceneLaneSegmentBase, SceneRoadSegment
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import CartesianPoint2D, FS_SX, RoadSignInfo
from decision_making.src.planning.types import LaneSegmentID
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, \
    FrenetSubSegment
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.scene.scene_static_model import SceneStaticModel


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
    @raises(RoadNotFound, NavigationPlanTooShort, LaneNotFound)
    def _advance_on_plan(initial_lane_id: int, initial_s: float, lookahead_distance: float, route_plan: RoutePlan) \
            -> Tuple[List[FrenetSubSegment], float]:
        """
        Advances downstream according to plan as long as there is a single valid (according to navigation plan)
        downstream lane only - this can be in case there is a single downstream lane, or in a case of a split where only
        one of the downstream lanes are in the navigation plan (the others are not)
        :param initial_lane_id: the initial lane_id (the vehicle is current on)
        :param initial_s: initial longitude along <initial_lane_id>
        :param lookahead_distance: the desired distance of lookahead in [m].
        :param route_plan: the relevant navigation plan to iterate over its road IDs.
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
                raise NavigationPlanTooShort("Cannot progress further on plan %s (leftover: %s [m]); "
                                             "current_segment_end_s=%f lookahead_distance=%f" %
                                             (route_plan.s_Data.a_i_road_segment_ids,
                                              lookahead_distance - cumulative_distance,
                                              current_segment_end_s, lookahead_distance))

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
    def get_stop_bar_and_stop_sign(lane_frenet: GeneralizedFrenetSerretFrame) -> List[RoadSignInfo]:
        """
        Returns a list of the locations (s coordinates) of stop signs and stop bars on the GFF, with their type
        The list is ordered from closest traffic flow control to farthest.
        :param lane_frenet: The GFF on which to retrieve the static flow controls.
        :return: A list of distances to stop signs and stop bars on the the GFF, ordered from closest traffic flow
        control to farthest, along with the type of the control.
        """
        road_signs = MapUtils.get_static_traffic_flow_controls_s(lane_frenet)  # ensures the returned value is sorted
        # TODO verify these are the correct stop bar enums
        desired_sign_types = [RoadObjectType.StopSign, RoadObjectType.StopBar_Left, RoadObjectType.StopBar_Right]
        stop_bars_and_signs = MapUtils.filter_flow_control_by_type(road_signs, desired_sign_types)
        return stop_bars_and_signs

    @staticmethod
    def filter_flow_control_by_type(road_signs: List[RoadSignInfo], road_sign_types: List[RoadObjectType]) -> List[
        RoadSignInfo]:
        """
        Filters the given road signs list to the desired road sign types
        :param road_signs: list of road signs to filter
        :param road_sign_types: list of desired road sign types
        :return: filtered list of road signs
        """
        selected_road_signs = []
        for road_sign in road_signs:
            if road_sign.sign_type in road_sign_types:
                selected_road_signs.append(road_sign)
        return selected_road_signs

    @staticmethod
    def get_static_traffic_flow_controls_s(lane_frenet: GeneralizedFrenetSerretFrame) -> List[RoadSignInfo]:
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
        road_sign_info_on_gff = [RoadSignInfo(sign_type=sign_type, s=s) for sign_type, s in
                                 zip(road_sign_types, road_sign_s_on_gff)]
        road_sign_info_on_gff.sort(key=lambda x: x.s)  # sort by distance after the conversion to real distance
        return road_sign_info_on_gff
