from typing import List

import numpy as np

from decision_making.src.global_constants import TRAJECTORY_ARCLEN_RESOLUTION
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FP_DX, FP_SX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from mapping.src.service.map_service import MapService
from mapping.src.transformations.geometry_utils import CartesianFrame


class MapUtils:

    @staticmethod
    def get_road_by_lane(lane_id: int) -> int:
        """
        get road_id containing the lane
        :param lane_id:
        :return: road_id
        """
        return MapService.get_instance()._lane_address[lane_id][0]

    @staticmethod
    def get_lane_index(lane_id: int) -> int:
        """
        get lane index of the lane on the road (the rightest lane's index is 0)
        :param lane_id:
        :return: lane index
        """
        return MapService.get_instance()._lane_address[lane_id][1]

    @staticmethod
    def get_lane_frenet_frame(lane_id: int) -> FrenetSerret2DFrame:
        """
        get Frenet frame of the whole center-lane for the given lane
        :param lane_id:
        :return: Frenet frame
        """
        return MapService.get_instance()._lane_frenet[lane_id]

    @staticmethod
    def get_lane_length(lane_id: int) -> float:
        """
        get the whole lane's length
        :param lane_id:
        :return: lane's length
        """
        return MapService.get_instance()._lane_frenet[lane_id].s_max

    @staticmethod
    def get_adjacent_lane(lane_id: int, relative_lane: RelativeLane) -> int:
        """
        get adjacent (right/left) lane to the given lane segment (if exists)
        :param lane_id:
        :param relative_lane: either right or left
        :return: adjacent lane id or None if it does not exist
        """
        map_api = MapService.get_instance()
        road_id, lane_idx = map_api._lane_address[lane_id]
        adjacent_idx = lane_idx + relative_lane.value
        return map_api._lane_by_address[(road_id, adjacent_idx)] \
            if (road_id, adjacent_idx) in map_api._lane_by_address else None

    @staticmethod
    def get_closest_lane(x: float, y: float, road_segment_id: int=None) -> int:
        """
        given cartesian coordinates, find the closest lane to the point
        :param x: X cartesian coordinate
        :param y: Y cartesian coordinate
        :param road_segment_id: optional argument for road_segment_id closest to the given point
        :return: closest lane segment id
        """
        map_api = MapService.get_instance()
        if road_segment_id is None:
            # find the closest road segment
            map_api = MapService.get_instance()
            relevant_road_ids = map_api._find_roads_containing_point(x, y)
            closest_road_id = map_api._find_closest_road(x, y, relevant_road_ids)
        else:
            closest_road_id = road_segment_id

        # find the closest lane segment, given the closest road segment
        num_lanes = map_api.get_road(closest_road_id).lanes_num
        # convert the given cpoint to fpoints w.r.t. to each lane's frenet frame
        fpoints = {}
        for lane_idx in range(num_lanes):
            lane_id = map_api._lane_by_address[(closest_road_id, lane_idx)]
            fpoints[lane_id] = map_api._lane_frenet[lane_id].cpoint_to_fpoint(np.array([x, y]))
        # find frenet points with minimal absolute latitude
        return min(fpoints.items(), key=lambda p: abs(p[1][FP_DX]))[0]

    @staticmethod
    def get_dist_from_lane_borders(lane_id: int, s: float) -> (float, float):
        """
        get distance from the lane center to the lane borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right lane border, distance from the left lane border
        """
        # this implementation assumes constant lane width (ignores the argument s)
        lane_width = MapService.get_instance().get_road(MapUtils.get_road_by_lane(lane_id)).lane_width
        return lane_width/2, lane_width/2

    @staticmethod
    def get_dist_from_road_borders(lane_id: int, s: float) -> (float, float):
        """
        get distance from the lane center to the road borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right road border, distance from the left road border
        """
        # this implementation assumes constant lane width (ignores the argument s), the same width of all road's lanes
        map_api = MapService.get_instance()
        road_segment_id = MapUtils.get_road_by_lane(lane_id)
        lane_width = map_api.get_road(road_segment_id).lane_width
        num_lanes = map_api.get_road(road_segment_id).lanes_num
        lane_idx = MapUtils.get_lane_index(lane_id)
        return (lane_idx + 0.5)*lane_width, (num_lanes - lane_idx - 0.5)*lane_width

    @staticmethod
    def get_lane_width(lane_id: int, s: float) -> float:
        """
        get lane width at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: lane width
        """
        dist_from_right, dist_from_left = MapUtils.get_dist_from_lane_borders(lane_id, s)
        return dist_from_right + dist_from_left

    @staticmethod
    def get_lookahead_frenet_frame(lane_id: int, starting_lon: float, lookahead_dist: float,
                                   navigation_plan: NavigationPlanMsg):
        """
        Get Frenet frame of a given length along lane center, starting from given lane's longitude (may be negative).
        When some lane finishes, it automatically continues to the next lane, according to the navigation plan.
        :param lane_id: starting lane_id
        :param starting_lon: starting longitude (may be negative) [m]
        :param lookahead_dist: lookahead distance for the output frame [m]
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: Frenet frame for the given route part
        """
        # in current implementation: if starting_lon < 0, extract Frenet frame with only positive longitudes
        if starting_lon < 0:
            lookahead_dist += starting_lon
            starting_lon = 0

        map_api = MapService.get_instance()
        road_segment_id = MapUtils.get_road_by_lane(lane_id)
        # convert starting lane fpoint to starting road fpoint
        starting_cpoint = MapUtils.get_lane_frenet_frame(lane_id).fpoint_to_cpoint(np.array([starting_lon, 0]))
        starting_road_fpoint = map_api._rhs_roads_frenet[road_segment_id].cpoint_to_fpoint(starting_cpoint)
        # use old get_lookahead_points by the road coordinates
        shifted, _ = map_api.get_lookahead_points(initial_road_id=road_segment_id,
                                                  initial_lon=starting_road_fpoint[FP_SX],
                                                  lookahead_dist=lookahead_dist,
                                                  desired_lat=starting_road_fpoint[FP_DX],
                                                  navigation_plan=navigation_plan)
        # TODO change to precise resampling
        _, resampled, _ = CartesianFrame.resample_curve(curve=shifted, step_size=TRAJECTORY_ARCLEN_RESOLUTION)

        center_lane_reference_route = FrenetSerret2DFrame.fit(resampled)
        return center_lane_reference_route

    @staticmethod
    def get_upstream_lanes(lane_id: int) -> List[int]:
        """
        get upstream lanes of the given lane
        :param lane_id:
        :return: list of upstream lanes
        """
        pass

    @staticmethod
    def get_downstream_lanes(lane_id: int) -> List[int]:
        """
        get downstream lanes of the given lane
        :param lane_id:
        :return: list of downstream lanes
        """
        pass

    @staticmethod
    def is_main_lane(lane_id: int) -> bool:
        """
        check if there is another lane with higher priority, having the same upstream lane
        :param lane_id:
        :return: True if there is no another lane with higher priority, having the same upstream lane
        """
        pass
