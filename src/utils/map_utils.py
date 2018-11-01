import numpy as np

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FP_DX, FP_SX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from mapping.src.service.map_service import MapService
from mapping.src.transformations.geometry_utils import CartesianFrame


class MapUtils:
    # TODO: replace with navigation plan aware function from map API

    @staticmethod
    def get_road_by_lane(lane_id):
        # type: (int) -> int
        """
        get road_id containing the lane
        :param lane_id:
        :return: road_id
        """
        return MapService.get_instance()._lane_address[lane_id][0]

    @staticmethod
    def get_lane_index(lane_id):
        # type: (int) -> int
        """
        get lane index of the lane on the road (the rightest lane's index is 0)
        :param lane_id:
        :return: lane index
        """
        return MapService.get_instance()._lane_address[lane_id][1]

    @staticmethod
    def get_lane_frenet(lane_id):
        # type: (int) -> FrenetSerret2DFrame
        """
        get Frenet frame of the whole center-lane for the given lane
        :param lane_id:
        :return: Frenet frame
        """
        return MapService.get_instance()._lane_frenet[lane_id]

    @staticmethod
    def get_lane_length(lane_id):
        # type: (int) -> float
        """
        get the whole lane's length
        :param lane_id:
        :return: lane's length
        """
        return MapService.get_instance()._lane_frenet[lane_id].s_max

    @staticmethod
    def get_adjacent_lane(lane_id, relative_lane):
        # type: (int, RelativeLane) -> int
        map_api = MapService.get_instance()
        road_id, lane_idx = map_api._lane_address[lane_id]
        adjacent_idx = lane_idx + relative_lane.value
        return map_api._lane_by_address[(road_id, adjacent_idx)] \
            if (road_id, adjacent_idx) in map_api._lane_by_address else None

    @staticmethod
    def get_closest_lane(x, y):
        # type: (float, float) -> int
        """
        given cartesian coordinates, find the closest lane to the point
        :param x: X cartesian coordinate
        :param y: Y cartesian coordinate
        :return: closest lane id
        """
        map_api = MapService.get_instance()
        relevant_road_ids = map_api._find_roads_containing_point(x, y)
        closest_road_id = map_api._find_closest_road(x, y, relevant_road_ids)
        return MapUtils.get_closest_lane_on_road(closest_road_id, x, y)

    @staticmethod
    def get_closest_lane_on_road(road_id, x, y):
        # type: (int, float, float) -> int
        map_api = MapService.get_instance()
        num_lanes = map_api.get_road(road_id).lanes_num
        # convert the given cpoint to fpoints w.r.t. to each lane's frenet frame
        fpoints = {}
        for lane_idx in range(num_lanes):
            lane_id = map_api._lane_by_address[(road_id, lane_idx)]
            fpoints[lane_id] = map_api._lane_frenet[lane_id].cpoint_to_fpoint(np.array([x, y]))
        # find frenet points with minimal absolute latitude
        return min(fpoints.items(), key=lambda p: abs(p[1][FP_DX]))[0]

    @staticmethod
    def get_dist_from_lane_borders(lane_id, s):
        # type: (int, float) -> (float, float)
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
    def get_dist_from_road_borders(lane_id, s):
        # type: (int, float) -> (float, float)
        """
        get distance from the lane center to the road borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right road border, distance from the left road border
        """
        # this implementation assumes constant lane width (ignores the argument s), the same width of all road's lanes
        map_api = MapService.get_instance()
        road_id = MapUtils.get_road_by_lane(lane_id)
        lane_width = map_api.get_road(road_id).lane_width
        num_lanes = map_api.get_road(road_id).lanes_num
        lane_idx = MapUtils.get_lane_index(lane_id)
        return (lane_idx + 0.5)*lane_width, (num_lanes - lane_idx - 0.5)*lane_width

    @staticmethod
    def get_lane_width(lane_id, s):
        # type: (int) -> float
        """
        get lane width at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: lane width
        """
        dist_from_right, dist_from_left = MapUtils.get_dist_from_lane_borders(lane_id, s)
        return dist_from_right + dist_from_left

    @staticmethod
    def get_uniform_path_lookahead(lane_id: int, lane_lat_shift: float, starting_lon: float, lon_step: float,
                                   steps_num: int, navigation_plan: NavigationPlanMsg):
        """
        Create array of uniformly distributed points along a given road, shifted laterally by by lat_shift.
        When some road finishes, it automatically continues to the next road, according to the navigation plan.
        The distance between consecutive points is lon_step.
        :param lane_id: starting lane_id
        :param lane_lat_shift: lateral shift from right side of the lane [m]
        :param starting_lon: starting longitude [m]
        :param lon_step: distance between consecutive points [m]
        :param steps_num: output points number
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: uniform sampled points array (Nx2)
        """
        map_api = MapService.get_instance()
        road_id = MapUtils.get_road_by_lane(lane_id)
        # convert starting lane fpoint to starting road fpoint
        starting_cpoint = MapUtils.get_lane_frenet(lane_id).fpoint_to_cpoint(np.array([starting_lon, lane_lat_shift]))
        starting_road_fpoint = map_api._rhs_roads_frenet[road_id].cpoint_to_fpoint(starting_cpoint)
        # use old get_lookahead_points by the road coordinates
        shifted, _ = map_api.get_lookahead_points(initial_road_id=road_id,
                                                  initial_lon=starting_road_fpoint[FP_SX],
                                                  lookahead_dist=lon_step * steps_num,
                                                  desired_lat=starting_road_fpoint[FP_DX],
                                                  navigation_plan=navigation_plan)
        # TODO change to precise resampling
        _, resampled, _ = CartesianFrame.resample_curve(curve=shifted, step_size=lon_step)
        return resampled
