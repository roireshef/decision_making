from enum import Enum
from typing import List
import numpy as np


class ObjectClassification(Enum):
    CAR = 1


class MapLaneType(Enum):
    REGULAR = 1


class LaneSegmentType(Enum):
    UP = 1
    DOWN = 2


class MapLaneMarkerType(Enum):
    REGULAR = 1


class RoadIntersection:
    pass


class RoadSegment:
    def __init__(self, road_segment_id: int, road_id: int):
        self.road_segment_id = road_segment_id
        self.road_id = road_id


class AdjacentLane:
    def __init__(self, lane_segment_id: int, moving_direction: LaneSegmentType, lane_type: MapLaneType):
        self.lane_segment_id = lane_segment_id
        self.moving_direction = moving_direction
        self.lane_type = lane_type


class NominalPathPoint:
    def __init__(self, east_x: float, north_y: float, heading: float, curvature: float, curvature_rate: float,
                 cross_slope: float, along_slope: float, s: float, left_offset: float, right_offset: float):
        self.east_x = east_x
        self.north_y = north_y
        self.heading = heading
        self.curvature = curvature
        self.curvature_rate = curvature_rate
        self.cross_slope = cross_slope
        self.along_slope = along_slope
        self.s = s
        self.left_offset = left_offset
        self.right_offset = right_offset


class Boundary:
    def __init__(self, type: MapLaneMarkerType, s_start: float, s_end: float):
        self.type = type
        self.s_start = s_start
        self.s_end = s_end


class LaneSegment:
    def __init__(self, lane_segment_id: int, road_segment_id: int, lane_type: MapLaneType, nominal_speed: float,
                 left_adjacent_lane_count: int, s_left_adjacent_lanes: List[AdjacentLane],
                 right_adjacent_lane_count: int, s_right_adjacent_lanes: List[AdjacentLane],
                 nominal_path_points_count: int, s_nominal_path_points: List[NominalPathPoint],
                 left_boundary_count: int, s_left_boundary_type: List[Boundary], right_boundary_count: int,
                 s_right_boundary_type: List[Boundary]):
        self.lane_segment_id = lane_segment_id
        self.road_segment_id = road_segment_id
        self.lane_type = lane_type
        self.nominal_speed = nominal_speed
        self.left_adjacent_lane_count = left_adjacent_lane_count
        self.s_left_adjacent_lanes = s_left_adjacent_lanes
        self.right_adjacent_lane_count = right_adjacent_lane_count
        self.s_right_adjacent_lanes = s_right_adjacent_lanes
        self.nominal_path_points_count = nominal_path_points_count
        self.s_nominal_path_points = s_nominal_path_points
        self.left_boundary_count = left_boundary_count
        self.s_left_boundary_type = s_left_boundary_type
        self.right_boundary_count = right_boundary_count
        self.s_right_boundary_type = s_right_boundary_type


class FrenetLocalization:
    def __init__(self, s: float, s_dot: float, s_dotdot: float, d: float, d_dot: float, d_dotdot: float):
        self.s = s
        self.s_dot = s_dot
        self.s_dotdot = s_dotdot
        self.d = d
        self.d_dot = d_dot
        self.d_dotdot = d_dotdot

    def numpy(self):
        return np.array([self.s, self.s_dot, self.s_dotdot, self.d, self.d_dot, self.d_dotdot])

    @classmethod
    def from_numpy(cls, arr: np.ndarray):
        cls(s=arr[0], s_dot=arr[1], s_dotdot=arr[2], d=arr[3], d_dot=arr[4], d_dotdot=arr[5])


class CartesianLocalization:
    def __init__(self, east_x: float, north_y: float, heading: float, yaw_rate: float, velocity_longitudinal: float,
                 acceleration_longitudinal: float, acceleration_lateral: float, curvature: float):
        self.east_x = east_x
        self.north_y = north_y
        self.heading = heading
        self.yaw_rate = yaw_rate
        self.velocity_longitudinal = velocity_longitudinal
        self.acceleration_longitudinal = acceleration_longitudinal
        self.acceleration_lateral = acceleration_lateral
        self.curvature = curvature

    def numpy(self):
        return np.array([self.east_x, self.north_y, self.heading, self.velocity_longitudinal,
                         self.acceleration_longitudinal, self.curvature])

    @classmethod
    def from_numpy(cls, arr: np.ndarray):
        cls(east_x=arr[0], north_y=arr[1], heading=arr[2], yaw_rate=arr[3] * arr[5], velocity_longitudinal=arr[3],
            acceleration_longitudinal=arr[4], acceleration_lateral=arr[5] * arr[3] ** 2, curvature=arr[5])


class HostLocalization:
    def __init__(self, road_segment_id: int, lane_segment_id: int, s_cartesian_localization: CartesianLocalization,
                 s_lane_frenet_coordinate: FrenetLocalization):
        self.road_segment_id = road_segment_id
        self.lane_segment_id = lane_segment_id
        self.s_cartesian_localization = s_cartesian_localization
        self.s_lane_frenet_coordinate = s_lane_frenet_coordinate


class ObjectHypothesis:
    def __init__(self, probability: float, s_cartesian_localization: CartesianLocalization,
                 s_lane_frenet_coordinate: FrenetLocalization, s_host_lane_frenet_coordinate: FrenetLocalization,
                 lane_segment_id: int, stationary_status, location_uncertainty_x,
                 location_uncertainty_y, location_uncertainty_yaw, host_lane_frenet_id: int):
        self.probability = probability
        self.s_cartesian_localization = s_cartesian_localization
        self.s_lane_frenet_coordinate = s_lane_frenet_coordinate
        self.s_host_lane_frenet_coordinate = s_host_lane_frenet_coordinate
        self.lane_segment_id = lane_segment_id
        self.stationary_status = stationary_status
        self.location_uncertainty_x = location_uncertainty_x
        self.location_uncertainty_y = location_uncertainty_y
        self.location_uncertainty_yaw = location_uncertainty_yaw
        self.host_lane_frenet_id = host_lane_frenet_id


class ObjectLocalization:
    def __init__(self, object_id: int, object_type: ObjectClassification,
                 obj_hypothesis_count: int, s_object_hypotheses: List[ObjectHypothesis]):
        self.object_id = object_id
        self.object_type = object_type
        self.obj_hypothesis_count = obj_hypothesis_count
        self.s_object_hypotheses = s_object_hypotheses


class SceneMessage:
    def __init__(self, timestamp_sec: float, perception_horizon_front_m: float, perception_horizon_rear_m: float,
                 road_segment_count: int, road_segment: List[RoadSegment], lane_segment_count: int,
                 lane_segment: List[LaneSegment], host_localization: HostLocalization,
                 object_count: int, object_localizations: List[ObjectLocalization],
                 road_intersection_count: int, road_intersection: List[RoadIntersection]):
        self.timestamp_sec = timestamp_sec
        self.perception_horizon_front_m = perception_horizon_front_m
        self.perception_horizon_rear_m = perception_horizon_rear_m
        self.road_segment_count = road_segment_count
        self.road_segment = road_segment
        self.lane_segment_count = lane_segment_count
        self.lane_segment = lane_segment
        self.host_localization = host_localization
        self.object_count = object_count
        self.object_localizations = object_localizations
        self.road_intersection_count = road_intersection_count
        self.road_intersection = road_intersection
