import copy
from typing import List, Union

import numpy as np
# from decision_making.src.map.constants import *
from decision_making.src.map.map_api import MapAPI
from decision_making.src.messages.dds_typed_message import DDSTypedMsg
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.utils.geometry_utils import CartesianFrame


class RoadLocalization(DDSTypedMsg):
    def __init__(self, road_id, lane_num, full_lat, intra_lane_lat, road_lon, intra_lane_yaw):
        # type: (int, int, float, float, float, float, float, float) -> None
        """
        absolute location in road coordinates (road_id, lat, lon)
        :param road_id:
        :param lane_num: 0 is the rightmost
        :param full_lat: in meters; full latitude from the right edge of the road
        :param intra_lane_lat: in meters, 0 is lane left edge
        :param road_lon: in meters, longitude relatively to the road start
        :param intra_lane_yaw: 0 is along road's local tangent
        """
        self.road_id = road_id
        self.lane_num = lane_num
        self.full_lat = full_lat
        self.intra_lane_lat = intra_lane_lat
        self.road_lon = road_lon
        self.intra_lane_yaw = intra_lane_yaw


class RelativeRoadLocalization(DDSTypedMsg):
    def __init__(self, rel_lat, rel_lon, rel_yaw):
        # type: (float, float, float) -> None
        """
        relative to ego location in road coordinates (road_id, lat, lon)
        :param rel_lat: in meters, latitude relatively to ego
        :param rel_lon: in meters, longitude relatively to ego
        :param rel_yaw: in radians, yaw relatively to ego
        """
        self.rel_lat = rel_lat
        self.rel_lon = rel_lon
        self.rel_yaw = rel_yaw


class OccupancyState(DDSTypedMsg):
    def __init__(self, timestamp, free_space, confidence):
        # type: (int, np.ndarray, np.ndarray) -> None
        """
        free space description
        :param timestamp of free space
        :param free_space: array of directed segments defines a free space border
        :param confidence: array per segment
        """
        self._timestamp = timestamp
        self.free_space = np.copy(free_space)
        self.confidence = np.copy(confidence)


class ObjectSize(DDSTypedMsg):
    def __init__(self, length, width, height):
        # type: (float, float, float) -> None
        self.length = length
        self.width = width
        self.height = height


class DynamicObject(DDSTypedMsg):
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence, v_x, v_y, acceleration_lon, yaw_deriv,
                 road_localization):
        # type: (int, int, float, float, float, float, ObjectSize, float, float, float, Union[float, None], Union[float, None], RoadLocalization) -> None
        """
        both ego and other dynamic objects
        :param obj_id: object id
        :param timestamp: time of perception
        :param x: for ego in world coordinates, for the rest relatively to ego
        :param y:
        :param z:
        :param yaw: for ego 0 means along X axis, for the rest 0 means forward direction relatively to ego
        :param size: class ObjectSize
        :param confidence: of object's existence
        :param v_x: in m/sec; for ego in world coordinates, for the rest relatively to ego
        :param v_y: in m/sec
        :param acceleration_lon: acceleration in longitude axis
        :param yaw_deriv: 0 for straight motion, positive for CCW (yaw increases), negative for CW
        """
        self.id = obj_id
        self._timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.size = copy.copy(size)
        self.confidence = confidence
        self.v_x = v_x
        self.v_y = v_y
        self.road_localization = road_localization

        if acceleration_lon is not None:
            self.acceleration_lon = acceleration_lon
        else:
            raise NotImplementedError()

        if yaw_deriv is not None:
            self.yaw_deriv = yaw_deriv
        else:
            raise NotImplementedError()

    def predict(self, goal_timestamp, map_api):
        # type: (int, MapAPI) -> DynamicObject
        """
        Predict the object's location for the future timestamp
        !!! This function changes the object's location, velocity and timestamp !!!
        :param goal_timestamp: the goal timestamp for prediction
        :param lane_width: closest lane_width
        :return: predicted DynamicObject
        """
        pass

    def get_relative_road_localization(self, ego_road_localization, ego_navigation_plan, map_api, max_lookahead_dist):
        # type: (RoadLocalization, NavigationPlanMsg, MapAPI, float) -> RelativeRoadLocalization
        """
        Returns a relative road localization (to given ego state)
        :param ego_road_localization: base road location
        :param ego_navigation_plan: the ego vehicle navigation plan
        :param map_api: the map which will be used to calculate the road localization
        :param max_lookahead_dist: lookahead in [m] to search from ego towards dynamic object on map
        :return: a RelativeRoadLocalization object
        """
        relative_lon = map_api.get_point_relative_longitude(from_road_id=ego_road_localization.road_id,
                                                            from_lon_in_road=ego_road_localization.road_lon,
                                                            to_road_id=self.road_localization.road_id,
                                                            to_lon_in_road=self.road_localization.road_lon,
                                                            max_lookahead_distance=max_lookahead_dist,
                                                            navigation_plan=ego_navigation_plan)

        if relative_lon is None:
            # set object at infinity
            RelativeRoadLocalization(rel_lat=0, rel_lon=np.inf, rel_yaw=0)
        else:
            relative_lat = self.road_localization.full_lat - ego_road_localization.full_lat
            relative_yaw = self.road_localization.intra_lane_yaw - ego_road_localization.intra_lane_yaw
            return RelativeRoadLocalization(rel_lat=relative_lat, rel_lon=relative_lon, rel_yaw=relative_yaw)


class EgoState(DynamicObject, DDSTypedMsg):
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence,
                 v_x, v_y, acceleration_lon, yaw_deriv, steering_angle, road_localization):
        # type: (int, int, float, float, float, float, ObjectSize, float, float, float, Union[float, None], Union[float, None], Union[float, None], RoadLocalization) -> None
        """
        :param obj_id:
        :param timestamp:
        :param x:
        :param y:
        :param z:
        :param yaw:
        :param size:
        :param confidence:
        :param v_x: in m/sec
        :param v_y: in m/sec
        :param acceleration_lon: in m/s^2
        :param yaw_deriv: radius of turning of the ego
        :param steering_angle: equivalent to knowing of turn_radius
        """
        DynamicObject.__init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence, v_x, v_y,
                               acceleration_lon, yaw_deriv, road_localization)
        if steering_angle is not None:
            self.steering_angle = steering_angle
        else:
            raise NotImplementedError()


class State(DDSTypedMsg):
    def __init__(self, occupancy_state, dynamic_objects, ego_state):
        # type: (OccupancyState, List[DynamicObject], EgoState) -> None
        """
        main class for the world state
        :param occupancy_state: free space
        :param dynamic_objects:
        :param ego_state:
        """
        self.occupancy_state = copy.deepcopy(occupancy_state)
        self.dynamic_objects = copy.deepcopy(dynamic_objects)
        self.ego_state = copy.deepcopy(ego_state)

    def update_objects(self):
        """
        insert object to state - will be implemented by Ron
        :return: merged State
        """
        pass

    def update_ego_state(self):
        """
        insert ego localization to state - will be implemented by Ron
        :return: merged State
        """
        pass

    def predict(self, goal_timestamp, map_api):
        # type: (int, MapAPI) -> State
        """
        predict the ego localization, other objects and free space for the future timestamp
        :param goal_timestamp:
        :return: predicted State
        """
        pass
