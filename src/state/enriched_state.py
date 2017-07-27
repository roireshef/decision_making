from typing import List
import copy
import numpy as np
from state.state import *


class EnrichedRoadLocalization(RoadLocalization):
    def __init__(self, road_id: int, lane: int, intra_lane_lat: float, road_lon: float, intra_lane_yaw: float):
        """
        location in road coordinates (road_id, lat, lon)
        :param road_id:
        :param lane: 0 is the leftmost
        :param intra_lane_lat: in meters, 0 is lane left edge
        :param road_lon: in meters, longitude relatively to the road start
        :param intra_lane_yaw: 0 is along road's local tangent
        """
        RoadLocalization.__init__(self, road_id, lane, intra_lane_lat, road_lon, intra_lane_yaw)


class EnrichedOccupancyState(OccupancyState):
    def __init__(self, free_space: np.ndarray, confidence: np.ndarray):
        """
        free space description
        :param free_space: array of directed segments defines a free space border
        :param confidence: array per segment
        """
        OccupancyState.__init__(self, free_space, confidence)


class EnrichedObjectSize(ObjectSize):
    def __init__(self, length: float, width: float, height: float):
        ObjectSize.__init__(self, length, width, height)


class EnrichedObjectState(ObjectState):
    def __init__(self, obj_id: int, timestamp: int, x: float, y: float, z: float, yaw: float, size: EnrichedObjectSize,
                 road_localization: EnrichedRoadLocalization, confidence: float, localization_confidence: float):
        """
        base class for ego, static & dynamic objects
        :param obj_id: object id
        :param timestamp: time of perception
        :param x: for ego in world coordinates, for the rest relatively to ego
        :param y:
        :param z:
        :param yaw: for ego 0 means along X axis, for the rest 0 means forward direction relatively to ego
        :param size: class ObjectSize
        :param road_localization: class RoadLocalization
        :param confidence: of object's existence
        :param localization_confidence: of location
        """
        ObjectState.__init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, confidence,
                             localization_confidence)


class EnrichedDynamicObject(EnrichedObjectState, DynamicObject):
    def __init__(self, obj_id: int, timestamp: int, x: float, y: float, z: float, yaw: float, size: EnrichedObjectSize,
                 road_localization: EnrichedRoadLocalization, confidence: float, localization_confidence: float,
                 v_x: float, v_y: float, acceleration_x: float, turn_radius: float):
        """
        both ego and other dynamic objects
        :param obj_id:
        :param timestamp:
        :param x:
        :param y:
        :param z:
        :param yaw:
        :param size:
        :param road_localization:
        :param confidence:
        :param localization_confidence:
        :param v_x: in m/sec; for ego in world coordinates, for the rest relatively to ego
        :param v_y: in m/sec
        :param acceleration_x: acceleration in longitude axis
        :param turn_radius: 0 for straight motion, positive for CW (yaw increases), negative for CCW
        """
        EnrichedObjectState.__init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, confidence,
                                     localization_confidence)
        DynamicObject.__init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, confidence,
                               localization_confidence,
                               v_x, v_y)
        self.acceleration_lon = acceleration_x
        self.turn_radius = turn_radius

    def predict(self, goal_timestamp: int):
        """
        predict the object's location for the future timestamp
        !!! This function changes the object's location, velocity and timestamp !!!
        :param timestamp:
        :return:
        """
        dt = goal_timestamp - self._timestamp
        sin_yaw = np.sin(self.yaw)
        cos_yaw = np.cos(self.yaw)
        acc_x = self.acceleration_lon * cos_yaw
        acc_y = self.acceleration_lon * sin_yaw
        if self.turn_radius is not None and self.turn_radius != 0:
            start_vel = np.sqrt(self.v_x * self.v_x + self.v_y * self.v_y)
            dist = start_vel * dt + 0.5 * self.acceleration_lon * dt * dt
            turn_angle = dist / self.turn_radius
            next_yaw = self.yaw + turn_angle
            circle_center = [self.x - self.turn_radius * sin_yaw,
                             self.y + self.turn_radius * cos_yaw]
            end_loc = [circle_center[0] + self.turn_radius * np.sin(next_yaw),
                       circle_center[1] - self.turn_radius * np.cos(next_yaw)]
        else:  # straight movement
            end_loc = [self.x + self.v_x * dt + 0.5 * acc_x * dt * dt, self.y + self.v_y * dt + 0.5 * acc_y * dt * dt]

        self.x = end_loc[0]
        self.y = end_loc[1]
        self.v_x += dt * acc_x
        self.v_y += dt * acc_y
        self._timestamp = goal_timestamp


class EnrichedEgoState(EnrichedDynamicObject, EgoState):
    def __init__(self, obj_id: int, timestamp: int, x: float, y: float, z: float, yaw: float, size: EnrichedObjectSize,
                 road_localization: EnrichedRoadLocalization, confidence: float, localization_confidence: float,
                 v_x: float, v_y: float, acceleration_x: float, turn_radius: float, steering_angle: float):
        """
        :param obj_id:
        :param timestamp:
        :param x:
        :param y:
        :param z:
        :param yaw:
        :param size:
        :param road_localization:
        :param confidence:
        :param localization_confidence:
        :param v_x: in m/sec
        :param v_y: in m/sec
        :param acceleration_x: in m/s^2
        :param turn_radius: radius of turning of the ego
        :param steering_angle: equivalent to knowing of turn_radius
        """
        EnrichedDynamicObject.__init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, confidence,
                                       localization_confidence,
                                       v_x, v_y, acceleration_x, turn_radius)
        EgoState.__init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, confidence,
                          localization_confidence,
                          v_x, v_y, steering_angle)


class EnrichedLanesStructure(LanesStructure):
    def __init__(self, center_of_lane_points: np.ndarray, width_vec: np.ndarray):
        """
        this class is instantiated for each lane
        :param center_of_lane_points:  points array for a given lane
        :param width_vec:  array of width: lane width per lane point
        """
        LanesStructure.__init__(self, center_of_lane_points, width_vec)


class EnrichedPerceivedRoad(PerceivedRoad):
    def __init__(self, timestamp: int, lanes_structure: List[EnrichedLanesStructure], confidence: float):
        """
        the road of ego as it viewed by perception
        :param timestamp:
        :param lanes_structure: list of elements of type LanesStructure, per lane
        :param confidence:
        """
        PerceivedRoad.__init__(self, timestamp, lanes_structure, confidence)


class EnrichedState(State):
    def __init__(self, occupancy_state: EnrichedOccupancyState, static_objects:
    List[EnrichedObjectState], dynamic_objects: List[EnrichedDynamicObject],
                 ego_state: EnrichedEgoState, perceived_road: EnrichedPerceivedRoad):
        """
        main class for the world state
        :param occupancy_state: free space
        :param static_objects:
        :param dynamic_objects:
        :param ego_state:
        :param perceived_road: the road of ego as it viewed by perception, relatively to ego
        """
        State.__init__(self, occupancy_state, static_objects, dynamic_objects, ego_state, perceived_road)

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

    def predict(self, timestamp: int):
        """
        predict the ego localization and other objects for the future timestamp
        :param timestamp:
        :return:
        """
        # update dynamic objects
        for dyn_obj in self.dynamic_objects:
            dyn_obj.predict(timestamp)
        # backup ego_state
        prev_ego_state = copy.copy(self.ego_state)
        # update ego_state
        self.ego_state.predict(timestamp)

        # since all objects are given relatively to ego, update them accordingly to ego change
        rot_ang = self.ego_state.yaw - prev_ego_state.yaw
        cosa = np.cos(rot_ang)
        sina = np.sin(rot_ang)
        dx = self.ego_state.x - prev_ego_state.x
        dy = self.ego_state.y - prev_ego_state.y
        # update static objects according to ego change
        for stat_obj in self.static_objects:
            new_x = stat_obj.x * cosa + stat_obj.y * sina - dx
            new_y = -stat_obj.y * sina + stat_obj.y * cosa - dy
            stat_obj.x = new_x
            stat_obj.y = new_y
        # update dynamic objects according to ego change
        for dyn_obj in self.dynamic_objects:
            new_x = dyn_obj.x * cosa + dyn_obj.y * sina - dx
            new_y = -dyn_obj.y * sina + dyn_obj.y * cosa - dy
            dyn_obj.x = new_x
            dyn_obj.y = new_y
            dyn_obj.v_x -= self.ego_state.v_x - prev_ego_state.v_x
            dyn_obj.v_y -= self.ego_state.v_y - prev_ego_state.v_y
            # TODO update free space similarly to static objects
