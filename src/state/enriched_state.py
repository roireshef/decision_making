
import numpy as np
from typing import Dict, Tuple, List
from state import state

class RoadLocalization(state.RoadLocalization):
    def __init__(self, road_id: int, lane: int, intra_lane_lat: float, road_lon: float, intra_lane_yaw: float):
        '''
        location in road coordinates (road_id, lat, lon)
        :param road_id:
        :param lane: 0 is the leftmost
        :param intra_lane_lat: in meters, 0 is lane left edge
        :param road_lon: in meters, longitude relatively to the road start
        :param intra_lane_yaw: 0 is along road's local tangent
        '''
        super().__init__(road_id, lane, intra_lane_lat, road_lon, intra_lane_yaw)

class OccupancyState(state.OccupancyState):
    def __init__(self, free_space: np.ndarray, confidence: np.ndarray):
        '''
        free space description
        :param free_space: array of directed segments defines a free space border
        :param confidence: array per segment
        '''
        super().__init__(free_space, confidence)

class ObjectSize(state.ObjectSize):
    def __init__(self, length: float, width: float, height: float):
        super().__init__(length, width, height)

class ObjectState(state.ObjectState):
    def __init__(self, id: int, timestamp: int, x: float, y: float, z: float, yaw: float, size: ObjectSize,
                 road_localization: RoadLocalization, confidence: float, localization_confidence: float):
        '''
        base class for ego, static & dynamic objects
        :param id: object id
        :param timestamp: time of perception
        :param x: for ego in world coordinates, for the rest relatively to ego
        :param y:
        :param z:
        :param yaw: for ego 0 means along X axis, for the rest 0 means forward direction relatively to ego
        :param size: class ObjectSize
        :param road_localization: class RoadLocalization
        :param confidence: of object's existence
        :param localization_confidence: of location
        '''
        super(ObjectState, self).__init__(id, timestamp, x,y,z,yaw, size, road_localization, confidence, localization_confidence)

class DynamicObject(ObjectState, state.DynamicObject):
    def __init__(self, id: int, timestamp: int, x: float, y: float, z: float, yaw: float, size: ObjectSize,
                 road_localization: RoadLocalization, confidence: float, localization_confidence: float,
                 v_x: float, v_y: float, acceleration_x: float, turn_radius: float):
        '''
        both ego and other dynamic objects
        :param id:
        :param timestamp:
        :param x:
        :param y:
        :param z:
        :param yaw:
        :param size:
        :param road_localization:
        :param confidence:
        :param localization_confidence:
        :param v_x: for ego in world coordinates, for the rest relatively to ego
        :param v_y:
        '''
        super().__init__(id, timestamp, x, y, z, yaw, size, road_localization, confidence, localization_confidence)
        super().__init__(id, timestamp, x, y, z, yaw, size, road_localization, confidence, localization_confidence, v_x, v_y)
        self.acceleration_x = acceleration_x
        self.turn_radius = turn_radius

    def predict(self, timestamp: int):
        '''
        predict the object's location for the future timestamp
        :param timestamp:
        :return:
        '''
        pass

class EgoState(DynamicObject, state.EgoState):
    def __init__(self, id: int, timestamp: int, x: float, y: float, z: float, yaw: float, size: ObjectSize,
                 road_localization: RoadLocalization, confidence: float, localization_confidence: float,
                 v_x: float, v_y: float, acceleration_x: float, turn_radius: float, steering_angle: float):
        '''
        :param id:
        :param timestamp:
        :param x:
        :param y:
        :param z:
        :param yaw:
        :param size:
        :param road_localization:
        :param confidence:
        :param localization_confidence:
        :param v_x:
        :param v_y:
        :param steering_angle: equivalent to knowing of turn_radius
        '''
        super().__init__(id, timestamp, x,y,z,yaw, size, road_localization, confidence, localization_confidence,
                         v_x, v_y, acceleration_x, turn_radius)
        super().__init__(id, timestamp, x, y, z, yaw, size, road_localization, confidence, localization_confidence,
                         v_x, v_y, steering_angle)

class LanesStructure(state.LanesStructure):
    def __init__(self, center_of_lane_points: np.ndarray, width_vec: np.ndarray):
        '''
        this class is instantiated for each lane
        :param center_of_lane_points:  points array for a given lane
        :param width_vec:  array of width: lane width per lane point
        '''
        super().__init__(center_of_lane_points, width_vec)

class PerceivedRoad(state.PerceivedRoad):
    def __init__(self, timestamp: int, lanes_structure: List[LanesStructure], confidence: float):
        '''
        the road of ego as it viewed by perception
        :param timestamp:
        :param lanes_structure: list of elements of type LanesStructure, per lane
        :param confidence:
        '''
        super().__init__(timestamp, lanes_structure, confidence)

class State(state.State):
    def __init__(self, occupancy_state: OccupancyState, static_objects: ObjectState, dynamic_objects: DynamicObject,
                 ego_state: EgoState, perceived_road: PerceivedRoad):
        '''
        main class for the world state
        :param occupancy_state: free space
        :param static_objects:
        :param dynamic_objects:
        :param ego_state:
        :param perceived_road: the road of ego as it viewed by perception, relatively to ego
        '''
        super().__init__(occupancy_state, static_objects, dynamic_objects, ego_state, perceived_road)

    def update_objects(self):
        '''
        insert object to state - will be implemented by Ron
        :return: merged State
        '''
        pass

    def update_ego_state(self):
        '''
        insert ego localization to state - will be implemented by Ron
        :return: merged State
        '''
        pass

    def predict(self, timestamp: int):
        '''
        predict the ego localization and other objects for the future timestamp
        :param timestamp:
        :return:
        '''
        pass

