import numpy as np


class RoadLocalization:
    def __init__(self, road_id, lane, intra_lane_lat, road_lon, intra_lane_yaw):
        self.road_id = road_id
        self.lane_num = lane
        self.intra_lane_lat = intra_lane_lat
        self.road_lon = road_lon
        self.intra_lane_yaw = intra_lane_yaw

class OccupancyState:
    def __init__(self, free_space):
        self.free_space = free_space

class ObjectSize:
    def __init__(self, length, width, height):
        self.length = length
        self.width = width
        self.height = height

class ObjectState:  # some fields are missing
    def __init__(self, id, timestamp, x,y,z,yaw, size, road_localization):
        self.id = id
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.object_size = size
        self.road_localization = road_localization

class DynamicObject(ObjectState):
    def __init__(self, id, timestamp, x,y,z,yaw, size, road_localization, v_x, v_y):
        super().__init__(id, timestamp, x,y,z,yaw, size, road_localization)
        self.v_x = v_x
        self.v_y = v_y

class EgoState(DynamicObject):
    def __init__(self, id, timestamp, x,y,z,yaw, size, road_localization, v_x, v_y, turn_radius):
        super().__init__(id, timestamp, x,y,z,yaw, size, road_localization, v_x, v_y)
        self.turn_radius = turn_radius

class LanesStructure:
    def __init__(self, center_of_lane_points, width_vec):
        self.center_of_lane_points = center_of_lane_points
        self.width_vec = width_vec

class PerceivedRoad:
    def __init__(self, lanes_structure):
        self.lanes_structure = lanes_structure

class State:  # should include the full State
    def __init__(self, occupancy_state, static_objects, dynamic_objects, ego_state, perceived_road):
        self.occupancy_state = occupancy_state
        self.static_objects = static_objects
        self.dynamic_objects = dynamic_objects
        self.ego_state = ego_state
        self.perceived_road = perceived_road
