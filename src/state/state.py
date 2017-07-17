import numpy as np

# location in road coordinates
class RoadLocalization:
    def __init__(self, road_id, lane, intra_lane_lat, road_lon, intra_lane_yaw):
        self.road_id = road_id
        self.lane_num = lane
        self.intra_lane_lat = intra_lane_lat
        self.road_lon = road_lon  # longitude relatively to the road start
        self.intra_lane_yaw = intra_lane_yaw  # yaw relatively to the local road tangent

class OccupancyState:
    def __init__(self, free_space, confidence):
        self.free_space = free_space  # list of directed segments defines a free space border
        self.confidence = confidence  # list per segment

class ObjectSize:
    def __init__(self, length, width, height):
        self.length = length
        self.width = width
        self.height = height

class ObjectState:
    def __init__(self, id, timestamp, x,y,z,yaw, size, road_localization, confidence, localization_confidence):
        self.id = id
        self._timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.object_size = size  # class ObjectSize
        self.road_localization = road_localization  # class RoadLocalization
        self.confidence = confidence
        self.localization_confidence = localization_confidence

class DynamicObject(ObjectState):
    def __init__(self, id, timestamp, x,y,z,yaw, size, road_localization, confidence, localization_confidence, v_x, v_y):
        super().__init__(id, timestamp, x,y,z,yaw, size, road_localization, confidence, localization_confidence)
        self.v_x = v_x
        self.v_y = v_y

class EgoState(DynamicObject):
    def __init__(self, id, timestamp, x,y,z,yaw, size, road_localization, confidence, localization_confidence, v_x, v_y, turn_radius):
        super().__init__(id, timestamp, x,y,z,yaw, size, road_localization, confidence, localization_confidence, v_x, v_y)
        self.turn_radius = turn_radius  # equivalent to knowing of steering angle

# this class is instantiated for each lane
class LanesStructure:
    def __init__(self, center_of_lane_points, width_vec):
        self.center_of_lane_points = center_of_lane_points  # points array for a given lane
        self.width_vec = width_vec  # array of width: lane width per lane point

class PerceivedRoad:
    def __init__(self, lanes_structure):
        self.lanes_structure = lanes_structure  # list of elements of type LanesStructure, per lane

# main class for the world state
class State:
    def __init__(self, occupancy_state, static_objects, dynamic_objects, ego_state, perceived_road):
        self.occupancy_state = occupancy_state  # free space
        self.static_objects = static_objects
        self.dynamic_objects = dynamic_objects
        self.ego_state = ego_state
        self.perceived_road = perceived_road  # the road of ego as it viewed by perception, relatively to ego
