class RoadLocalization:
    def __init__(self, road_id, lane, intra_lane_lat, road_lon, intra_lane_yaw):
        """
        location in road coordinates (road_id, lat, lon)
        :param road_id:
        :param lane: 0 is the leftmost
        :param intra_lane_lat: in meters, 0 is lane left edge
        :param road_lon: in meters, longitude relatively to the road start
        :param intra_lane_yaw: 0 is along road's local tangent
        """
        self.road_id = road_id
        self.lane_num = lane
        self.intra_lane_lat = intra_lane_lat
        self.road_lon = road_lon
        self.intra_lane_yaw = intra_lane_yaw


class OccupancyState:
    def __init__(self, free_space, confidence):
        """
        free space description
        :param free_space: list of directed segments defines a free space border
        :param confidence: list per segment
        """
        self.free_space = free_space
        self.confidence = confidence


class ObjectSize:
    def __init__(self, length, width, height):
        self.length = length
        self.width = width
        self.height = height


class ObjectState:
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, confidence, localization_confidence):
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
        self.id = obj_id
        self._timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.object_size = size
        self.road_localization = road_localization
        self.confidence = confidence
        self.localization_confidence = localization_confidence


class DynamicObject(ObjectState):
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, confidence, localization_confidence,
                 v_x, v_y):
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
        """
        ObjectState.__init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, confidence, localization_confidence)
        self.v_x = v_x
        self.v_y = v_y


class EgoState(DynamicObject):
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, confidence, localization_confidence,
                 v_x, v_y, steering_angle):
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
        :param steering_angle: equivalent to knowing of turn_radius
        """
        DynamicObject.__init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, confidence, localization_confidence,
                         v_x, v_y)
        self.steering_angle = steering_angle


class LanesStructure:
    def __init__(self, center_of_lane_points, width_vec):
        """
        this class is instantiated for each lane
        :param center_of_lane_points:  # points array for a given lane
        :param width_vec:  # array of width: lane width per lane point
        """
        self.center_of_lane_points = center_of_lane_points
        self.width_vec = width_vec


class PerceivedRoad:
    def __init__(self, timestamp, lanes_structure, confidence):
        """
        the road of ego as it viewed by perception
        :param timestamp:
        :param lanes_structure: list of elements of type LanesStructure, per lane
        :param confidence:
        """
        self._timestamp = timestamp
        self.lanes_structure = lanes_structure
        self.confidence = confidence


class State():
    def __init__(self, occupancy_state, static_objects, dynamic_objects, ego_state, perceived_road):
        """
        main class for the world state
        :param occupancy_state: free space
        :param static_objects:
        :param dynamic_objects:
        :param ego_state:
        :param perceived_road: the road of ego as it viewed by perception, relatively to ego
        """
        super.__init__()
        self.occupancy_state = occupancy_state
        self.static_objects = static_objects
        self.dynamic_objects = dynamic_objects
        self.ego_state = ego_state
        self.perceived_road = perceived_road
