from typing import List
import numpy as np
import copy

from decision_making.src.planning.utils.geometry_utils import Dynamics
from decision_making.src.map.constants import *
from decision_making.src.messages.dds_typed_message import DDSTypedMsg


class RoadLocalization(DDSTypedMsg):
    def __init__(self, road_id: int, lane_num: int, intra_lane_lat: float, road_lon: float, intra_lane_yaw: float,
                 road_confidence: float, lane_confidence: float):
        """
        location in road coordinates (road_id, lat, lon)
        :param road_id:
        :param lane_num: 0 is the leftmost
        :param intra_lane_lat: in meters, 0 is lane left edge
        :param road_lon: in meters, longitude relatively to the road start
        :param intra_lane_yaw: 0 is along road's local tangent
        :param road_confidence: confidence of road_id & road_lon
        :param lane_confidence: confidence of lane, intra_lane params
        """
        self.road_id = road_id
        self.lane_num = lane_num
        self.intra_lane_lat = intra_lane_lat
        self.road_lon = road_lon
        self.intra_lane_yaw = intra_lane_yaw
        self.road_confidence = road_confidence
        self.lane_confidence = lane_confidence


class OccupancyState(DDSTypedMsg):
    def __init__(self, timestamp: int, free_space: np.ndarray, confidence: np.ndarray):
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
    def __init__(self, length: float, width: float, height: float):
        self.length = length
        self.width = width
        self.height = height


class DynamicObject(DDSTypedMsg):
    def __init__(self, obj_id: int, timestamp: int, x: float, y: float, z: float, yaw: float, size: ObjectSize,
                 road_localization: RoadLocalization, confidence: float, localization_confidence: float,
                 v_x: float, v_y: float, acceleration_lon: float, turn_radius: float):
        """
        both ego and other dynamic objects
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
        :param v_x: in m/sec; for ego in world coordinates, for the rest relatively to ego
        :param v_y: in m/sec
        :param acceleration_lon: acceleration in longitude axis
        :param turn_radius: 0 for straight motion, positive for CW (yaw increases), negative for CCW
        """
        self.id = obj_id
        self._timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.size = copy.deepcopy(size)
        self.road_localization = copy.deepcopy(road_localization)
        self.confidence = confidence
        self.localization_confidence = localization_confidence
        self.v_x = v_x
        self.v_y = v_y
        self.acceleration_lon = acceleration_lon
        self.turn_radius = turn_radius

    def predict(self, goal_timestamp: int, lane_width: float) -> None:
        """
        Predict the object's location for the future timestamp
        !!! This function changes the object's location, velocity and timestamp !!!
        :param goal_timestamp: the goal timestamp for prediction
        :param lane_width: closest lane_width
        :return: None
        """
        (goal_x, goal_y, goal_yaw, goal_v_x, goal_v_y) = \
            Dynamics.predict_dynamics(self.x, self.y, self.yaw, self.v_x, self.v_y, self.acceleration_lon,
                                      self.turn_radius, 0.001 * (goal_timestamp - self._timestamp))

        # Predict the object's road_localization in the future goal_timestamp.
        # Here we assume that the lanes are straight and have the same width.
        # calc full latitude from the left road's edge
        full_lat = self.road_localization.lane_num * lane_width + self.road_localization.intra_lane_lat

        # calc velocity relatively to the road
        vel = np.sqrt(self.v_x * self.v_x + self.v_y * self.v_y)
        rel_road_v_x = vel * np.cos(self.road_localization.intra_lane_yaw)
        rel_road_v_y = vel * np.sin(self.road_localization.intra_lane_yaw)

        (goal_lon, goal_lat, goal_lane_yaw, _, _) = \
            Dynamics.predict_dynamics(x=self.road_localization.road_lon, y=full_lat,
                                      yaw=self.road_localization.intra_lane_yaw,
                                      v_x=rel_road_v_x, v_y=rel_road_v_y,
                                      accel_lon=self.acceleration_lon, turn_radius=self.turn_radius,
                                      dt=0.001 * (goal_timestamp - self._timestamp))
        # update road_localization
        self.road_localization.road_lon = goal_lon
        self.road_localization.lane_num = int(goal_lat / lane_width)
        self.road_localization.intra_lane_lat = goal_lat % lane_width
        self.road_localization.intra_lane_yaw = goal_lane_yaw

        (self.x, self.y, self.yaw, self.v_x, self.v_y) = (goal_x, goal_y, goal_yaw, goal_v_x, goal_v_y)

        self._timestamp = goal_timestamp


class EgoState(DynamicObject, DDSTypedMsg):
    def __init__(self, obj_id: int, timestamp: int, x: float, y: float, z: float, yaw: float, size: ObjectSize,
                 road_localization: RoadLocalization, confidence: float, localization_confidence: float,
                 v_x: float, v_y: float, acceleration_lon: float, turn_radius: float, steering_angle: float):
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
        :param acceleration_lon: in m/s^2
        :param turn_radius: radius of turning of the ego
        :param steering_angle: equivalent to knowing of turn_radius
        """
        DynamicObject.__init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, confidence,
                               localization_confidence, v_x, v_y, acceleration_lon, turn_radius)
        self.steering_angle = steering_angle


class LanesStructure(DDSTypedMsg):
    def __init__(self, center_of_lane_points: np.ndarray, width_vec: np.ndarray):
        """
        this class is instantiated for each lane
        :param center_of_lane_points:  points array for a given lane
        :param width_vec:  array of width: lane width per lane point
        """
        self.center_of_lane_points = copy.deepcopy(center_of_lane_points)
        self.width_vec = copy.deepcopy(width_vec)


class PerceivedRoad(DDSTypedMsg):
    def __init__(self, timestamp: int, lanes_structure: List[LanesStructure], confidence: float):
        """
        the road of ego as it viewed by perception
        :param timestamp:
        :param lanes_structure: list of elements of type LanesStructure, per lane
        :param confidence:
        """
        self.timestamp = timestamp
        self.lanes_structure = copy.deepcopy(lanes_structure)
        self.confidence = confidence

    def serialize(self):
        serialized_state = super().serialize()
        # handle lists of complex types
        lanes_list = list()
        for lane in self.lanes_structure:
            lanes_list.append(lane.serialize())

        serialized_state['lanes_structure'] = lanes_list

        return serialized_state


class State(DDSTypedMsg):
    def __init__(self, occupancy_state: OccupancyState, dynamic_objects: List[DynamicObject],
                 ego_state: EgoState, perceived_road: PerceivedRoad):
        """
        main class for the world state
        :param occupancy_state: free space
        :param dynamic_objects:
        :param ego_state:
        :param perceived_road: the road of ego as it viewed by perception, relatively to ego
        """
        self.occupancy_state = copy.deepcopy(occupancy_state)
        self.dynamic_objects = copy.deepcopy(dynamic_objects)
        self.ego_state = copy.deepcopy(ego_state)
        self.perceived_road = copy.deepcopy(perceived_road)

    @classmethod
    def create_empty(cls):
        occupancy_state = OccupancyState(0, np.array([]), np.array([]))
        dynamic_objects = []
        size = ObjectSize(0, 0, 0)
        road_localization = RoadLocalization(0, 0, 0, 0, 0, 0, 0)
        ego_state = EgoState(0, 0, 0, 0, 0, 0, size, road_localization, 0, 0, 0, 0, 0)
        perceived_road = PerceivedRoad(0, [], 0)
        state = cls(occupancy_state, dynamic_objects, ego_state, perceived_road)
        return state

    def serialize(self):
        serialized_state = super().serialize()
        # handle lists of complex types
        dynamic_objects_list = list()
        for obj in self.dynamic_objects:
            dynamic_objects_list.append(obj.serialize())

        serialized_state['dynamic_objects'] = dynamic_objects_list

        return serialized_state

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

    def predict(self, goal_timestamp: int):
        """
        predict the ego localization, other objects and free space for the future timestamp
        :param goal_timestamp:
        :return:
        """
        # backup ego_state
        prev_ego_state = copy.copy(self.ego_state)

        # get the closest lane width
        if len(self.perceived_road.lanes_structure) > self.ego_state.road_localization.lane_num:
            lane_width = self.perceived_road.lanes_structure[self.ego_state.road_localization.lane_num].width_vec[0]
        else:
            lane_width = CUSTOM_LANE_WIDTH

        # update ego_state
        self.ego_state.predict(goal_timestamp, lane_width)

        # update dynamic objects parameters without consideration of the ego accelerations (lon & angular)
        for dyn_obj in self.dynamic_objects:
            # convert to global velocity
            dyn_obj.v_x += prev_ego_state.v_x
            dyn_obj.v_y += prev_ego_state.v_y
            # predict dyn_obj for the goal_timestamp
            dyn_obj.predict(goal_timestamp, lane_width)

        # since all objects are given relatively to ego, update them accordingly to ego change
        rot_ang = prev_ego_state.yaw - self.ego_state.yaw
        cosa = np.cos(rot_ang)
        sina = np.sin(rot_ang)
        # calc ego change
        dx = prev_ego_state.x - self.ego_state.x
        dy = prev_ego_state.y - self.ego_state.y
        dz = prev_ego_state.z - self.ego_state.z
        d_yaw = prev_ego_state.yaw - self.ego_state.yaw
        # compensate on ego motion
        for dyn_obj in self.dynamic_objects:
            (dyn_obj.x, dyn_obj.y) = Dynamics.rotate_and_shift_point(dyn_obj.x, dyn_obj.y, cosa, sina, dx, dy)
            # convert to velocity relative to ego
            (dyn_obj.v_x, dyn_obj.v_y) = Dynamics.rotate_and_shift_point(dyn_obj.v_x, dyn_obj.v_y, cosa, sina,
                                                                         -self.ego_state.v_x, -self.ego_state.v_y)
            dyn_obj.z += dz
            dyn_obj.yaw += d_yaw

        # update free space vertices according to ego change
        self.occupancy_state.free_space = \
            Dynamics.rotate_and_shift_points(self.occupancy_state.free_space, cosa, sina, dx, dy)
