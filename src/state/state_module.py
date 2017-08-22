import time

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from spav.decision_making.src.map.naive_cache_map import NaiveCacheMap
from spav.decision_making.src.global_constants import *
from spav.decision_making.src.infra.dm_module import DmModule
from spav.decision_making.src.state.state import *
from logging import Logger


class StateModule(DmModule):
    def __init__(self, dds: DdsPubSub, logger: Logger):
        super().__init__(dds, logger)

        self.map_api = NaiveCacheMap(MAP_FILE_NAME)

        occupancy_state = OccupancyState(0, np.array([]), np.array([]))
        dynamic_objects = []
        size = ObjectSize(0, 0, 0)
        ego_state = EgoState(0, 0, 0, 0, 0, 0, size, 0, 0, 0, 0, 0, 0, 0)
        self.state = State(occupancy_state, dynamic_objects, ego_state)

    def _start_impl(self):
        self.dds.subscribe(DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC, self.__dynamic_obj_callback)
        self.dds.subscribe(SELF_LOCALIZATION_SUBSCRIBE_TOPIC, self.__self_localization_callback)
        self.dds.subscribe(OCCUPANCY_STATE_SUBSCRIBE_TOPIC, self.__occupancy_state_callback)

    def _stop_impl(self):
        self.dds.unsubscribe(DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC)
        self.dds.unsubscribe(SELF_LOCALIZATION_SUBSCRIBE_TOPIC)
        self.dds.unsubscribe(OCCUPANCY_STATE_SUBSCRIBE_TOPIC)

    def _periodic_action_impl(self):
        pass

    def __dynamic_obj_callback(self, objects: dict):
        self.logger.info("got dynamic objects %s", objects)

        timestamp = objects["timestamp"]
        dyn_obj_list_dict = objects["timestamp"]["dynamic_objects"]

        dyn_obj_list = self.state.dynamic_objects
        for dyn_obj_dict in dyn_obj_list_dict:
            id = dyn_obj_dict["id"]
            x = dyn_obj_dict["location"]["x"]
            y = dyn_obj_dict["location"]["y"]
            z = 0
            yaw = dyn_obj_dict["bbox"]["yaw"]
            confidence = dyn_obj_dict["location"]["confidence"]
            length = dyn_obj_dict["bbox"]["length"]
            width = dyn_obj_dict["bbox"]["width"]
            height = dyn_obj_dict["bbox"]["height"]
            size = ObjectSize(length, width, height)
            v_x = dyn_obj_dict["velocity"]["v_x"]
            v_y = dyn_obj_dict["velocity"]["v_y"]

            dyn_obj = DynamicObject(id, timestamp, x, y, z, yaw, size, confidence, v_x, v_y, None, None)
            self.fill_road_localization(dyn_obj)
            dyn_obj_list.append(dyn_obj)

    def __self_localization_callback(self, ego_localization: dict):
        self.logger.debug("got self localization %s", ego_localization)

        confidence = ego_localization["location"]["confidence"]
        timestamp = ego_localization["timestamp"]
        x = ego_localization["location"]["x"]
        y = ego_localization["location"]["y"]
        z = 0
        yaw = ego_localization["yaw"]
        v_x = ego_localization["velocity"]["v_x"]
        v_y = ego_localization["velocity"]["v_y"]
        size = ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT)
        self.state.ego_state = EgoState(0, timestamp, x, y, z, yaw, size, confidence, v_x, v_y, None, None, None)
        self.fill_road_localization(self.state.ego_state)

    def __occupancy_state_callback(self, occupancy: dict):
        self.logger.debug("got occupancy status %s", occupancy)
        timestamp = occupancy["timestamp"]
        points_list_dict = occupancy["points_list"]
        points_list = []
        confidence_list = []
        for pnt_dict in points_list_dict:
            pnt = [pnt_dict["x"], pnt_dict["y"], pnt_dict["z"]]
            points_list.append(pnt)
            confidence_list.append(pnt_dict["confidence"])
        self.state.occupancy_state = OccupancyState(timestamp, np.ndarray(points_list), np.ndarray(confidence_list))

    def __actuator_status_callback(self, actuator: dict):
        self.logger.debug("got actuator status %s", actuator)
        self.state.ego_state.steering_angle = actuator["steering_angle"]

    def fill_road_localization(self, obj):
        # type: (DynamicObject) -> None
        """
        given ego_state fill dyn_obj.road_localization & dyn_obj.rel_road_localization
        :param obj: dynamic object whose road_localization should be filled (may be ego)
        :return: None
        """
        ego = self.state.ego_state
        if type(obj) is not EgoState:  # if the object is not ego
            rel_pos = np.array([obj.x, obj.y, obj.z])
            inv_ego_position = np.array([-ego.x, -ego.y, -ego.z])
            glob_pos = CartesianFrame.get_vector_in_objective_frame(rel_pos, inv_ego_position, -ego.yaw)
            glob_yaw = obj.yaw + ego.yaw
        else:  # if obj is ego, then global & local coordinates are the same
            glob_pos = np.array([obj.x, obj.y, obj.z])
            glob_yaw = obj.yaw

        # calculate road coordinates for global coordinates
        road_id, lane_num, full_lat, intra_lane_lat, lon, intra_lane_yaw = \
            self.map_api.convert_world_to_lat_lon(glob_pos[0], glob_pos[1], glob_pos[2], glob_yaw)
        # fill road_localization
        obj.road_localization = RoadLocalization(road_id, lane_num, full_lat, intra_lane_lat, lon, intra_lane_yaw)

        # calculate relative road localization
        if type(obj) is not EgoState:  # if obj is not ego
            obj.rel_road_localization = \
                RelativeRoadLocalization(obj.road_localization.full_lat - ego.road_localization.full_lat,
                                         obj.road_localization.road_lon - ego.road_localization.road_lon,
                                         obj.road_localization.intra_lane_yaw - ego.road_localization.intra_lane_yaw)
        else:  # if the object is ego, then rel_road_localization is irrelevant
            obj.rel_road_localization = None
