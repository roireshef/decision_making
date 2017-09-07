from threading import Lock
from typing import Optional

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import *
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.planning.utils.geometry_utils import CartesianFrame
from decision_making.src.state.state import *


class StateModule(DmModule):
    # TODO: temporary solution for unknown class members on initialization
    UNKNWON_DEFAULT_VAL = 0.0

    # TODO: implement double-buffer mechanism for locks wherever needed
    def __init__(self, dds: DdsPubSub, logger: Logger, map_api: MapAPI, occupancy_state: Optional[OccupancyState],
                 dynamic_objects: Optional[List[DynamicObject]], ego_state: Optional[EgoState]):
        super().__init__(dds, logger)
        self._map_api = map_api

        self._occupancy_state = occupancy_state
        self._occupancy_state_lock = Lock()

        self._dynamic_objects = dynamic_objects
        self._dynamic_objects_lock = Lock()

        self._ego_state = ego_state
        self._ego_state_lock = Lock()

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

    def __dynamic_obj_callback(self, objects: dict) -> None:
        self.logger.info("got dynamic objects %s", objects)

        if self._ego_state is None:
            self.logger.warning("StateModule is trying to parse dynamic objects with None EgoState. " +
                                "Since objects are given in ego-vehicle's coordinate frame this is impossible. Aborting.")
            return

        ego = self._ego_state
        ego_pos = np.array([ego.x, ego.y, ego.z])
        ego_yaw = ego.yaw

        timestamp = objects["timestamp"]
        dyn_obj_list_dict = objects["dynamic_objects"]

        # TODO: can we use history-knowledge about dynamic objects?
        dyn_obj_list = []
        for dyn_obj_dict in dyn_obj_list_dict:
            id = dyn_obj_dict["id"]
            x = dyn_obj_dict["location"]["x"]
            y = dyn_obj_dict["location"]["y"]
            z = DEFAULT_OBJECT_Z_VALUE
            yaw = dyn_obj_dict["bbox"]["yaw"]
            confidence = dyn_obj_dict["location"]["confidence"]
            length = dyn_obj_dict["bbox"]["length"]
            width = dyn_obj_dict["bbox"]["width"]
            height = dyn_obj_dict["bbox"]["height"]
            size = ObjectSize(length, width, height)
            v_x = dyn_obj_dict["velocity"]["v_x"]
            v_y = dyn_obj_dict["velocity"]["v_y"]

            obj_pos = np.array([x, y, z])

            road_localtization = StateModule.__compute_obj_road_localization(obj_pos, yaw, ego_pos, ego_yaw,
                                                                             self._map_api)

            # TODO: replace UNKNWON_DEFAULT_VAL with actual implementation
            dyn_obj = DynamicObject(id, timestamp, x, y, z, yaw, size, confidence, v_x, v_y,
                                    self.UNKNWON_DEFAULT_VAL, self.UNKNWON_DEFAULT_VAL, road_localtization)
            dyn_obj_list.append(dyn_obj)

        with self._dynamic_objects_lock:
            self._dynamic_objects = dyn_obj_list

        self.__publish_state_if_full()

    def __self_localization_callback(self, ego_localization: dict):
        self.logger.debug("got self localization %s", ego_localization)

        confidence = ego_localization["location"]["confidence"]
        timestamp = ego_localization["timestamp"]
        x = ego_localization["location"]["x"]
        y = ego_localization["location"]["y"]
        z = 0.0
        yaw = ego_localization["yaw"]
        v_x = ego_localization["velocity"]["v_x"]
        v_y = ego_localization["velocity"]["v_y"]
        size = ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT)

        road_localization = StateModule.__compute_ego_road_localization(np.array([x, y, z]), yaw, self._map_api)

        with self._ego_state_lock:
            # TODO: replace UNKNWON_DEFAULT_VAL with actual implementation
            self._ego_state = EgoState(0, timestamp, x, y, z, yaw, size, confidence, v_x, v_y, self.UNKNWON_DEFAULT_VAL,
                                       self.UNKNWON_DEFAULT_VAL, self.UNKNWON_DEFAULT_VAL, road_localization)

        self.__publish_state_if_full()

    def __occupancy_state_callback(self, occupancy: dict):
        self.logger.debug("got occupancy status %s", occupancy)
        timestamp = occupancy["timestamp"]

        free_space_points = np.array(occupancy["free_space_points"], dtype=float)
        points_list = free_space_points[:, :3]
        confidence_list = free_space_points[:, 3]

        with self._occupancy_state_lock:
            self._occupancy_state = OccupancyState(timestamp, np.array(points_list), np.array(confidence_list))

        self.__publish_state_if_full()

    # TODO: integrate compensation for time differences (aka short-time predict)
    def __publish_state_if_full(self):
        # if some part of the state is missing, don't publish state message
        if self._occupancy_state is None or self._dynamic_objects is None or self._ego_state is None:
            return

        with self._occupancy_state_lock, self._ego_state_lock, self._dynamic_objects_lock:
            state = State(self._occupancy_state, self._dynamic_objects, self._ego_state)
        self.logger.debug("publishing state %s", state.serialize())

        self.dds.publish(STATE_PUBLISH_TOPIC, state.serialize())

    # TODO: solve the fact that actuator status can be outdated and no one will ever know
    def __actuator_status_callback(self, actuator: dict):
        self.logger.debug("got actuator status %s", actuator)
        pass  # TODO: update self._ego_state.steering_angle. Don't forget to lock self._ego_state!

    @staticmethod
    def __compute_ego_road_localization(pos: np.ndarray, yaw: float, map_api: MapAPI) -> RoadLocalization:
        """
        calculate road coordinates for global coordinates for ego
        :param pos: 1D numpy array of ego vehicle's [x,y,z] in global coordinate-frame
        :param yaw: in global coordinate-frame
        :param map_api: MapAPI instance
        :return: the road localization
        """
        # road_id, lane_num, full_lat, intra_lane_lat, lon, intra_lane_yaw = \
        #     map_api.convert_world_to_lat_lon(pos[0], pos[1], pos[2], yaw)
        # return RoadLocalization(road_id, lane_num, full_lat, intra_lane_lat, lon, intra_lane_yaw)
        # TODO - return real localization
        return RoadLocalization(0, 0, 0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def __compute_obj_road_localization(obj_pos: np.ndarray, obj_yaw: float, ego_pos: np.ndarray, ego_yaw: float,
                                        map_api: MapAPI) -> RoadLocalization:
        """
        given an object in ego-vehicle's coordinate-frame, calculate its road coordinates

        :return:
        """
        global_obj_pos = CartesianFrame.convert_relative_to_global_frame(obj_pos, ego_pos, ego_yaw)
        global_obj_yaw = obj_yaw + ego_yaw

        # road_id, lane_num, full_lat, intra_lane_lat, lon, intra_lane_yaw = \
        #     map_api.convert_world_to_lat_lon(global_obj_pos[0], global_obj_pos[1], global_obj_pos[2], global_obj_yaw)
        # return RoadLocalization(road_id, lane_num, full_lat, intra_lane_lat, lon, intra_lane_yaw)
        # TODO - return real localization
        return RoadLocalization(0, 0, 0, 0, 0, 0)
