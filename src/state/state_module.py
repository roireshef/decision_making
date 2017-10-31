from threading import Lock
from typing import Optional, Tuple

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import *
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.state.state import *
from mapping.src.exceptions import MapCellNotFound
from mapping.src.transformations.geometry_utils import CartesianFrame


class StateModule(DmModule):
    # TODO: temporary solution for unknown class members on initialization
    UNKNWON_DEFAULT_VAL = 0.0

    # TODO: implement double-buffer mechanism for locks wherever needed
    def __init__(self, dds: DdsPubSub, logger: Logger, map_api: MapAPI, occupancy_state: Optional[OccupancyState],
                 dynamic_objects: Optional[List[DynamicObject]], ego_state: Optional[EgoState],
                 dynamic_objects_average_location: dict = {}, dynamic_objects_history: dict = {}):
        super().__init__(dds, logger)
        self._map_api = map_api

        self._occupancy_state = occupancy_state
        self._occupancy_state_lock = Lock()

        self._dynamic_objects = dynamic_objects
        self._dynamic_objects_lock = Lock()

        self._ego_state = ego_state
        self._ego_state_lock = Lock()

        self._dynamic_objects_average_location = dynamic_objects_average_location
        self._dynamic_objects_history = dynamic_objects_history

    def _start_impl(self):
        self.dds.subscribe(DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC, self._dynamic_obj_callback)
        self.dds.subscribe(SELF_LOCALIZATION_SUBSCRIBE_TOPIC, self._self_localization_callback)
        # TODO: invalid!
        # self.dds.subscribe(OCCUPANCY_STATE_SUBSCRIBE_TOPIC, self._occupancy_state_callback)

    def _stop_impl(self):
        self.dds.unsubscribe(DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC)
        self.dds.unsubscribe(SELF_LOCALIZATION_SUBSCRIBE_TOPIC)
        # TODO: invalid!
        # self.dds.unsubscribe(OCCUPANCY_STATE_SUBSCRIBE_TOPIC)

    def _periodic_action_impl(self):
        pass

    def _dynamic_obj_callback(self, objects: dict) -> None:
        try:
            self.logger.info("got dynamic objects %s", objects)

            if self._ego_state is None:
                self.logger.warning(
                    "StateModule is trying to parse dynamic objects with None EgoState. Since objects " +
                    "are given in ego-vehicle's coordinate frame this is impossible. Aborting.")
                return

            ego = self._ego_state
            ego_pos = np.array([ego.x, ego.y, ego.z])
            ego_yaw = ego.yaw

            timestamp = objects["timestamp"]
            dyn_obj_list_dict = objects["dynamic_objects"]

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
                is_predicted = dyn_obj_dict["tracking_status"]["is_predicted"]
                in_fov = dyn_obj_dict["tracking_status"]["in_fov"]
                omega_yaw = dyn_obj_dict["velocity"]["omega_yaw"]

                # TODO - temporary! conversion to global coords, until perception delivers the global coords.
                #global_coordinates, global_yaw = CartesianFrame.convert_relative_to_global_frame(
                #    np.array([x, y, z]), yaw, ego_pos, ego_yaw)
                global_coordinates = np.array([x, y, z])
                global_yaw = yaw

                try:
                    # Try to localize object on road. If not successful, warn.
                    road_localtization = DynamicObject.compute_road_localization(global_coordinates, global_yaw,
                                                                                self._map_api)

                    dyn_obj = DynamicObject(id, timestamp, global_coordinates[0], global_coordinates[1],
                                            global_coordinates[2], global_yaw, size, confidence, v_x, v_y,
                                            self.UNKNWON_DEFAULT_VAL, omega_yaw, road_localtization)
                    dyn_obj_list.append(dyn_obj)
                except MapCellNotFound:
                    self.logger.warning(
                        "Couldn't localize object id {} on road. Object location: ({}, {}, {})".format(id, x, y, z))

            with self._dynamic_objects_lock:
                self._dynamic_objects = dyn_obj_list

            self._publish_state_if_full()
        except Exception as e:
            self.logger.error("StateModule._dynamic_obj_callback failed due to {}".format(e))

    def _self_localization_callback(self, ego_localization: dict):
        try:
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

            road_localization = DynamicObject.compute_road_localization(np.array([x, y, z]), yaw, self._map_api)

            with self._ego_state_lock:
                # TODO: replace UNKNWON_DEFAULT_VAL with actual implementation
                self._ego_state = EgoState(0, timestamp, x, y, z, yaw, size, confidence, v_x, v_y,
                                           self.UNKNWON_DEFAULT_VAL,
                                           self.UNKNWON_DEFAULT_VAL, self.UNKNWON_DEFAULT_VAL, road_localization)

            self._publish_state_if_full()
        except Exception as e:
            self.logger.error("StateModule._self_localization_callback failed due to {}".format(e))

    # TODO: handle invalid data
    def _occupancy_state_callback(self, occupancy: dict):
        try:
            self.logger.debug("got occupancy status %s", occupancy)
            timestamp = occupancy["timestamp"]

            free_space_points = np.array(occupancy["free_space_points"], dtype=float)
            points_list = free_space_points[:, :3]
            confidence_list = free_space_points[:, 3]

            with self._occupancy_state_lock:
                self._occupancy_state = OccupancyState(timestamp, np.array(points_list), np.array(confidence_list))

            self._publish_state_if_full()
        except Exception as e:
            self.logger.error("StateModule._occupancy_state_callback failed due to {}".format(e))

    # TODO: integrate compensation for time differences (aka short-time predict)
    def _publish_state_if_full(self):
        # if some part of the state is missing, don't publish state message
        if self._occupancy_state is None or self._dynamic_objects is None or self._ego_state is None:
            return

        with self._occupancy_state_lock, self._ego_state_lock, self._dynamic_objects_lock:
            state = State(self._occupancy_state, self._dynamic_objects, self._ego_state)
        self.logger.debug("publishing state %s", state.serialize())

        self.dds.publish(STATE_PUBLISH_TOPIC, state.serialize())

    # TODO: solve the fact that actuator status can be outdated and no one will ever know
    def _actuator_status_callback(self, actuator: dict):
        self.logger.debug("got actuator status %s", actuator)
        pass  # TODO: update self._ego_state.steering_angle. Don't forget to lock self._ego_state!

    @staticmethod
    def _compute_road_localization(global_pos: np.ndarray, global_yaw: float, map_api: MapAPI) -> RoadLocalization:
        """
        calculate road coordinates for global coordinates for ego
        :param global_pos: 1D numpy array of ego vehicle's [x,y,z] in global coordinate-frame
        :param global_yaw: in global coordinate-frame
        :param map_api: MapAPI instance
        :return: the road localization
        """
        closest_road_id, lon, lat, global_yaw, is_on_road = map_api.convert_global_to_road_coordinates(global_pos[0],
                                                                                                       global_pos[1],
                                                                                                       global_yaw)
        lane_width = map_api.get_road(closest_road_id).lane_width
        lane = np.math.floor(lat / lane_width)
        intra_lane_lat = lat - lane * lane_width

        return RoadLocalization(closest_road_id, int(lane), lat, intra_lane_lat, lon, global_yaw)
