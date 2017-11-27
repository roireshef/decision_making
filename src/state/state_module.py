from logging import Logger
from threading import Lock
from typing import Optional, List

import numpy as np

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC, SELF_LOCALIZATION_SUBSCRIBE_TOPIC, \
    DEFAULT_OBJECT_Z_VALUE, EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT, STATE_PUBLISH_TOPIC
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.state.state import OccupancyState, EgoState, DynamicObject, ObjectSize, State
from mapping.src.exceptions import MapCellNotFound
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.model.map_api import MapAPI


class StateModule(DmModule):
    # TODO: temporary solution for unknown class members on initialization
    UNKNWON_DEFAULT_VAL = 0.0

    # TODO: implement double-buffer mechanism for locks wherever needed. Current lock mechanism may slow the
    # TODO(cont): processing when multiple events come in consecutively.
    def __init__(self, dds: DdsPubSub, logger: Logger, map_api: MapAPI, occupancy_state: Optional[OccupancyState],
                 dynamic_objects: Optional[List[DynamicObject]], ego_state: Optional[EgoState],
                 dynamic_objects_memory_map: dict = {}) -> None:
        """
        :param dds: Inter-process communication interface.
        :param logger: Logging module.
        :param occupancy_state: Initial state occupancy object.
        :param dynamic_objects: Initial state dynamic objects.
        :param ego_state: Initial ego-state object.
        :param dynamic_objects_memory_map: Initial memory object.
        """
        super().__init__(dds, logger)
        self._map_api = map_api

        # save initial state and generate type-specific locks
        self._occupancy_state = occupancy_state
        self._occupancy_state_lock = Lock()

        self._dynamic_objects = dynamic_objects
        self._dynamic_objects_lock = Lock()

        self._ego_state = ego_state
        self._ego_state_lock = Lock()

        self._dynamic_objects_memory_map = dynamic_objects_memory_map

    def _start_impl(self) -> None:
        """
        When starting the State Module, subscribe to dynamic objects, ego state and occupancy state services.
        """
        self.dds.subscribe(DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC, self._dynamic_obj_callback)
        self.dds.subscribe(SELF_LOCALIZATION_SUBSCRIBE_TOPIC, self._self_localization_callback)
        # TODO: Coordinate interface with perception
        # self.dds.subscribe(OCCUPANCY_STATE_SUBSCRIBE_TOPIC, self._occupancy_state_callback)

    def _stop_impl(self) -> None:
        """
        Unsubscribe from process communication services.
        """
        self.dds.unsubscribe(DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC)
        self.dds.unsubscribe(SELF_LOCALIZATION_SUBSCRIBE_TOPIC)
        # TODO: Coordinate interface with perception
        # self.dds.unsubscribe(OCCUPANCY_STATE_SUBSCRIBE_TOPIC)

    def _periodic_action_impl(self) -> None:
        pass

    def _dynamic_obj_callback(self, objects: dict) -> None:
        """
        Deserialize dynamic objects message and replace pointer reference to dynamic object list under lock
        :param objects: Serialized dynamic object list

        """
        try:
            self.logger.info("got dynamic objects %s", objects)

            if self._ego_state is None:
                self.logger.warning(
                    "StateModule is trying to parse dynamic objects with None EgoState. Since objects " +
                    "are given in ego-vehicle's coordinate frame this is impossible. Aborting.")
                return

            dyn_obj_list = self.create_dyn_obj_list(objects)

            with self._dynamic_objects_lock:
                self._dynamic_objects = dyn_obj_list

            self._publish_state_if_full()
        except Exception as e:
            self.logger.error("StateModule._dynamic_obj_callback failed due to {}".format(e))

    def create_dyn_obj_list(self, objects) -> List[DynamicObject]:
        """
        :param objects: Serialized dynamic objects list.
        :return: List of dynamic object in DM format.
        """
        ego = self._ego_state
        ego_pos = np.array([ego.x, ego.y, ego.z])
        ego_yaw = ego.yaw
        timestamp = objects["timestamp"]
        dyn_obj_list_dict = objects["dynamic_objects"]
        dyn_obj_list = []
        for dyn_obj_dict in dyn_obj_list_dict:
            in_fov = dyn_obj_dict["tracking_status"]["in_fov"]
            id = dyn_obj_dict["id"]
            if in_fov:
                # object is in FOV, so we take its latest detection.
                x = dyn_obj_dict["location"]["x"]
                y = dyn_obj_dict["location"]["y"]
                z = DEFAULT_OBJECT_Z_VALUE
                confidence = dyn_obj_dict["location"]["confidence"]

                yaw = dyn_obj_dict["bbox"]["yaw"]
                length = dyn_obj_dict["bbox"]["length"]
                width = dyn_obj_dict["bbox"]["width"]
                height = dyn_obj_dict["bbox"]["height"]
                size = ObjectSize(length, width, height)

                v_x = dyn_obj_dict["velocity"]["v_x"]
                v_y = dyn_obj_dict["velocity"]["v_y"]
                omega_yaw = dyn_obj_dict["velocity"]["omega_yaw"]

                is_predicted = dyn_obj_dict["tracking_status"]["is_predicted"]

                global_coordinates = np.array([x, y, z])
                global_yaw = yaw

                try:
                    # Try to localize object on road. If not successful, warn.
                    road_localization = DynamicObject.compute_road_localization(global_coordinates, global_yaw,
                                                                                self._map_api)

                    # Filter objects out of road:
                    road_width = self._map_api.get_road(road_id=road_localization.road_id).road_width
                    if road_width + ROAD_SHOULDERS_WIDTH > road_localization.full_lat > -ROAD_SHOULDERS_WIDTH:
                        dyn_obj = DynamicObject(id, timestamp, global_coordinates[0], global_coordinates[1],
                                                global_coordinates[2], global_yaw, size, confidence, v_x, v_y,
                                                self.UNKNWON_DEFAULT_VAL, omega_yaw, road_localization)
                        self._dynamic_objects_memory_map[id] = dyn_obj
                        dyn_obj_list.append(dyn_obj)  # update the list of dynamic objects
                    else:
                        continue

                except MapCellNotFound:
                    self.logger.warning(
                        "Couldn't localize object id {} on road. Object location: ({}, {}, {})".format(id, x, y, z))
            else:
                # object is out of FOV, using its last known location and timestamp.
                dyn_obj = self._dynamic_objects_memory_map.get(id)
                if dyn_obj is not None:
                    dyn_obj_list.append(dyn_obj)  # update the list of dynamic objects
                else:
                    self.logger.warning("received out of FOV object which is not in memory.")
        return dyn_obj_list

    def _self_localization_callback(self, ego_localization: dict) -> None:
        """
        Deserialize localization information, convert to road coordinates and update state information.
        :param ego_localization: Serialized ego localization message.
        """
        try:
            # de-serialize relevant fields
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

            # Update state information under lock
            with self._ego_state_lock:
                # TODO: replace UNKNWON_DEFAULT_VAL with actual implementation
                self._ego_state = EgoState(0, timestamp, x, y, z, yaw, size, confidence, v_x, v_y,
                                           self.UNKNWON_DEFAULT_VAL,
                                           self.UNKNWON_DEFAULT_VAL, self.UNKNWON_DEFAULT_VAL, road_localization)

            self._publish_state_if_full()
        except Exception as e:
            self.logger.error("StateModule._self_localization_callback failed due to {}".format(e))

    # TODO: handle invalid data
    def _occupancy_state_callback(self, occupancy: dict) -> None:
        """
        De-serialize occupancy message and update state information.
        :param occupancy: Serialized occupancy state message.
        """
        try:
            self.logger.debug("got occupancy status %s", occupancy)
            timestamp = occupancy["timestamp"]

            # de-serialize relevant information
            free_space_points = np.array(occupancy["free_space_points"], dtype=float)
            points_list = free_space_points[:, :3]
            confidence_list = free_space_points[:, 3]

            # Update state information under lock
            with self._occupancy_state_lock:
                self._occupancy_state = OccupancyState(timestamp, np.array(points_list), np.array(confidence_list))

            self._publish_state_if_full()
        except Exception as e:
            self.logger.error("StateModule._occupancy_state_callback failed due to {}".format(e))

    def _publish_state_if_full(self) -> None:
        """
        Publish the currently updated state information.
        NOTE: Different state objects are updated at different times which means that a given output state
              message is not intrinsically synchronized. This is NOT handled here as each object arrives
              at the clients with its original time stamp and the required compensation (using prediction)
              is performed remotely.
        """
        # if some part of the state is missing, don't publish state message
        if self._occupancy_state is None or self._dynamic_objects is None or self._ego_state is None:
            return

        # Update state under lock (so that new events will not corrupt the data)
        with self._occupancy_state_lock, self._ego_state_lock, self._dynamic_objects_lock:
            state = State(self._occupancy_state, self._dynamic_objects, self._ego_state)
        self.logger.debug("publishing state %s", state.serialize())

        self.dds.publish(STATE_PUBLISH_TOPIC, state.serialize())

    # TODO: solve the fact that actuator status can be outdated and no one will ever know
    def _actuator_status_callback(self, actuator: dict) -> None:
        """
        Placeholder for future utilization of actuator information (e.g., steering and gas pedal).
        :param actuator: Serialized actuator message.
        """
        self.logger.debug("got actuator status %s", actuator)
        pass  # TODO: update self._ego_state.steering_angle. Don't forget to lock self._ego_state!
