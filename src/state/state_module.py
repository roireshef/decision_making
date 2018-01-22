from logging import Logger
from threading import Lock
from typing import Optional, List, Dict

import numpy as np

from decision_making.src.global_constants import DEFAULT_OBJECT_Z_VALUE, EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT, EGO_ID, \
    UNKNOWN_DEFAULT_VAL
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.state.state import OccupancyState, EgoState, DynamicObject, ObjectSize, State
from mapping.src.exceptions import MapCellNotFound, raises
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH

from mapping.src.service.map_service import MapService

from common_data.src.communication.pubsub.pubsub import PubSub
from common_data.lcm.config import pubsub_topics
from common_data.lcm.generatedFiles.gm_lcm import LcmPerceivedDynamicObjectList
from common_data.lcm.generatedFiles.gm_lcm import LcmPerceivedSelfLocalization


class StateModule(DmModule):

    # TODO: implement double-buffer mechanism for locks wherever needed. Current lock mechanism may slow the
    # TODO(cont): processing when multiple events come in concurrently.
    def __init__(self, pubsub: PubSub, logger: Logger, occupancy_state: Optional[OccupancyState],
                 dynamic_objects: Optional[List[DynamicObject]], ego_state: Optional[EgoState],
                 dynamic_objects_memory_map: Dict[int,DynamicObject] = {}) ->  None:
        super().__init__(pubsub, logger)
        """
        :param dds: Inter-process communication interface.
        :param logger: Logging module.
        :param occupancy_state: Initial state occupancy object.
        :param dynamic_objects: Initial state dynamic objects.
        :param ego_state: Initial ego-state object.
        :param dynamic_objects_memory_map: Initial memory dict.
        """
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
        self.pubsub.subscribe(pubsub_topics.PERCEIVED_DYNAMIC_OBJECTS_TOPIC
                            , self._dynamic_obj_callback)
        self.pubsub.subscribe(pubsub_topics.PERCEIVED_SELF_LOCALIZATION_TOPIC
                            , self._self_localization_callback)


    # TODO - implement unsubscribe only when logic is fixed in LCM
    def _stop_impl(self) -> None:
        """
        Unsubscribe from process communication services.
        """

    def _periodic_action_impl(self) -> None:
        pass

    def _dynamic_obj_callback(self, objects: LcmPerceivedDynamicObjectList):
        """
        Deserialize dynamic objects message and replace pointer reference to dynamic object list under lock
        :param objects: Serialized dynamic object list
        """
        try:
            # TODO: think how to print perceived dynamic objects, since they are not our objects
            self.logger.info("got perceived dynamic objects {}".format(objects))

            dyn_obj_list = self.create_dyn_obj_list(objects)

            with self._dynamic_objects_lock:
                self._dynamic_objects = dyn_obj_list

            self._publish_state_if_full()
        except Exception as e:
            self.logger.error("StateModule._dynamic_obj_callback failed due to {}".format(e))

    @raises(MapCellNotFound)
    def create_dyn_obj_list(self, dyn_obj_list: LcmPerceivedDynamicObjectList) -> List[DynamicObject]:
        """
        Convert serialized object perception and global localization data into a DM object (This also includes computation
        of the object's road localization). Additionally store the object in memory as preparation for the case where it will leave
        the field of view.
        :param objects: Serialized dynamic objects list.
        :return: List of dynamic object in DM format.
        """

        timestamp = dyn_obj_list.timestamp
        lcm_dyn_obj_list = dyn_obj_list.dynamic_objects
        dyn_obj_list = []
        for lcm_dyn_obj in lcm_dyn_obj_list:
            ''' lcm_dyn_obj is an instance of LcmPerceivedDynamicObject class '''
            in_fov = lcm_dyn_obj.tracking_status.in_fov
            id = lcm_dyn_obj.id
            if in_fov:
                # object is in FOV, so we take its latest detection.
                x = lcm_dyn_obj.location.x
                y = lcm_dyn_obj.location.y
                z = DEFAULT_OBJECT_Z_VALUE  # lcm_dyn_obj.location.z ?
                confidence = lcm_dyn_obj.location.confidence
                yaw = lcm_dyn_obj.bbox.yaw
                length = lcm_dyn_obj.bbox.length
                width = lcm_dyn_obj.bbox.width
                height = lcm_dyn_obj.bbox.height
                size = ObjectSize(length, width, height)
                glob_v_x = lcm_dyn_obj.velocity.v_x
                glob_v_y = lcm_dyn_obj.velocity.v_y
                omega_yaw = lcm_dyn_obj.velocity.omega_yaw

                # convert velocity from map coordinates to relative to its own yaw
                v_x = np.cos(yaw) * glob_v_x + np.sin(yaw) * glob_v_y
                v_y = -np.sin(yaw) * glob_v_x + np.cos(yaw) * glob_v_y

                is_predicted = lcm_dyn_obj.tracking_status.is_predicted

                global_coordinates = np.array([x, y, z])
                global_yaw = yaw

                try:
                    # Try to localize object on road. If not successful, warn.
                    road_localization = MapService.get_instance().compute_road_localization(global_coordinates, global_yaw)

                    # Filter objects out of road:
                    road_width = MapService.get_instance().get_road(road_id=road_localization.road_id).road_width
                    if road_width + ROAD_SHOULDERS_WIDTH > road_localization.intra_road_lat > -ROAD_SHOULDERS_WIDTH:
                        dyn_obj = DynamicObject(id, timestamp, global_coordinates[0], global_coordinates[1],
                                                global_coordinates[2], global_yaw, size, confidence, v_x, v_y,
                                                UNKNOWN_DEFAULT_VAL, omega_yaw)
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

    def _self_localization_callback(self, self_localization: LcmPerceivedSelfLocalization) -> None:
        """
        Deserialize localization information, convert to road coordinates and update state information.
        :param self_localization: Serialized self localization message.
        """
        try:
            # TODO: think how to print perceived self localization, since it's not our object
            self.logger.debug("got perceived self localization {}".format(self_localization))
            confidence = self_localization.location.confidence
            timestamp = self_localization.timestamp
            x = self_localization.location.x
            y = self_localization.location.y
            z = DEFAULT_OBJECT_Z_VALUE
            yaw = self_localization.yaw
            v_x = self_localization.velocity.v_x
            v_y = self_localization.velocity.v_y
            a_x = self_localization.acceleration.a_x
            size = ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT)

            # Update state information under lock
            with self._ego_state_lock:
                # TODO: replace UNKNOWN_DEFAULT_VAL with actual implementation
                self._ego_state = EgoState(EGO_ID, timestamp, x, y, z, yaw, size, confidence, v_x, v_y, a_x,
                                           UNKNOWN_DEFAULT_VAL, UNKNOWN_DEFAULT_VAL)

            self._publish_state_if_full()
        except Exception as e:
            self.logger.error("StateModule._self_localization_callback failed due to {}".format(e))

    # TODO: convert from lcm...
    # TODO: handle invalid data - occupancy is currently unused throughout the system
    def _occupancy_state_callback(self, occupancy: dict) -> None:
        """
        De-serialize occupancy message and update state information.
        :param occupancy: Serialized occupancy state message.
        """
        try:
            # TODO: think how to print occupancy status, since it's not our object
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
        self.logger.debug("publishing state %s", state)

        self.pubsub.publish(pubsub_topics.STATE_TOPIC, state.serialize())

    # TODO: LCM?
    # TODO: solve the fact that actuator status can be outdated and no one will ever know
    def _actuator_status_callback(self, actuator: dict) -> None:
        """
        Placeholder for future utilization of actuator information (e.g., steering and gas pedal).
        :param actuator: Serialized actuator message.
        """
        self.logger.debug("got actuator status %s", actuator)
        pass  # TODO: update self._ego_state.steering_angle. Don't forget to lock self._ego_state!

