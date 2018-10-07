from logging import Logger
from threading import Lock
from traceback import format_exc
from typing import Optional, List, Dict

import numpy as np

import rte.python.profiler as prof
from common_data.lcm.config import pubsub_topics
from common_data.src.communication.middleware.idl_generated_files import LcmPerceivedDynamicObjectList
from common_data.src.communication.middleware.idl_generated_files import LcmPerceivedSelfLocalization
from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.global_constants import DEFAULT_OBJECT_Z_VALUE, EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT, EGO_ID, \
    UNKNOWN_DEFAULT_VAL, FILTER_OFF_ROAD_OBJECTS, LOG_MSG_STATE_MODULE_PUBLISH_STATE, VELOCITY_MINIMAL_THRESHOLD
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.planning.types import FS_SV
from decision_making.src.planning.utils.transformations import Transformations
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import OccupancyState, ObjectSize, State, \
    DynamicObject, EgoState
from decision_making.src.utils.map_utils import MapUtils
from mapping.src.exceptions import MapCellNotFound, raises


class StateModule(DmModule):

    # TODO: implement double-buffer mechanism for locks wherever needed. Current lock mechanism may slow the
    # TODO(cont): processing when multiple events come in concurrently.
    def __init__(self, pubsub: PubSub, logger: Logger, occupancy_state: Optional[OccupancyState],
                 dynamic_objects: Optional[List[DynamicObject]], ego_state: Optional[EgoState],
                 dynamic_objects_memory_map: Dict[int, DynamicObject] = {}) -> None:
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

    @prof.ProfileFunction()
    def _dynamic_obj_callback(self, objects: LcmPerceivedDynamicObjectList):
        """
        Deserialize dynamic objects message and replace pointer reference to dynamic object list under lock
        :param objects: Serialized dynamic object list
        """
        try:
            dyn_obj_list = self.create_dyn_obj_list(objects)

            with self._dynamic_objects_lock:
                self._dynamic_objects = dyn_obj_list

            self._publish_state_if_full()
        except Exception as e:
            self.logger.error("StateModule._dynamic_obj_callback failed due to %s", format_exc())

    @raises(MapCellNotFound)
    def create_dyn_obj_list(self, dyn_obj_list: LcmPerceivedDynamicObjectList) -> List[DynamicObject]:
        """
        Convert serialized object perception and global localization data into a DM object (This also includes computation
        of the object's road localization). Additionally store the object in memory as preparation for the case where it will leave
        the field of view.
        :param dyn_obj_list: Serialized dynamic objects list.
        :return: List of dynamic object in DM format.
        """

        timestamp = dyn_obj_list.timestamp
        lcm_dyn_obj_list = dyn_obj_list.dynamic_objects
        objects_list = []
        for obj_idx in range(dyn_obj_list.num_objects):
            lcm_dyn_obj = lcm_dyn_obj_list[obj_idx]
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

                # convert velocity from map coordinates to relative to its own yaw
                # TODO: ask perception to send v_x, v_y in dynamic vehicle's coordinate frame and not global frame.
                v_x = np.cos(yaw) * glob_v_x + np.sin(yaw) * glob_v_y
                v_y = -np.sin(yaw) * glob_v_x + np.cos(yaw) * glob_v_y

                total_v = np.linalg.norm([v_x, v_y])

                # TODO: currently acceleration_lon is 0 for dynamic_objects.
                # TODO: When it won't be zero, consider that the speed and acceleration should be in the same direction
                # TODO: The same for curvature.
                acceleration_lon = UNKNOWN_DEFAULT_VAL
                curvature = UNKNOWN_DEFAULT_VAL

                global_coordinates = np.array([x, y, z])
                # TODO: we might consider using velocity_yaw = np.arctan2(object_state.v_y, object_state.v_x)
                global_yaw = yaw

                try:
                    dyn_obj = DynamicObject.create_from_cartesian_state(
                        obj_id=id, timestamp=timestamp, size=size, confidence=confidence,
                        cartesian_state=np.array([x, y, global_yaw, total_v, acceleration_lon, curvature]))

                    # When filtering off-road objects, try to localize object on road.
                    if not FILTER_OFF_ROAD_OBJECTS or MapUtils.is_object_on_road(dyn_obj.map_state):

                        # Required to verify the object has map state and that the velocity exceeds a minimal value.
                        # If FILTER_OFF_ROAD_OBJECTS is true, it means that the object is on road - therfore has map
                        # state
                        if FILTER_OFF_ROAD_OBJECTS and dyn_obj.map_state.road_fstate[FS_SV] < VELOCITY_MINIMAL_THRESHOLD:
                            thresholded_road_fstate = np.copy(dyn_obj.map_state.road_fstate)
                            thresholded_road_fstate[FS_SV] = VELOCITY_MINIMAL_THRESHOLD
                            dyn_obj = dyn_obj.clone_from_map_state(
                                map_state=MapState(road_fstate=thresholded_road_fstate,
                                                   road_id=dyn_obj.map_state.road_id))

                        self._dynamic_objects_memory_map[id] = dyn_obj
                        objects_list.append(dyn_obj)  # update the list of dynamic objects
                    else:
                        continue

                except MapCellNotFound:
                    x, y, z = global_coordinates
                    self.logger.warning(
                        "Couldn't localize object on road. Object location: ({}, {}, {})".format(id, x, y, z))


            else:
                # object is out of FOV, using its last known location and timestamp.
                dyn_obj = self._dynamic_objects_memory_map.get(id)
                if dyn_obj is not None:
                    objects_list.append(dyn_obj)  # update the list of dynamic objects
                else:
                    self.logger.warning("received out of FOV object which is not in memory.")
        return objects_list

    @prof.ProfileFunction()
    def _self_localization_callback(self, self_localization: LcmPerceivedSelfLocalization) -> None:
        """
        Deserialize localization information, convert to road coordinates and update state information.
        :param self_localization: Serialized self localization message.
        """
        try:
            confidence = self_localization.location.confidence
            timestamp = self_localization.timestamp
            x = self_localization.location.x
            y = self_localization.location.y
            z = DEFAULT_OBJECT_Z_VALUE
            yaw = self_localization.yaw
            v_x = self_localization.velocity.v_x
            v_y = self_localization.velocity.v_y
            a_x = self_localization.acceleration.a_x
            curvature = self_localization.curvature
            size = ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT)

            total_v = np.linalg.norm([v_x, v_y])

            # Update state information under lock
            with self._ego_state_lock:
                self._ego_state = EgoState.create_from_cartesian_state(
                    obj_id=EGO_ID, timestamp=timestamp, size=size, confidence=confidence,
                    cartesian_state=np.array([x, y, yaw, total_v, a_x, curvature]))
                self._ego_state = Transformations.transform_ego_from_origin_to_center(self._ego_state)

            self._publish_state_if_full()

        except Exception as e:
            self.logger.exception('StateModule._self_localization_callback failed due to')

    @prof.ProfileFunction()
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
        self.logger.debug("%s %s", LOG_MSG_STATE_MODULE_PUBLISH_STATE, state)

        self.pubsub.publish(pubsub_topics.STATE_TOPIC, state.serialize())
