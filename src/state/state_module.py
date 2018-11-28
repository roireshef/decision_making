from logging import Logger
from threading import Lock
from traceback import format_exc
from typing import Optional, Any, Dict

import numpy as np

import rte.python.profiler as prof
from common_data.interface.py.idl_generated_files.Rte_Types.TsSYS_SceneDynamic import TsSYSSceneDynamic
from common_data.interface.py.pubsub.Rte_Types_pubsub_topics import SCENE_DYNAMIC
from common_data.lcm.config import pubsub_topics
from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.global_constants import EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT, LOG_MSG_STATE_MODULE_PUBLISH_STATE
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.scene_dynamic_message import SceneDynamic
from decision_making.src.state.state import OccupancyState, ObjectSize, State, \
    DynamicObject, EgoState


class StateModule(DmModule):

    # TODO: implement double-buffer mechanism for locks wherever needed. Current lock mechanism may slow the
    # TODO(cont): processing when multiple events come in concurrently.
    def __init__(self, pubsub: PubSub, logger: Logger, scene_dynamic: Optional[SceneDynamic],
                 dynamic_objects_memory_map: Dict[int, DynamicObject] = {}) -> None:
        """
        :param pubsub: Inter-process communication interface
        :param logger: Logging module
        :param scene_dynamic:
        :param dynamic_objects_memory_map: Initial memory dict.
        """
        super().__init__(pubsub, logger)
        # save initial state and generate type-specific locks
        self._scene_dynamic = scene_dynamic
        self._scene_dynamic_lock = Lock()

        self._dynamic_objects_memory_map = dynamic_objects_memory_map

    def _start_impl(self) -> None:
        """
        When starting the State Module, subscribe to dynamic objects, ego state and occupancy state services.
        """
        self.pubsub.subscribe(SCENE_DYNAMIC, self._scene_dynamic_callback)

    # TODO - implement unsubscribe only when logic is fixed in LCM
    def _stop_impl(self) -> None:
        """
        Unsubscribe from process communication services.
        """
        pass

    def _periodic_action_impl(self) -> None:
        pass

    @prof.ProfileFunction()
    def _scene_dynamic_callback(self, scene_dynamic: TsSYSSceneDynamic, args: Any):
        try:
            with self._scene_dynamic_lock:
                self._scene_dynamic = SceneDynamic.deserialize(scene_dynamic)
                timestamp = DynamicObject.sec_to_ticks(self._scene_dynamic.s_Data.s_ComputeTimestamp.timestamp_in_seconds)
                occupancy_state = OccupancyState(0, np.array([0]), np.array([0]))
                ego_state = EgoState(obj_id=0,
                                     timestamp=timestamp,
                                     cartesian_state=self._scene_dynamic.s_Data.s_host_localization.a_cartesian_pose,
                                     map_state=self._scene_dynamic.s_Data.s_host_localization.a_lane_frenet_pose,
                                     map_state_on_host_lane=self._scene_dynamic.s_Data.s_host_localization.a_lane_frenet_pose,
                                     size=ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT),
                                     confidence=1.0)

                # TODO: handle multiple hypotheses
                dynamic_objects = []
                for obj_loc in self._scene_dynamic.s_Data.as_object_localization:
                    dyn_obj = DynamicObject(obj_id=obj_loc.e_Cnt_object_id,
                                            timestamp=timestamp,
                                            cartesian_state=obj_loc.as_object_hypothesis[0].a_cartesian_pose,
                                            map_state=obj_loc.as_object_hypothesis[0].a_lane_frenet_pose,
                                            map_state_on_host_lane=obj_loc.as_object_hypothesis[0].a_host_lane_frenet_pose,
                                            size=ObjectSize(obj_loc.s_bounding_box.e_l_length,
                                                            obj_loc.s_bounding_box.e_l_width,
                                                            obj_loc.s_bounding_box.e_l_height),
                                            confidence=obj_loc.as_object_hypothesis[0].e_r_probability)
                    dynamic_objects.append(dyn_obj)

                state = State(occupancy_state, dynamic_objects, ego_state)
                self.logger.debug("%s %s", LOG_MSG_STATE_MODULE_PUBLISH_STATE, state)

                self.pubsub.publish(pubsub_topics.STATE_TOPIC, state.serialize())

        except Exception as e:
            self.logger.error("StateModule._scene_dynamic_callback failed due to %s", format_exc())

            

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