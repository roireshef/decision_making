import numpy as np
import rte.python.profiler as prof
from common_data.interface.Rte_Types.python import Rte_Types_pubsub as pubsub_topics
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_SceneDynamic import TsSYSSceneDynamic
from decision_making.src.exceptions import ObjectHasNegativeVelocityError
from decision_making.src.global_constants import EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT, LOG_MSG_STATE_MODULE_PUBLISH_STATE, \
    VELOCITY_MINIMAL_THRESHOLD
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.messages.scene_dynamic_message import SceneDynamic, ObjectLocalization
from decision_making.src.planning.types import FS_SV, C_V
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import OccupancyState, ObjectSize, State, \
    DynamicObject, EgoState
from logging import Logger
from threading import Lock
from traceback import format_exc
from typing import Optional, Any, List


class DynamicObjectsData:
    def __init__(self, num_objects: int, objects_localization: List[ObjectLocalization], timestamp: int):
        self.num_objects = num_objects
        self.objects_localization = objects_localization
        self.timestamp = timestamp


class StateModule(DmModule):

    # TODO: implement double-buffer mechanism for locks wherever needed. Current lock mechanism may slow the
    # TODO(cont): processing when multiple events come in concurrently.
    def __init__(self, pubsub: PubSub, logger: Logger, scene_dynamic: Optional[SceneDynamic]) -> None:
        """
        :param pubsub: Inter-process communication interface
        :param logger: Logging module
        :param scene_dynamic:
        """
        super().__init__(pubsub, logger)
        # save initial state and generate type-specific locks
        self._scene_dynamic = scene_dynamic
        self._scene_dynamic_lock = Lock()

    def _start_impl(self) -> None:
        """
        When starting the State Module, subscribe to dynamic objects, ego state and occupancy state services.
        """
        self.pubsub.subscribe(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_SCENE_DYNAMIC"], self._scene_dynamic_callback)

    # TODO - implement unsubscribe only when logic is fixed in LCM
    def _stop_impl(self) -> None:
        """
        Unsubscribe from process communication services.
        """
        pass

    def _periodic_action_impl(self) -> None:
        pass

    @staticmethod
    def create_dyn_obj_list(dyn_obj_data: DynamicObjectsData) -> List[DynamicObject]:
        """
        Convert serialized object perception and global localization data into a DM object (This also includes
        computation of the object's road localization). Additionally store the object in memory as preparation for
        the case where it will leave the field of view.
        :param dyn_obj_data:
        :return: List of dynamic object in DM format.
        """
        timestamp = dyn_obj_data.timestamp
        objects_list = []
        for obj_idx in range(dyn_obj_data.num_objects):
            obj_loc = dyn_obj_data.objects_localization[obj_idx]
            id = obj_loc.e_Cnt_object_id
            # TODO: Handle multiple hypotheses
            cartesian_state = obj_loc.as_object_hypothesis[0].a_cartesian_pose
            map_state = MapState(obj_loc.as_object_hypothesis[0].a_lane_frenet_pose, obj_loc.as_object_hypothesis[0].e_i_lane_segment_id)
            # TODO: map_state_on_host_lane now unused, see if it makes more sense to send ego lane_id in its map_state
            map_state_on_host_lane = MapState(obj_loc.as_object_hypothesis[0].a_host_lane_frenet_pose, obj_loc.as_object_hypothesis[0].e_i_lane_segment_id)
            size = ObjectSize(obj_loc.s_bounding_box.e_l_length,
                              obj_loc.s_bounding_box.e_l_width,
                              obj_loc.s_bounding_box.e_l_height)
            confidence = obj_loc.as_object_hypothesis[0].e_r_probability

            dyn_obj = DynamicObject(obj_id=obj_loc.e_Cnt_object_id,
                                    timestamp=timestamp,
                                    cartesian_state=cartesian_state,
                                    map_state=map_state if map_state.lane_id > 0 else None,
                                    map_state_on_host_lane=map_state_on_host_lane if map_state_on_host_lane.lane_id > 0 else None,
                                    size=size,
                                    confidence=confidence)

            # TODO: Handle negative velocities properly
            if dyn_obj.cartesian_state[C_V] < 0:
                raise ObjectHasNegativeVelocityError('Dynamic object with id %d was received with negative velocity %f'
                                                     % (dyn_obj.obj_id, dyn_obj.cartesian_state[C_V]))

            # TODO: Figure out if we need SceneProvider to let us know if an object is not on road
            # Required to verify the object has map state and that the velocity exceeds a minimal value.
            if dyn_obj.map_state.lane_fstate[FS_SV] < VELOCITY_MINIMAL_THRESHOLD:
                thresholded_lane_fstate = np.copy(dyn_obj.map_state.lane_fstate)
                thresholded_lane_fstate[FS_SV] = VELOCITY_MINIMAL_THRESHOLD
                dyn_obj = dyn_obj.clone_from_map_state(
                    map_state=MapState(lane_fstate=thresholded_lane_fstate,
                                       lane_id=dyn_obj.map_state.lane_id))

            objects_list.append(dyn_obj)  # update the list of dynamic objects

        return objects_list

    @staticmethod
    def create_state_from_scene_dynamic(scene_dynamic: SceneDynamic) -> State:
        """
        This methods takes an already deserialized SceneDynamic message and converts it to a State object
        :param scene_dynamic:
        :return: valid State object
        """

        timestamp = DynamicObject.sec_to_ticks(scene_dynamic.s_Data.s_RecvTimestamp.timestamp_in_seconds)
        occupancy_state = OccupancyState(0, np.array([0]), np.array([0]))
        ego_map_state = MapState(lane_fstate=scene_dynamic.s_Data.s_host_localization.a_lane_frenet_pose,
                                 lane_id=scene_dynamic.s_Data.s_host_localization.e_i_lane_segment_id)
        ego_state = EgoState(obj_id=0,
                             timestamp=timestamp,
                             cartesian_state=scene_dynamic.s_Data.s_host_localization.a_cartesian_pose,
                             map_state=ego_map_state,
                             map_state_on_host_lane=ego_map_state,
                             size=ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT),
                             confidence=1.0)

        if ego_state.cartesian_state[C_V] < 0:
            raise ObjectHasNegativeVelocityError(
                'Ego was received with negative velocity %f' % ego_state.cartesian_state[C_V])

        dyn_obj_data = DynamicObjectsData(num_objects=scene_dynamic.s_Data.e_Cnt_num_objects,
                                          objects_localization=scene_dynamic.s_Data.as_object_localization,
                                          timestamp=timestamp)
        dynamic_objects = StateModule.create_dyn_obj_list(dyn_obj_data)
        return State(occupancy_state, dynamic_objects, ego_state)

    @prof.ProfileFunction()
    def _scene_dynamic_callback(self, scene_dynamic: TsSYSSceneDynamic, args: Any):
        try:
            with self._scene_dynamic_lock:

                self._scene_dynamic = SceneDynamic.deserialize(scene_dynamic)

                state = self.create_state_from_scene_dynamic(self._scene_dynamic)
                
                self.logger.debug("%s %s", LOG_MSG_STATE_MODULE_PUBLISH_STATE, state)

                self.pubsub.publish(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_STATE_LCM"], state.serialize())

        except ObjectHasNegativeVelocityError as e:
            self.logger.error(e)

        except Exception:
            self.logger.error("StateModule._scene_dynamic_callback failed due to %s", format_exc())
