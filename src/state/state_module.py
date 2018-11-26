from logging import Logger
from threading import Lock
from traceback import format_exc
from typing import Optional, Any

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