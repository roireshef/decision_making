from logging import Logger
from threading import Lock
from traceback import format_exc
from typing import Optional, Any, List

import numpy as np
import rte.python.profiler as prof
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_SceneDynamic import TsSYSSceneDynamic
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_SCENE_DYNAMIC
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_STATE
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_TRAJECTORY_PARAMS

from decision_making.src.global_constants import EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT, LOG_MSG_STATE_MODULE_PUBLISH_STATE
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.messages.scene_dynamic_message import SceneDynamic, ObjectLocalization
from decision_making.src.planning.types import C_V, FS_SV, C_A, FS_SA
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import OccupancyState, ObjectSize, State, \
    DynamicObject, EgoState
from decision_making.src.messages.trajectory_parameters import TrajectoryParams


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
        self.pubsub.subscribe(UC_SYSTEM_SCENE_DYNAMIC, self._scene_dynamic_callback)
        self.pubsub.subscribe(UC_SYSTEM_TRAJECTORY_PARAMS)

    # TODO - implement unsubscribe only when logic is fixed in LCM
    def _stop_impl(self) -> None:
        """
        Unsubscribe from process communication services.
        """
        self.pubsub.unsubscribe(UC_SYSTEM_SCENE_DYNAMIC)
        # TODO: why unsubscribing the following breaks the unit test?
        # self.pubsub.unsubscribe(UC_SYSTEM_TRAJECTORY_PARAMS)

    def _periodic_action_impl(self) -> None:
        pass

    @prof.ProfileFunction()
    def _scene_dynamic_callback(self, scene_dynamic: TsSYSSceneDynamic, args: Any):
        try:
            with self._scene_dynamic_lock:

                self._scene_dynamic = SceneDynamic.deserialize(scene_dynamic)

                last_gff_segment_ids = self._get_last_action_gff()

                state = self.create_state_from_scene_dynamic(self._scene_dynamic, last_gff_segment_ids, self.logger)

                postprocessed_state = self._handle_negative_velocities(state)

                self.logger.debug("%s %s", LOG_MSG_STATE_MODULE_PUBLISH_STATE, postprocessed_state)

                self.pubsub.publish(UC_SYSTEM_STATE, postprocessed_state.serialize())

        except Exception:
            self.logger.error("StateModule._scene_dynamic_callback failed due to %s", format_exc())

    def _handle_negative_velocities(self, state: State) -> State:
        """
        Handles cases of ego state or dynamic objects with negative velocities.
        It modifies their velocities so they will equal zero and writes a warning in the log.
        :param state: Possibly containing objects with negative velocities
        :return: State containing objects with non-negative velocities
        """
        if state.ego_state.cartesian_state[C_V] < 0:
            state.ego_state.cartesian_state[C_V] = 0
            state.ego_state.map_state.lane_fstate[FS_SV] = 0
            self.logger.warning('Ego was received with negative velocity %f' % state.ego_state.cartesian_state[C_V])
        elif state.ego_state.cartesian_state[C_V] == 0 and state.ego_state.cartesian_state[C_A] < 0:
            state.ego_state.cartesian_state[C_A] = 0
            state.ego_state.map_state.lane_fstate[FS_SA] = 0
            self.logger.warning('Ego was received with zero velocity and negative acceleration %f' % state.ego_state.cartesian_state[C_A])

        for i in range(len(state.dynamic_objects)):
            if state.dynamic_objects[i].cartesian_state[C_V] < 0:
                state.dynamic_objects[i].cartesian_state[C_V] = 0
                state.dynamic_objects[i].map_state.lane_fstate[FS_SV] = 0
                self.logger.warning(
                    'Dynamic object with obj_id %s was received with negative velocity %f',
                    state.dynamic_objects[i].obj_id, state.dynamic_objects[i].cartesian_state[C_V])

        return state

    @staticmethod
    def create_state_from_scene_dynamic(scene_dynamic: SceneDynamic, last_gff_segment_ids: np.ndarray, logger: Logger) -> State:
        """
        This methods takes an already deserialized SceneDynamic message and converts it to a State object
        :param scene_dynamic:
        :param last_gff_segment_ids: list of GFF segment ids for the last selected action
        :param logger: Logging module
        :return: valid State object
        """

        timestamp = DynamicObject.sec_to_ticks(scene_dynamic.s_Data.s_RecvTimestamp.timestamp_in_seconds)
        occupancy_state = OccupancyState(0, np.array([0]), np.array([0]))

        selected_host_hyp_idx = 0

        host_hyp_lane_ids = [hyp.e_i_lane_segment_id
                             for hyp in scene_dynamic.s_Data.s_host_localization.as_host_hypothesis]

        if len(host_hyp_lane_ids) > 1 and len(last_gff_segment_ids) > 0:

            # find all common lae ids in host hypotheses and last gff segments
            common_lane_ids = np.intersect1d(host_hyp_lane_ids, last_gff_segment_ids)

            if len(common_lane_ids) == 1:
                selected_host_hyp_idx = np.argwhere(host_hyp_lane_ids == common_lane_ids[0])[0][0]
            elif len(common_lane_ids) > 1:
                # take the hyp. whose lane has the least distance from the host, i.e.,
                # the min. index in host_hyp_lane_ids since it is sorted based on the distance
                selected_host_hyp_idx = min([np.argwhere(host_hyp_lane_ids == common_lane_ids[i])[0][0]
                                             for i in range(len(common_lane_ids))])
            else:
                # there are no common ids between localization and prev. gff
                # raise a warning and choose the closet lane
                logger.warning("None of the host localization hypotheses matches the previous planning action")
                selected_host_hyp_idx = 0

        ego_map_state = MapState(lane_fstate=scene_dynamic.s_Data.s_host_localization.
                                 as_host_hypothesis[selected_host_hyp_idx].a_lane_frenet_pose,
                                 lane_id=scene_dynamic.s_Data.s_host_localization.
                                 as_host_hypothesis[selected_host_hyp_idx].e_i_lane_segment_id)
        ego_state = EgoState(obj_id=0,
                             timestamp=timestamp,
                             cartesian_state=scene_dynamic.s_Data.s_host_localization.a_cartesian_pose,
                             map_state=ego_map_state,
                             size=ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT),
                             confidence=1.0, off_map=False)

        dyn_obj_data = DynamicObjectsData(num_objects=scene_dynamic.s_Data.e_Cnt_num_objects,
                                          objects_localization=scene_dynamic.s_Data.as_object_localization,
                                          timestamp=timestamp)
        dynamic_objects = StateModule.create_dyn_obj_list(dyn_obj_data)
        return State(False, occupancy_state, dynamic_objects, ego_state)

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
            cartesian_state = obj_loc.a_cartesian_pose
            # TODO: Handle multiple hypotheses
            map_state = MapState(obj_loc.as_object_hypothesis[0].a_lane_frenet_pose, obj_loc.as_object_hypothesis[0].e_i_lane_segment_id)
            size = ObjectSize(obj_loc.s_bounding_box.e_l_length,
                              obj_loc.s_bounding_box.e_l_width,
                              obj_loc.s_bounding_box.e_l_height)
            confidence = obj_loc.as_object_hypothesis[0].e_r_probability
            off_map = obj_loc.as_object_hypothesis[0].e_b_off_lane
            dyn_obj = DynamicObject(obj_id=id,
                                    timestamp=timestamp,
                                    cartesian_state=cartesian_state,
                                    map_state=map_state if map_state.lane_id > 0 else None,
                                    size=size,
                                    confidence=confidence,
                                    off_map=off_map)

            objects_list.append(dyn_obj)  # update the list of dynamic objects

        return objects_list

    def _get_last_action_gff(self) -> np.ndarray:
        """
        Returns the array of GFF lane segments from the last action sent to trajectory planner.
        If no valid action is available, returns an empty array
        :return: array of GFF lane segments
        """
        last_gff_segment_ids = np.array([])

        is_success, serialized_params = self.pubsub.get_latest_sample(topic=UC_SYSTEM_TRAJECTORY_PARAMS)
        # TODO: Do we need a timeout here in the case trajectory parameters is empty for a while
        if serialized_params is not None:
            trajectory_params = TrajectoryParams.deserialize(serialized_params)
            last_gff_segment_ids = trajectory_params.reference_route.segment_ids

        return last_gff_segment_ids

