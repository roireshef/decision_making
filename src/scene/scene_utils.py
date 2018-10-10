from typing import List

import numpy as np

from decision_making.src.global_constants import EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT
from decision_making.src.planning.types import CartesianExtendedState
from decision_making.src.scene.scene_message import SceneMessage
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import EgoState, ObjectSize, DynamicObject, State, OccupancyState


class SceneUtils:

    @staticmethod
    def get_state_from_scene(scene: SceneMessage) -> State:

        # TODO: This logic assumes :
        # TODO: 1.The same frenet curve for all objects (based on ego's lane)
        # TODO: 2.There is only one hypothesis for an object's location
        road_id = scene.host_localization.lane_segment_id
        obj_id = 0
        timestamp = DynamicObject.sec_to_ticks(scene.timestamp_sec)
        cartesian_state = scene.host_localization.s_cartesian_localization.numpy()
        road_fstate = scene.host_localization.s_lane_frenet_coordinate.numpy()
        size = ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT)
        confidence = 1.0

        ego_state = EgoState(obj_id, timestamp, cartesian_state, MapState(road_fstate, road_id), size, confidence)

        dynamic_objects = []
        for object_loc in scene.object_localizations:
            cartesian_state = object_loc.s_object_hypotheses[0].s_cartesian_localization.numpy()
            road_fstate = object_loc.s_object_hypotheses[0].s_lane_frenet_coordinate.numpy()
            size = size
            confidence = 1.0
            dynamic_objects.append(DynamicObject(object_loc.object_id, timestamp, cartesian_state,
                                                 MapState(road_fstate, road_id), size, confidence))

        return State(occupancy_state=OccupancyState(0, np.array([0]), np.array([0])),
                     dynamic_objects=dynamic_objects, ego_state=ego_state)

    @staticmethod
    def get_cstate_from_scene(scene: SceneMessage, obj_id: int) -> CartesianExtendedState:

        # TODO: This logic assumes there is only one hypothesis for an object's location
        if obj_id == 0:
            return scene.host_localization.s_cartesian_localization.numpy()
        else:

            desired_object_loc = [object_loc for object_loc in scene.object_localizations
                                  if object_loc.object_id == obj_id]
            assert len(desired_object_loc) == 1
            return desired_object_loc[0].s_object_hypotheses[0].s_cartesian_localization.numpy()

    @staticmethod
    def get_mapstate_from_scene(scene: SceneMessage, obj_id: int) -> MapState:

        if obj_id == 0:
            fstate = scene.host_localization.s_lane_frenet_coordinate.numpy()
            return MapState(fstate, scene.host_localization.lane_segment_id)
        else:

            desired_object_loc = [object_loc for object_loc in scene.object_localizations
                                  if object_loc.object_id == obj_id]
            assert len(desired_object_loc) == 1
            fstate = desired_object_loc[0].s_object_hypotheses[0].s_lane_frenet_coordinate.numpy()
            return MapState(fstate, desired_object_loc[0].s_lane_frenet_coordinate.lane_segment_id)

    @staticmethod
    def get_cartesian_point_lane_index(x: float, y: float) -> int:
        """
        returns the lane id containing the cartesian point [x,y]
        :param x: [float]
        :param y: [float]
        :return: lane id [int]
        """
        # TODO: replace later
        return 0

    @staticmethod
    def get_lane_num(lane_id: int) -> int:
        """
        returns the lane number based on the lane id.
        :param lane_id: int
        :return: lane number [int]
        """
        # TODO: replace later
        return 0

    @staticmethod
    def get_center_lanes_latitudes(scene: SceneMessage) -> List[float]:
        return [1.0, 2.0, 3.0]

    @staticmethod
    def get_num_lanes(scene: SceneMessage) -> int:
        return 3
