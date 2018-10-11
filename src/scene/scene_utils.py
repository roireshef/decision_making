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
    def get_cartesian_point_lane_index(x: float, y: float, map_object: MapObject) -> int:
        """
        returns the lane id containing the cartesian point [x,y]
        :param x: [float]
        :param y: [float]
        :param map_object: [MapObject]
        :return: lane id [int]
        """
        # TODO: replace later
        return 0

    @staticmethod
    def get_lane_num(lane_id: int, map_object: MapObject) -> int:
        """
        returns the lane number based on the lane id.
        :param lane_id: int
        :param map_object: [MapObject]
        :return: lane number [int]
        """
        # TODO: replace later
        return 0

    @staticmethod
    def get_center_lanes_latitudes(map_object: MapObject) -> List[float]:
        """
        :param map_object: [MapObject]
        :return:
        """
        return [1.0, 2.0, 3.0]

    @staticmethod
    def get_num_lanes(map_object: MapObject) -> int:
        """
        :param map_object:
        :return:
        """
        return 3

    @staticmethod
    def get_uniform_path_lookahead(road_id, lat_shift, starting_lon, lon_step, steps_num, navigation_plan):
        # type: (int, float, float, float, int, NavigationPlanMsg) -> np.ndarray
        """
        Create array of uniformly distributed points along a given road, shifted laterally by by lat_shift.
        When some road finishes, it automatically continues to the next road, according to the navigation plan.
        The distance between consecutive points is lon_step.
        :param road_id: starting road_id
        :param lat_shift: lateral shift from right side of the road [m]
        :param starting_lon: starting longitude [m]
        :param lon_step: distance between consecutive points [m]
        :param steps_num: output points number
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: uniform sampled points array (Nx2)
        """
        return np.ndarray([0])