import numpy as np

from decision_making.src.global_constants import EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT
from decision_making.src.scene.scene_message import SceneMessage
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import EgoState, ObjectSize, DynamicObject, State, OccupancyState


class SceneUtils:
    @staticmethod
    def get_state_from_scene(scene: SceneMessage):

        # TODO: This logic assumes the same frenet curve for all objects (based on ego's lane)
        road_id = scene.host_localization.lane_segment_id
        obj_id = 0
        timestamp = DynamicObject.sec_to_ticks(scene.timestamp_sec)
        cartesian_state = np.array([scene.host_localization.s_cartesian_localization.east_x,
                                    scene.host_localization.s_cartesian_localization.north_y,
                                    scene.host_localization.s_cartesian_localization.heading,
                                    scene.host_localization.s_cartesian_localization.velocity_longitudinal,
                                    scene.host_localization.s_cartesian_localization.acceleration_longitudinal,
                                    scene.host_localization.s_cartesian_localization.curvature])
        road_fstate = np.array([scene.host_localization.s_lane_frenet_coordinate.s,
                                scene.host_localization.s_lane_frenet_coordinate.s_dot,
                                scene.host_localization.s_lane_frenet_coordinate.s_dotdot,
                                scene.host_localization.s_lane_frenet_coordinate.d,
                                scene.host_localization.s_lane_frenet_coordinate.d_dot,
                                scene.host_localization.s_lane_frenet_coordinate.d_dotdot])
        size = ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT)
        confidence = 1.0

        ego_state = EgoState(obj_id, timestamp, cartesian_state, MapState(road_fstate, road_id), size, confidence)

        dynamic_objects = []
        for object_loc in scene.object_localizations:
            cartesian_state = np.array([object_loc.s_cartesian_localization.east_x,
                                        object_loc.s_cartesian_localization.north_y,
                                        object_loc.s_cartesian_localization.heading,
                                        object_loc.s_cartesian_localization.velocity_longitudinal,
                                        object_loc.s_cartesian_localization.acceleration_longitudinal,
                                        object_loc.s_cartesian_localization.curvature])
            road_fstate = np.array([object_loc.s_lane_frenet_coordinate.s,
                                    object_loc.s_lane_frenet_coordinate.s_dot,
                                    object_loc.s_lane_frenet_coordinate.s_dotdot,
                                    object_loc.s_lane_frenet_coordinate.d,
                                    object_loc.s_lane_frenet_coordinate.d_dot,
                                    object_loc.s_lane_frenet_coordinate.d_dotdot])
            size = size
            confidence = 1.0
            dynamic_objects.append(DynamicObject(object_loc.object_id, timestamp, cartesian_state,
                                                 MapState(road_fstate, road_id), size, confidence))

        return State(occupancy_state=OccupancyState(0, np.array([0]), np.array([0])),
                     dynamic_objects=dynamic_objects, ego_state=ego_state)
