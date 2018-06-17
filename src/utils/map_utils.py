import numpy as np
from decision_making.src.planning.types import FP_SX, FrenetState2D, FS_SX, C_X, C_Y, CartesianExtendedState, FS_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import DynamicObject, EgoState
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.service.map_service import MapService


class MapUtils:
    # TODO: replace with navigation plan aware function from map API
    @staticmethod
    def get_road_rhs_frenet(obj: DynamicObject):
        return MapService.get_instance()._rhs_roads_frenet[obj.road_localization.road_id]

    # TODO: replace this call with the road localization once it is updated to be hold a frenet state
    @staticmethod
    def get_object_road_localization(obj: DynamicObject, road_frenet: FrenetSerret2DFrame) -> FrenetState2D:
        target_obj_fpoint = road_frenet.cpoint_to_fpoint(np.array([obj.x, obj.y]))
        obj_init_fstate = road_frenet.cstate_to_fstate(np.array([
            obj.x, obj.y,
            obj.yaw,
            obj.total_speed,
            obj.acceleration_lon,
            road_frenet.get_curvature(target_obj_fpoint[FP_SX])  # We zero the object's curvature relative to the road
        ]))
        return obj_init_fstate

    @staticmethod
    def get_ego_road_localization(ego: EgoState, road_frenet: FrenetSerret2DFrame):
        ego_init_cstate = np.array([ego.x, ego.y, ego.yaw, ego.v_x, ego.acceleration_lon, ego.curvature])
        return road_frenet.cstate_to_fstate(ego_init_cstate)

    @staticmethod
    def convert_cartesian_to_map_state(cartesian_state: CartesianExtendedState):
        # type: (CartesianExtendedState) -> MapState
        # TODO: replace with query that returns only the relevant road id
        map_api = MapService.get_instance()

        relevant_road_ids = map_api._find_roads_containing_point(cartesian_state[C_X], cartesian_state[C_Y])
        closest_road_id = map_api._find_closest_road(cartesian_state[C_X], cartesian_state[C_Y], relevant_road_ids)

        road_frenet = map_api._rhs_roads_frenet[closest_road_id]

        obj_fstate = road_frenet.cstate_to_fstate(cartesian_state)

        return MapState(obj_fstate, closest_road_id)

    @staticmethod
    def convert_map_to_cartesian_state(map_state):
        # type: (MapState) -> CartesianExtendedState
        map_api = MapService.get_instance()

        road_frenet = map_api._rhs_roads_frenet[map_state.road_id]

        return road_frenet.fstate_to_cstate(map_state.road_fstate)

    #TODO: Note! This function is only valid when the frenet reference frame is from the right side of the road
    @staticmethod
    def is_object_on_road(map_state):
        # type: (MapState) -> bool
        """
        Returns true of the object is on the road. False otherwise.
        Note! This function is valid only when the frenet reference frame is from the right side of the road
        :param map_state: the map state to check
        :return: Returns true of the object is on the road. False otherwise.
        """
        road_width = MapService.get_instance().get_road(road_id=map_state.road_id).road_width
        is_on_road = road_width + ROAD_SHOULDERS_WIDTH > map_state.road_fstate[FS_DX] > -ROAD_SHOULDERS_WIDTH
        return is_on_road
