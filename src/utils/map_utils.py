import numpy as np
from decision_making.src.planning.types import FP_SX, FrenetState2D, FS_SX, C_X, C_Y, CartesianExtendedState
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import DynamicObject, EgoState, NewDynamicObject, NewEgoState
from mapping.src.service.map_service import MapService


class MapUtils:
    # TODO: replace with navigation plan aware function from map API
    @staticmethod
    def get_road_rhs_frenet(obj: NewDynamicObject):
        return MapService.get_instance()._rhs_roads_frenet[obj.map_state.road_id]

    # TODO: replace this call with the road localization once it is updated to be hold a frenet state
    @staticmethod
    def get_object_road_localization(obj: NewDynamicObject, road_frenet: FrenetSerret2DFrame) -> FrenetState2D:
        return obj.map_state.road_fstate

    @staticmethod
    def get_ego_road_localization(ego: NewEgoState, road_frenet: FrenetSerret2DFrame):
        return ego.map_state.road_fstate

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