from decision_making.src.planning.types import C_X, C_Y, CartesianExtendedState, FS_DX, FS_SX
from decision_making.src.state.map_state import MapState
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.service.map_service import MapService


class MapUtils:
    # TODO: replace with navigation plan aware function from map API
    @staticmethod
    def get_road_rhs_frenet(obj):
        return MapService.get_instance()._rhs_roads_frenet[obj.map_state.road_id]

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

    # TODO: Note! This function is only valid when the frenet reference frame is from the right side of the road
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
        is_on_road_lat = road_width + ROAD_SHOULDERS_WIDTH > map_state.road_fstate[FS_DX] > -ROAD_SHOULDERS_WIDTH
        is_on_road_lon = map_state.road_fstate[FS_SX] < 995
        return is_on_road_lat and is_on_road_lon
