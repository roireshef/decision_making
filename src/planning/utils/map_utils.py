from decision_making.src.planning.types import CartesianExtendedState, C_X, C_Y
from decision_making.src.planning.types import FrenetState2D
from decision_making.src.state.state import MapState
from mapping.src.localization.localization import GlobalLocalization
from mapping.src.service.map_service import MapService


class MapUtils:
    @staticmethod
    def convert_cartesian_to_map_state(cartesian_state: CartesianExtendedState):
        # type: (CartesianExtendedState) -> MapState
        # TODO: replace with query that returns only the relevant road id
        map_api = MapService.get_instance()

        x, y = cartesian_state[C_X], cartesian_state[C_Y]
        global_localization = GlobalLocalization(x, y, 0, None)
        lane = map_api.resolve_location_on_global(global_localization)
        lane_fstate = lane.frame.center_frenet_frame.cstate_to_fstate(cartesian_state)

        segment = lane.segment
        road = lane.segment.road

        return MapState(lane_fstate, road.id, segment.id, lane.id)

    @staticmethod
    def convert_map_to_cartesian_state(map_state):
        # type: (MapState) -> FrenetState2D
        map_api = MapService.get_instance()

        lane = map_api.get_lane(map_state.lane_id)

        cartesian_state = lane.frame.frenet_to_cartesian(map_state.lane_state)

        return cartesian_state

    @staticmethod
    def convert_lane_to_road_localization(lane_state, road_id, segment_id, lane_id):
        # type: (FrenetState2D, float, float, float) -> FrenetState2D
        map_api = MapService.get_instance()

        road = map_api._map_model._map.get_road(road_id)
        segment = road.get_segment(segment_id)
        lane = segment.get_lane(lane_id)

        road_frenet = road.frame
        lane_frenet = lane.frame

        return road_frenet.cartesian_to_frenet(lane_frenet.frenet_to_cartesian(lane_state))

    @staticmethod
    def convert_road_to_lane_localization(lane_state, road_id, segment_id, lane_id):
        # type: (FrenetState2D, float, float, float) -> FrenetState2D
        map_api = MapService.get_instance()

        road = map_api._map_model._map.get_road(road_id)
        segment = road.get_segment(segment_id)
        lane = segment.get_lane(lane_id)

        road_frenet = road.frame.center_frenet_frame
        lane_frenet = lane.frame.center_frenet_frame

        return lane_frenet.cstate_to_fstate(road_frenet.fstate_to_cstate(lane_state))

    @staticmethod
    def convert_lane_to_segment_localization(lane_state, road_id, segment_id, lane_id):
        # type: (FrenetState2D, float, float, float) -> FrenetState2D
        map_api = MapService.get_instance()

        road = map_api._map_model._map.get_road(road_id)
        segment = road.get_segment(segment_id)
        lane = segment.get_lane(lane_id)

        segment_frenet = segment.frame.center_frenet_frame
        lane_frenet = lane.frame.center_frenet_frame

        return segment_frenet.cstate_to_fstate(lane_frenet.fstate_to_cstate(lane_state))

    @staticmethod
    def convert_segment_to_lane_localization(lane_state, road_id, segment_id, lane_id):
        # type: (FrenetState2D, float, float, float) -> FrenetState2D
        map_api = MapService.get_instance()

        road = map_api._map_model._map.get_road(road_id)
        segment = road.get_segment(segment_id)
        lane = segment.get_lane(lane_id)

        segment_frenet = segment.frame.center_frenet_frame
        lane_frenet = lane.frame.center_frenet_frame

        return lane_frenet.cstate_to_fstate(segment_frenet.fstate_to_cstate(lane_state))

    @staticmethod
    def convert_road_to_segment_localization(lane_state, road_id, segment_id):
        # type: (FrenetState2D, float, float, float) -> FrenetState2D
        map_api = MapService.get_instance()

        road = map_api._map_model._map.get_road(road_id)
        segment = road.get_segment(segment_id)

        road_frenet = road.frame.center_frenet_frame
        segment_frenet = segment.frame.center_frenet_frame

        return segment_frenet.cstate_to_fstate(road_frenet.fstate_to_cstate(lane_state))

    @staticmethod
    def convert_segment_to_road_localization(lane_state, road_id, segment_id):
        # type: (FrenetState2D, float, float, float) -> FrenetState2D
        map_api = MapService.get_instance()

        road = map_api._map_model._map.get_road(road_id)
        segment = road.get_segment(segment_id)

        road_frenet = road.frame.center_frenet_frame
        segment_frenet = segment.frame.center_frenet_frame

        return road_frenet.cstate_to_fstate(segment_frenet.fstate_to_cstate(lane_state))
