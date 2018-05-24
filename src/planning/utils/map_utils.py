from decision_making.src.planning.types import CartesianExtendedState, C_X, C_Y
from decision_making.src.planning.types import FrenetState2D
from decision_making.src.state.state import MapState
from mapping.src.localization.localization import GlobalLocalization, SegmentLocalization, MapLocalization, \
    LaneLocalization, RoadLocalization
from mapping.src.service.map_service import MapService


class MapUtils:
    @staticmethod
    def convert_cartesian_to_map_state(cartesian_state: CartesianExtendedState):
        # type: (CartesianExtendedState) -> MapState
        # TODO: replace with query that returns only the relevant road id
        map_api = MapService.get_instance()

        x, y = cartesian_state[C_X], cartesian_state[C_Y]
        global_localization = GlobalLocalization(x, y, 0, None)
        map_localization = map_api.global_to_map_localization(global_localization)

        lane = map_api.resolve_location_on_map(map_localization)
        lane_fstate = lane.get_as_frame().get_center_frame().cstate_to_fstate(cartesian_state)

        segment = lane.parent

        road = lane.parent.parent
        road_fstate = road.get_as_frame().get_center_frame().cstate_to_fstate(cartesian_state)

        return MapState(lane_fstate, road_fstate, road.id, segment.id, lane.id)

    @staticmethod
    def convert_map_to_cartesian_state(map_state):
        # type: (MapState) -> FrenetState2D
        map_api = MapService.get_instance()

        road_localization = RoadLocalization(map_state.road_id, 0)
        segment_localization = SegmentLocalization(map_state.segment_id, 0, 0)
        lane_localization = LaneLocalization(map_state.lane_id, 0, 0, None)
        map_localization = MapLocalization(road_localization, segment_localization, lane_localization)

        lane = map_api.resolve_location_on_map(map_localization)

        frenet = lane.get_as_frame().get_center_frame()

        return frenet.cstate_to_fstate(map_state.lane_state)

    @staticmethod
    def convert_lane_to_road_localization(lane_state, road_id, segment_id, lane_id):
        # type: (FrenetState2D, float, float, float) -> FrenetState2D
        map_api = MapService.get_instance()

        road = map_api._map_model._map.get_road(road_id)
        segment = road.get_segment(segment_id)
        lane = segment.get_lane(lane_id)

        road_frenet = road.get_as_frame().get_center_frame()
        lane_frenet = lane.get_as_frame().get_center_frame()

        return road_frenet.cstate_to_fstate(lane_frenet.fstate_to_cstate(lane_state))

    @staticmethod
    def convert_road_to_lane_localization(lane_state, road_id, segment_id, lane_id):
        # type: (FrenetState2D, float, float, float) -> FrenetState2D
        map_api = MapService.get_instance()

        road = map_api._map_model._map.get_road(road_id)
        segment = road.get_segment(segment_id)
        lane = segment.get_lane(lane_id)

        road_frenet = road.get_as_frame().get_center_frame()
        lane_frenet = lane.get_as_frame().get_center_frame()

        return lane_frenet.cstate_to_fstate(road_frenet.fstate_to_cstate(lane_state))

    @staticmethod
    def convert_lane_to_segment_localization(lane_state, road_id, segment_id, lane_id):
        # type: (FrenetState2D, float, float, float) -> FrenetState2D
        map_api = MapService.get_instance()

        road = map_api._map_model._map.get_road(road_id)
        segment = road.get_segment(segment_id)
        lane = segment.get_lane(lane_id)

        segment_frenet = segment.get_as_frame().get_center_frame()
        lane_frenet = lane.get_as_frame().get_center_frame()

        return segment_frenet.cstate_to_fstate(lane_frenet.fstate_to_cstate(lane_state))

    @staticmethod
    def convert_segment_to_lane_localization(lane_state, road_id, segment_id, lane_id):
        # type: (FrenetState2D, float, float, float) -> FrenetState2D
        map_api = MapService.get_instance()

        road = map_api._map_model._map.get_road(road_id)
        segment = road.get_segment(segment_id)
        lane = segment.get_lane(lane_id)

        segment_frenet = segment.get_as_frame().get_center_frame()
        lane_frenet = lane.get_as_frame().get_center_frame()

        return lane_frenet.cstate_to_fstate(segment_frenet.fstate_to_cstate(lane_state))

    @staticmethod
    def convert_road_to_segment_localization(lane_state, road_id, segment_id):
        # type: (FrenetState2D, float, float, float) -> FrenetState2D
        map_api = MapService.get_instance()

        road = map_api._map_model._map.get_road(road_id)
        segment = road.get_segment(segment_id)

        road_frenet = road.get_as_frame().get_center_frame()
        segment_frenet = segment.get_as_frame().get_center_frame()

        return segment_frenet.cstate_to_fstate(road_frenet.fstate_to_cstate(lane_state))

    @staticmethod
    def convert_segment_to_road_localization(lane_state, road_id, segment_id):
        # type: (FrenetState2D, float, float, float) -> FrenetState2D
        map_api = MapService.get_instance()

        road = map_api._map_model._map.get_road(road_id)
        segment = road.get_segment(segment_id)

        road_frenet = road.get_as_frame().get_center_frame()
        segment_frenet = segment.get_as_frame().get_center_frame()

        return road_frenet.cstate_to_fstate(segment_frenet.fstate_to_cstate(lane_state))
