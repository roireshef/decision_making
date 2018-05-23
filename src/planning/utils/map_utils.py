from decision_making.src.planning.types import CartesianExtendedState, C_X, C_Y
from decision_making.src.planning.types import FrenetState2D
from decision_making.src.state.state import MapState
from mapping.src.localization.localization import GlobalLocalization, SegmentLocalization, MapLocalization, \
    LaneLocalization, RoadLocalization
from mapping.src.service.map_service import MapService
import numpy as np


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








    # TODO: replace with navigation plan aware function from map API
    @staticmethod
    def get_road_rhs_frenet(obj: DynamicObject):
        return MapService.get_instance()._rhs_roads_frenet[obj.road_localization.road_id]

    # TODO: replace this call with the road localization once it is updated to be hold a frenet state
    @staticmethod
    def get_object_road_localization(obj: DynamicObject, road_frenet: FrenetSerret2DFrame):
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
    def nonoverlapping_longitudinal_distance(ego_fstate: FrenetState2D, obj_fstate: FrenetState2D,
                                             ego_length: float, obj_length: float):
        """
        Given dynamic object in a cell, calculate the distance from the object's boundaries to ego vehicle boundaries
        :param ego_fstate: FrenetState2D of the ego vehicle
        :param obj_fstate: FrenetState2D of the another object
        :param ego_length: the length of the ego vehicle
        :param obj_length: the length of the another object
        :return: if object is in front of ego, then the returned value is positive and reflect the longitudinal distance
        between object's rear to ego-front. If the object behind ego, then the returned value is negative and reflect
        the longitudinal distance between object's front to ego-rear. If there's an overlap between ego and object on
        the longitudinal axis, the returned value is 0
        """
        # Relative longitudinal distance
        object_relative_lon = obj_fstate[FS_SX] - ego_fstate[FS_SX]

        if object_relative_lon > (obj_length / 2 + ego_length / 2):
            return object_relative_lon - (obj_length / 2 + ego_length / 2)
        elif object_relative_lon < -(obj_length / 2 + ego_length / 2):
            return object_relative_lon + (obj_length / 2 + ego_length / 2)
        else:
            return 0