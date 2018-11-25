from typing import List

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import C_X, C_Y, CartesianPoint2D
from decision_making.src.planning.types import CartesianExtendedState, FS_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.map_state import MapState
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.service.map_service import MapService


class MapUtils:
    # TODO: remove this on Lane-based planner PR
    @staticmethod
    def get_road_rhs_frenet(obj):
        return MapService.get_instance()._rhs_roads_frenet[obj.map_state.road_id]

    # TODO: remove this on Lane-based planner PR
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

    # TODO: remove this on Lane-based planner PR
    @staticmethod
    def convert_map_to_cartesian_state(map_state):
        # type: (MapState) -> CartesianExtendedState
        map_api = MapService.get_instance()

        road_frenet = map_api._rhs_roads_frenet[map_state.road_id]

        return road_frenet.fstate_to_cstate(map_state.road_fstate)

    # TODO: remove this on Lane-based planner PR
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
        is_on_road = road_width + ROAD_SHOULDERS_WIDTH > map_state.road_fstate[FS_DX] > -ROAD_SHOULDERS_WIDTH
        return is_on_road

    @staticmethod
    def get_road_segment_id_from_lane_id(lane_id: int) -> int:
        """
        get road_segment_id containing the lane
        :param lane_id:
        :return: road_segment_id
        """
        pass

    @staticmethod
    def get_lane_ordinal(lane_id: int) -> int:
        """
        get lane ordinal of the lane on the road (the rightest lane's ordinal is 0)
        :param lane_id:
        :return: lane's ordinal
        """
        pass

    @staticmethod
    def get_lane_frenet_frame(lane_id: int) -> FrenetSerret2DFrame:
        """
        get Frenet frame of the whole center-lane for the given lane
        :param lane_id:
        :return: Frenet frame
        """
        pass

    @staticmethod
    def get_lane_length(lane_id: int) -> float:
        """
        get the whole lane's length
        :param lane_id:
        :return: lane's length
        """
        pass

    @staticmethod
    def get_adjacent_lanes(lane_id: int, relative_lane: RelativeLane) -> List[int]:
        """
        get sorted adjacent (right/left) lanes relative to the given lane segment
        :param lane_id:
        :param relative_lane: either right or left
        :return: adjacent lanes ids sorted by their distance from the given lane
        """
        pass

    # TODO: remove it after introduction of the new mapping module
    @staticmethod
    def get_closest_lane(cartesian_point: CartesianPoint2D, road_segment_id: int = None) -> int:
        """
        given cartesian coordinates, find the closest lane to the point
        :param cartesian_point: 2D cartesian coordinates
        :param road_segment_id: optional argument for road_segment_id closest to the given point
        :return: closest lane segment id
        """
        pass

    @staticmethod
    def get_dist_from_lane_center_to_lane_borders(lane_id: int, s: float) -> (float, float):
        """
        get distance from the lane center to the lane borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right lane border, distance from the left lane border
        """
        pass

    @staticmethod
    def get_dist_from_lane_center_to_road_borders(lane_id: int, s: float) -> (float, float):
        """
        get distance from the lane center to the road borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right road border, distance from the left road border
        """
        pass

    @staticmethod
    def get_lane_width(lane_id: int, s: float) -> float:
        """
        get lane width at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: lane width
        """
        pass

    @staticmethod
    def get_lookahead_frenet_frame(lane_id: int, starting_lon: float, lookahead_dist: float,
                                   navigation_plan: NavigationPlanMsg):
        """
        Get Frenet frame of a given length along lane center, starting from given lane's longitude (may be negative).
        When some lane finishes, it automatically continues to the next lane, according to the navigation plan.
        :param lane_id: starting lane_id
        :param starting_lon: starting longitude (may be negative) [m]
        :param lookahead_dist: lookahead distance for the output frame [m]
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: Frenet frame for the given route part
        """
        pass

    @staticmethod
    def get_upstream_lanes(lane_id: int) -> List[int]:
        """
        get upstream lanes (incoming) of the given lane
        :param lane_id:
        :return: list of upstream lanes ids
        """
        pass

    @staticmethod
    def get_downstream_lanes(lane_id: int) -> List[int]:
        """
        get downstream lanes (outgoing) of the given lane
        :param lane_id:
        :return: list of downstream lanes ids
        """
        pass

    @staticmethod
    def get_lanes_id_from_road_segment_id(road_segment_id: int) -> List[int]:
        """
        Get sorted list of lanes for given road segment. The output lanes are ordered by the lanes' ordinal,
        i.e. from the rightest lane to the most left.
        :param road_segment_id:
        :return: sorted list of lane segments' IDs
        """
        pass
