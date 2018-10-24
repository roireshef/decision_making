from typing import List

import numpy as np

from decision_making.src.planning.behavioral.behavioral_grid_state import RelativeLane
from decision_making.src.planning.types import C_X, C_Y, CartesianExtendedState, FS_DX, FrenetState2D, FS_SX
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import DynamicObject
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
        # TODO: MapService
        closest_lane_id = map_api._find_closest_lane(closest_road_id, cartesian_state[C_X], cartesian_state[C_Y])

        lane_frenet = map_api.get_lane_frenet(closest_lane_id)

        obj_fstate = lane_frenet.cstate_to_fstate(cartesian_state)

        return MapState(obj_fstate, closest_lane_id)

    @staticmethod
    def convert_map_to_cartesian_state(map_state):
        # type: (MapState) -> CartesianExtendedState
        # TODO: MapService
        lane_frenet = MapService.get_instance().get_lane_frenet(map_state.lane_id)
        return lane_frenet.fstate_to_cstate(map_state.lane_fstate)

    @staticmethod
    def is_object_on_road(map_state):
        # type: (MapState) -> bool
        """
        Returns true of the object is on the road. False otherwise.
        Note! This function is valid only when the frenet reference frame is from the right side of the road
        :param map_state: the map state to check
        :return: Returns true of the object is on the road. False otherwise.
        """
        dist_from_right, dist_from_left = MapService.get_instance().dist_from_lane_borders(
            map_state.lane_id, map_state.lane_fstate[FS_SX])
        is_on_road = -dist_from_right - ROAD_SHOULDERS_WIDTH < map_state.lane_fstate[FS_DX] < dist_from_left + ROAD_SHOULDERS_WIDTH
        return is_on_road

    @staticmethod
    def project_on_relative_lanes(obj, relative_lanes):
        # type: (DynamicObject, List[RelativeLane]) -> List[FrenetState2D]
        """
        Calculate frenet-states of the given object w.r.t. the relative (adjacent) lanes
        :param obj: dynamic object
        :param relative_lanes: list of relative lanes (same, left, right)
        :return: list of frenet-states of size len(relative_lanes)
        """
        return [None
                if relative_lane != RelativeLane.LEFT_LANE else obj.left_map_state.lane_fstate
                if relative_lane != RelativeLane.RIGHT_LANE else obj.right_map_state.lane_fstate
                if relative_lane != RelativeLane.SAME_LANE else obj.map_state.lane_fstate
                for relative_lane in relative_lanes]
