from logging import Logger

import numpy as np

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.filtering.safety_utils import SafetyUtils
from decision_making.src.planning.types import C_V, C_YAW, C_Y, C_X, C_K, C_A
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.state import EgoState, ObjectSize, DynamicObject, State
from mapping.src.service.map_service import MapService


def test_isSafeSpec():
    logger = Logger("test_safetyUtils")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    car_length = 4
    size = ObjectSize(car_length, 2, 1)
    road_frenet = get_road_rhs_frenet_by_road_id(road_id)

    ego_vel = 14
    ego_init_fstate = np.array([ego_lon, ego_vel, 0, lane_width / 2, 0, 0])
    cstate = road_frenet.fstate_to_cstate(ego_init_fstate)
    ego = EgoState(0, 0, cstate[C_X], cstate[C_Y], 0, cstate[C_YAW], size, 0, cstate[C_V], 0, cstate[C_A], cstate[C_K])

    F_lon = ego_lon + 40
    F_vel = 10
    F = create_canonic_object(1, 0, F_lon, lane_width / 2, F_vel, size, road_frenet)

    objects = [F]
    state = State(None, objects, ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    target_vel = ego_vel
    T = 10
    s = T * (ego_vel + target_vel)/2
    spec = ActionSpec(T, target_vel, ego_lon + s, lane_width / 2)

    is_safe = SafetyUtils.is_safe_spec(behavioral_state, ego_init_fstate, spec)
    is_safe = is_safe


def create_canonic_object(obj_id: int, timestamp: int, lon: float, lat: float, vel: float, size: ObjectSize,
                          road_frenet: FrenetSerret2DFrame) -> DynamicObject:
    """
    Create object with zero lateral velocity and zero accelerations
    """
    fstate = np.array([lon, vel, 0, lat, 0, 0])
    cstate = road_frenet.fstate_to_cstate(fstate)
    return DynamicObject(obj_id, timestamp, cstate[C_X], cstate[C_Y], 0, cstate[C_YAW], size, 0, cstate[C_V], 0,
                         cstate[C_A], cstate[C_K])


def get_road_rhs_frenet_by_road_id(road_id: int):
    return MapService.get_instance()._rhs_roads_frenet[road_id]