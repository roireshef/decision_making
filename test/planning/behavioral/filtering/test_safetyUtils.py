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


def test_isSafeSpecCalcLaneChangeLastSafeTime_3objects4specs_fastUnsafeLF_slowUnsafeLB():
    """
    This function tests two safety functions together since their results are tightly connected
    and they share the same code.
    The function is_safe_spec() is used by a filter and checks safety w.r.t. LF for the whole trajectory, and
    w.r.t. F & LB only during the first T_d_min (e.g. 3) seconds.
    The function calc_lane_change_last_safe_time() is used by BP costs and checks last safe time only for lane change
    actions and only w.r.t. F & LB.

    The state contains 3 objects: F, LF, LB.
    Test 4 different specs: one of them same_lane, the rest goto_left.
    When the velocity is too fast, ego is unsafe w.r.t. LF
    When the velocity is too slow, ego is unsafe w.r.t. LB.
    """
    logger = Logger("test_safetyUtils")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    car_length = 4
    T_d_min = 3.
    size = ObjectSize(car_length, 2, 1)
    road_frenet = get_road_rhs_frenet_by_road_id(road_id)

    ego_vel = 14
    ego_init_fstate = np.array([ego_lon, ego_vel, 0, lane_width / 2, 0, 0])
    cstate = road_frenet.fstate_to_cstate(ego_init_fstate)
    ego = EgoState(0, 0, cstate[C_X], cstate[C_Y], 0, cstate[C_YAW], size, 0, cstate[C_V], 0, cstate[C_A], cstate[C_K])

    F  = create_canonic_object(1, 0, ego_lon + 110,   lane_width/2, vel=10, size=size, road_frenet=road_frenet)
    LF = create_canonic_object(2, 0, ego_lon + 30,  3*lane_width/2, vel=16, size=size, road_frenet=road_frenet)
    LB = create_canonic_object(3, 0, ego_lon - 70,  3*lane_width/2, vel=18, size=size, road_frenet=road_frenet)

    objects = [F, LF, LB]
    state = State(None, objects, ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)
    T = 10

    target_vel = ego_vel+4
    s = T * (ego_vel + target_vel)/2
    spec = ActionSpec(T, target_vel, ego_lon + s, lane_width/2)  # same lane
    is_safe = SafetyUtils.is_safe_spec(behavioral_state, ego_init_fstate, spec)
    last_safe_time = SafetyUtils.calc_lane_change_last_safe_time(behavioral_state, ego_init_fstate, spec)
    assert is_safe and last_safe_time == np.inf

    target_vel = ego_vel+4
    s = T * (ego_vel + target_vel)/2
    spec = ActionSpec(T, target_vel, ego_lon + s, 3*lane_width/2)  # goto left
    is_safe = SafetyUtils.is_safe_spec(behavioral_state, ego_init_fstate, spec)
    last_safe_time = SafetyUtils.calc_lane_change_last_safe_time(behavioral_state, ego_init_fstate, spec)
    assert not is_safe and last_safe_time == np.inf  # unsafe w.r.t. LF because of high velocity

    target_vel = ego_vel+2
    s = T * (ego_vel + target_vel)/2
    spec = ActionSpec(T, target_vel, ego_lon + s, 3*lane_width/2)  # goto left
    is_safe = SafetyUtils.is_safe_spec(behavioral_state, ego_init_fstate, spec)
    last_safe_time = SafetyUtils.calc_lane_change_last_safe_time(behavioral_state, ego_init_fstate, spec)
    assert is_safe and T_d_min < last_safe_time < np.inf  # last_safe_time w.r.t. LB

    target_vel = ego_vel-4
    s = T * (ego_vel + target_vel)/2
    spec = ActionSpec(T, target_vel, ego_lon + s, 3*lane_width/2)  # goto left
    is_safe = SafetyUtils.is_safe_spec(behavioral_state, ego_init_fstate, spec)
    last_safe_time = SafetyUtils.calc_lane_change_last_safe_time(behavioral_state, ego_init_fstate, spec)
    assert not is_safe and last_safe_time < T_d_min  # unsafe w.r.t. LB because of slow velocity


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