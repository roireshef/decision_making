import numpy as np
from decision_making.src.global_constants import DRIVER_INITIATED_MOTION_PEDAL_THRESH, \
    DRIVER_INITIATED_MOTION_PEDAL_TIME, DRIVER_INITIATED_MOTION_STOP_BAR_HORIZON, \
    DRIVER_INITIATED_MOTION_VELOCITY_LIMIT, DRIVER_INITIATED_MOTION_MAX_TIME_TO_STOP_BAR, \
    DRIVER_INITIATED_MOTION_TIMEOUT
from decision_making.src.messages.pedal_position_message import PedalPosition
from enum import Enum

from decision_making.src.planning.types import FS_SX, FS_SV, FrenetState2D
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.utils.map_utils import MapUtils


class DIM_States(Enum):
    """
    States of Driver Initiated Motion mechanism
    """
    NORMAL = 0,
    READY = 1,
    ACTIVE = 2


class DriverInitiatedMotionState:
    """
    This class is an implementation of the state machine of DIM:
    0. Start with NORMAL state.
    1. When the velocity is below a threshold and a next stop bar is close, goto READY state and remember the stop bar id.
    2. When the acceleration pedal was pressed for enough time, goto ACTIVE state.
    3. After crossing this stop bar or after timeout since the pedal was released, goto NORMAL state.
    """
    state = DIM_States                  # the current state
    pedal_pressed_from_time = float     # from which time the pedal is pressed continuously till now
    pedal_pressed_last_time = float     # last time the pedal was pressed continuously for time > time_threshold
    stop_bar_id = (int, int)            # the closest stop bar id at the moment of pressing the pedal

    def __init__(self):
        self._reset()

    def update_state(self, ego_lane_id: int, ego_lane_fstate: FrenetState2D, reference_route: GeneralizedFrenetSerretFrame,
                     pedal_position: PedalPosition) -> None:
        """
        Update DriverInitiatedMotionState according to acceleration pedal strength and how much time it's held
        Remark: the first two parameters may be replaced by MapState but importing MapState causes circular import error
        :param ego_lane_id: lane_id of ego
        :param ego_lane_fstate: lane Frenet state of ego (from MapState)
        :param reference_route: reference route to look for stop bar
        :param pedal_position: holds acceleration pedal strength
        """
        ego_velocity = ego_lane_fstate[FS_SV]
        ego_s = reference_route.convert_from_segment_state(ego_lane_fstate, ego_lane_id)[FS_SX]
        if self.state == DIM_States.NORMAL and ego_velocity <= DRIVER_INITIATED_MOTION_VELOCITY_LIMIT:
            stop_bar_horizon = max(DRIVER_INITIATED_MOTION_STOP_BAR_HORIZON,
                                   ego_velocity * DRIVER_INITIATED_MOTION_MAX_TIME_TO_STOP_BAR)
            # find the next bar in the horizon
            stop_bars = MapUtils.get_stop_bar_and_stop_sign(reference_route)
            close_stop_bars = [stop_bar.s for stop_bar in stop_bars if 0 < stop_bar.s - ego_s < stop_bar_horizon]
            if len(close_stop_bars):
                lane_id, lane_fstate = reference_route.convert_to_segment_state(np.array([close_stop_bars[0], 0, 0, 0, 0, 0]))
                self.stop_bar_id = (lane_id, lane_fstate[FS_SX])
                self.state = DIM_States.READY
        if self.state == DIM_States.NORMAL:
            return

        # check the acceleration pedal position
        accel_pedal_position = pedal_position.s_Data.e_Pct_AcceleratorPedalPosition
        timestamp_in_sec = pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds
        if accel_pedal_position >= DRIVER_INITIATED_MOTION_PEDAL_THRESH:
            self.pedal_pressed_from_time = min(self.pedal_pressed_from_time, timestamp_in_sec)  # update pedal_from_time
            # if the pedal was pressed for enough time, DIM state becomes ACTIVE
            if timestamp_in_sec - self.pedal_pressed_from_time >= DRIVER_INITIATED_MOTION_PEDAL_TIME:
                self.pedal_pressed_last_time = timestamp_in_sec
                self.state = DIM_States.ACTIVE
        # if the pedal was released when DIM state is not active, then reset the pedal times
        elif self.state == DIM_States.READY:  # no pedal
            self.pedal_pressed_from_time = self.pedal_pressed_last_time = np.inf

        # check if ego crossed the stop bar or timeout after releasing of pedal
        if self.state == DIM_States.ACTIVE:
            stop_sign_s = reference_route.convert_from_segment_state(np.array([self.stop_bar_id[1], 0, 0, 0, 0, 0]),
                                                                     self.stop_bar_id[0])[FS_SX]
            if stop_sign_s < ego_s or timestamp_in_sec - self.pedal_pressed_last_time > DRIVER_INITIATED_MOTION_TIMEOUT:
                self._reset()  # NORMAL state

    def _reset(self):
        """
        Reset the state to its default settings
        """
        self.state = DIM_States.NORMAL
        self.pedal_pressed_from_time = self.pedal_pressed_last_time = np.inf
        self.stop_bar_id = None

    def stop_bar_to_ignore(self):
        """
        Return stop bar id if the state is ACTIVE
        :return: stop bar id or None if not active
        """
        return self.stop_bar_id if self.state == DIM_States.ACTIVE else None

