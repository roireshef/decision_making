import numpy as np
from decision_making.src.global_constants import DRIVER_INITIATED_MOTION_PEDAL_THRESH, \
    DRIVER_INITIATED_MOTION_PEDAL_TIME, DRIVER_INITIATED_MOTION_STOP_BAR_HORIZON, \
    DRIVER_INITIATED_MOTION_VELOCITY_LIMIT, DRIVER_INITIATED_MOTION_MAX_TIME_TO_STOP_BAR, \
    DRIVER_INITIATED_MOTION_TIMEOUT
from decision_making.src.messages.pedal_position_message import PedalPosition
from enum import Enum

from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.types import FS_SX, FS_SV, FrenetState2D
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.utils.map_utils import MapUtils


class DIM_States(Enum):
    """
    States of Driver Initiated Motion mechanism
    """
    DISABLED = 0,   # the state is disabled (no close stop bars, the velocity is too high)
    PENDING = 1,    # there is a close stop bar, but the acceleration pedal was not pressed for enough time
    CONFIRMED = 2   # specific stop bar is ignored by filters


class DriverInitiatedMotionState:
    """
    This class is an implementation of the state machine of DIM:
    0. Start with NORMAL state.
    1. When the velocity is below a threshold and a next stop bar is close, goto READY state and remember the stop bar id.
    2. When the acceleration pedal was pressed for enough time, goto ACTIVE state.
    3. After crossing this stop bar or after timeout since the pedal was released, goto NORMAL state.
    """
    state = DIM_States                  # the current state
    pedal_last_change_time = float      # when pedal state (pressed/released) was changed last time
    is_pedal_pressed = bool             # True if the pedal is currently pressed
    stop_bar_id = (int, int)            # the closest stop bar id at the moment of pressing the pedal

    def __init__(self):
        self._reset()

    def update_state(self, ego_lane_id: int, ego_lane_fstate: FrenetState2D, route_plan: RoutePlan,
                     pedal_position: PedalPosition) -> None:
        """
        Update DriverInitiatedMotionState according to acceleration pedal strength and how much time it's held
        Remark: the first two parameters may be replaced by MapState but importing MapState causes circular import error
        :param ego_lane_id: lane_id of ego
        :param ego_lane_fstate: lane Frenet state of ego (from MapState)
        :param route_plan: the route plan
        :param pedal_position: holds acceleration pedal strength
        """
        # check if we can pass to PENDING state
        can_pass_to_pending_state, stop_bar_id = self._can_pass_to_pending_state(ego_lane_id, ego_lane_fstate, route_plan)
        if can_pass_to_pending_state:
            self.stop_bar_id = stop_bar_id
            self.state = DIM_States.PENDING
        if self.state == DIM_States.DISABLED:
            self._reset()
            return

        # update pedal press/release times according to the acceleration pedal position
        self._update_pedal_times(pedal_position)

        # check if we can pass to CONFIRMED state
        if self._can_pass_to_confirmed_state(pedal_position):
            self.state = DIM_States.CONFIRMED

        # if ego crossed the stop bar or timeout after releasing of pedal then pass to DISABLED state
        if self._can_pass_to_disabled_state(ego_lane_id, ego_lane_fstate, route_plan,
                                            pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds):
            self._reset()  # set DISABLED state

    def stop_bar_to_ignore(self):
        """
        Return stop bar id if the state is ACTIVE
        :return: stop bar id or None if not active
        """
        return self.stop_bar_id if self.state == DIM_States.CONFIRMED else None

    def _reset(self):
        """
        Reset the state to its default settings
        """
        self.state = DIM_States.DISABLED
        self.pedal_last_change_time = np.inf
        self.is_pedal_pressed = False
        self.stop_bar_id = None

    def _update_pedal_times(self, pedal_position: PedalPosition) -> None:
        """
        Update last pedal press/release times according to the current pedal position
        :param pedal_position: current pedal position
        """
        # update self.pedal_pressed_from_time according to the acceleration pedal position
        currently_pressed = pedal_position.s_Data.e_Pct_AcceleratorPedalPosition >= DRIVER_INITIATED_MOTION_PEDAL_THRESH
        if currently_pressed != self.is_pedal_pressed:
            self.pedal_last_change_time = pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds
            self.is_pedal_pressed = currently_pressed

    def _can_pass_to_pending_state(self, ego_lane_id: int, ego_lane_fstate: FrenetState2D,
                                   route_plan: RoutePlan) -> [bool, float]:
        """
        Check if the DIM state machine can pass to the state PENDING
        For passing to PENDING state ego velocity should be under a threshold and should be close stop bar.
        :param ego_lane_id: ego lane segment id
        :param ego_lane_fstate: ego lane Frenet state (in its lane segment)
        :param route_plan: the route plan
        :return: 1. whether we can pass to PENDING state,
                 2. if yes, s (relatively to reference_route) of the closest stop bar, None otherwise
        """
        ego_velocity = ego_lane_fstate[FS_SV]
        ego_s = reference_route.convert_from_segment_state(ego_lane_fstate, ego_lane_id)[FS_SX]
        # for passing to PENDING state ego velocity should be under a threshold
        if self.state == DIM_States.DISABLED and ego_velocity <= DRIVER_INITIATED_MOTION_VELOCITY_LIMIT:
            # stop bar should be closer than a threshold in meters or in seconds
            stop_bar_horizon = max(DRIVER_INITIATED_MOTION_STOP_BAR_HORIZON,
                                   ego_velocity * DRIVER_INITIATED_MOTION_MAX_TIME_TO_STOP_BAR)
            # find the next bar in the horizon
            stop_bars = MapUtils.get_stop_bar_and_stop_sign(reference_route)
            close_stop_bars = [stop_bar.s for stop_bar in stop_bars if 0 < stop_bar.s - ego_s < stop_bar_horizon]
            if len(close_stop_bars) > 0:
                return True, close_stop_bars[0]

        return False, None

    def _can_pass_to_confirmed_state(self, pedal_position: PedalPosition):
        """
        Check if the DIM state machine can pass to the state CONFIRMED
        For passing to CONFIRMED state the acceleration pedal should be pressed strong enough for enough time.
        :param pedal_position: the current pedal position
        :return: True if we can pass to the state CONFIRMED
        """
        if self.state == DIM_States.PENDING:
            timestamp_in_sec = pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds
            # if the pedal was pressed strongly enough and for enough time
            return self.is_pedal_pressed and timestamp_in_sec - self.pedal_last_change_time >= DRIVER_INITIATED_MOTION_PEDAL_TIME

        return False

    def _can_pass_to_disabled_state(self, ego_lane_id: int, ego_lane_fstate: FrenetState2D,
                                    reference_route: GeneralizedFrenetSerretFrame, timestamp_in_sec: float):
        """
        Check if the DIM state machine can pass to the state DISABLED
        :param ego_lane_id: ego lane segment id
        :param ego_lane_fstate: ego lane Frenet state (in its lane segment)
        :param reference_route: the target reference route (GFF)
        :param timestamp_in_sec: [sec] pedal message timestamp
        :return: True if
        """
        if self.state == DIM_States.CONFIRMED:
            ego_s = reference_route.convert_from_segment_state(ego_lane_fstate, ego_lane_id)[FS_SX]
            stop_sign_s = reference_route.convert_from_segment_state(np.array([self.stop_bar_id[1], 0, 0, 0, 0, 0]),
                                                                     self.stop_bar_id[0])[FS_SX]
            return stop_sign_s < ego_s or \
                   (not self.is_pedal_pressed and timestamp_in_sec - self.pedal_last_change_time > DRIVER_INITIATED_MOTION_TIMEOUT)

        return False
