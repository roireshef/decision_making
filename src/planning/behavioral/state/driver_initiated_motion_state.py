import multiprocessing as mp
from enum import Enum
from logging import Logger
from typing import Tuple

import numpy as np
from decision_making.src.global_constants import DRIVER_INITIATED_MOTION_PEDAL_THRESH, \
    DRIVER_INITIATED_MOTION_PEDAL_TIME, DRIVER_INITIATED_MOTION_STOP_BAR_HORIZON, \
    DRIVER_INITIATED_MOTION_VELOCITY_LIMIT, DRIVER_INITIATED_MOTION_MAX_TIME_TO_STOP_BAR, \
    DRIVER_INITIATED_MOTION_TIMEOUT, STOP_BAR_IND, STOP_BAR_DISTANCE_IND, \
    DIM_MARGIN_TO_STOP_BAR
from decision_making.src.messages.pedal_position_message import PedalPosition
from decision_making.src.messages.scene_static_message import TrafficControlBar
from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import FS_SV, FrenetState2D
from decision_making.src.utils.dummy_queue import DummyQueue


class DIM_States(Enum):
    """
    States of Driver Initiated Motion mechanism
    """
    DISABLED = 0    # the state is disabled (no close stop bars, the velocity is too high)
    PENDING = 1     # there is a close stop bar, but the acceleration pedal was not pressed for enough time
    CONFIRMED = 2   # specific stop bar is ignored by filters


class DriverInitiatedMotionState(PUBSUB_MSG_IMPL):
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
    stop_bar_id = int                   # the closest stop bar id at the moment of pressing the pedal

    def __init__(self, logger: Logger, visualizer_queue: mp.Queue = DummyQueue()):
        self.logger = logger
        self.visualizer_queue = visualizer_queue
        self._reset()

    def to_dict(self, left_out_fields=None):
        return {k: self._serialize_element(v) for k, v in self.__dict__.items() if k != 'logger' and k != 'visualizer_queue'}

    def update_state(self, timestamp_in_sec: float, ego_lane_fstate: FrenetState2D,
                     ego_s: float, closestTCB: Tuple[TrafficControlBar, float], ignored_TCB_distance: float) -> None:
        """
        Update DriverInitiatedMotionState according to acceleration pedal strength and how much time it's held
        :param timestamp_in_sec: current timestamp
        :param ego_lane_fstate: lane Frenet state of ego (from MapState)
        :param ego_s: s location of ego in GFF
        :param closestTCB: Tuple of TCB object and its s location in GFF
        :param ignored_TCB_distance: ignored TCB s-location
        """
        # check if we can pass to PENDING state
        can_pass_to_pending_state, stop_bar_s = self._can_pass_to_pending_state(ego_lane_fstate, ego_s, closestTCB)
        if can_pass_to_pending_state:
            self.stop_bar_id = closestTCB[STOP_BAR_IND].e_i_traffic_control_bar_id
            self.state = DIM_States.PENDING
            self.logger.debug('DIM state: PENDING; stop_bar_id %s', self.stop_bar_id)

        # check if we can pass to CONFIRMED state
        if self._can_pass_to_confirmed_state(timestamp_in_sec):
            self.state = DIM_States.CONFIRMED
            self.visualizer_queue.put(self.state)
            self.logger.debug('DIM state: CONFIRMED; stop_bar_id %s', self.stop_bar_id)
            # don't move to the "if _can_pass_to_disabled_state" since the ignored is not set properly yet.
            # Will be set in the next cycle
            return

        # if ego crossed the stop bar or timeout after releasing of pedal then pass to DISABLED state
        if self._can_pass_to_disabled_state(ego_s, ignored_TCB_distance, timestamp_in_sec):
            self._reset()  # set DISABLED state
            self.logger.debug('DIM state: DISABLED; ')
        self.visualizer_queue.put(self.state)

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
        self.logger.debug('DIM state: DISABLED; ')
        self.visualizer_queue.put(self.state)

    def update_pedal_times(self, pedal_position: PedalPosition) -> None:
        """
        Update last pedal press/release times according to the current pedal position
        :param pedal_position: current pedal position
        """
        # update self.pedal_pressed_from_time according to the acceleration pedal position
        currently_pressed = pedal_position.s_Data.e_Pct_AcceleratorPedalPosition >= DRIVER_INITIATED_MOTION_PEDAL_THRESH
        if currently_pressed != self.is_pedal_pressed:
            self.pedal_last_change_time = pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds
            self.is_pedal_pressed = currently_pressed

    def _can_pass_to_pending_state(self, ego_lane_fstate: FrenetState2D, ego_s: float,
                                   closestTCB: Tuple[TrafficControlBar, float]) -> [bool, float]:
        """
        Check if the DIM state machine can pass to the state PENDING
        For passing to PENDING state ego velocity should be under a threshold and should be close stop bar.
        :param ego_lane_fstate: ego lane Frenet state (in its lane segment)
        :param ego_s: location of ego in GFF
        :param closestTCB: Tuple of TCB object and its s location in GFF
        :return: 1. whether we can pass to PENDING state,
                 2. if yes, s (relatively to reference_route) of the closest stop bar, None otherwise
        """
        ego_velocity = ego_lane_fstate[FS_SV]
        # for passing to PENDING state ego velocity should be under a threshold
        if self.state == DIM_States.DISABLED and ego_velocity <= DRIVER_INITIATED_MOTION_VELOCITY_LIMIT:
            # stop bar should be closer than a threshold in meters or in seconds
            stop_bar_horizon = max(DRIVER_INITIATED_MOTION_STOP_BAR_HORIZON,
                                   ego_velocity * DRIVER_INITIATED_MOTION_MAX_TIME_TO_STOP_BAR)
            # find the next bar in the horizon
            if closestTCB is not None and closestTCB[STOP_BAR_IND] is not None:
                closest_tcb_distance = closestTCB[STOP_BAR_DISTANCE_IND]
                if ego_s < closest_tcb_distance < ego_s + stop_bar_horizon:
                    return True, closest_tcb_distance

        return False, None

    def _can_pass_to_confirmed_state(self, timestamp_in_sec: float):
        """
        Check if the DIM state machine can pass to the state CONFIRMED
        For passing to CONFIRMED state the acceleration pedal should be pressed strong enough for enough time.
        :param timestamp_in_sec: current timestamp
        :return: True if we can pass to the state CONFIRMED
        """
        if self.state == DIM_States.PENDING:
            # if the pedal was pressed strongly enough and for enough time
            return self.is_pedal_pressed and timestamp_in_sec - self.pedal_last_change_time >= DRIVER_INITIATED_MOTION_PEDAL_TIME

        return False

    def _can_pass_to_disabled_state(self, ego_s: float, ignored_TCB_distance: float, timestamp_in_sec: float) -> bool:
        """
        Check if the DIM state machine can pass to the state DISABLED
        :param ego_s: s location of the ego in GFF
        :param ignored_TCB_distance: ignored TCB s-distance
        :param timestamp_in_sec: [sec] current timestamp
        :return: True if we can pass to the state DISABLED
        """
        if self.state == DIM_States.CONFIRMED:
            # vehicle passed the stop bar by more than DIM_MARGIN_TO_STOP_BAR or
            # timed out
            return ignored_TCB_distance is None or \
                   ignored_TCB_distance + DIM_MARGIN_TO_STOP_BAR < ego_s or \
                   (not self.is_pedal_pressed and timestamp_in_sec - self.pedal_last_change_time > DRIVER_INITIATED_MOTION_TIMEOUT)
        return False
