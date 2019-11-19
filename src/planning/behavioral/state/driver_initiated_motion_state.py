import numpy as np
from decision_making.src.global_constants import DRIVER_INITIATED_MOTION_PEDAL_THRESH, \
    DRIVER_INITIATED_MOTION_PEDAL_TIME
from decision_making.src.messages.pedal_position_message import PedalPosition


class DriverInitiatedMotionState:
    pedal_from_time = float
    active_from_time = float

    def __init__(self):
        self.pedal_from_time = np.inf
        self.active_from_time = np.inf

    def update(self, pedal_position: PedalPosition) -> None:
        """
        Update DriverInitiatedMotionState according to acceleration pedal strength and how much time it's held
        :param pedal_position: holds acceleration pedal strength
        """
        accel_pedal_position = pedal_position.s_Data.e_Pct_AcceleratorPedalPosition
        timestamp_in_sec = pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds
        if accel_pedal_position >= DRIVER_INITIATED_MOTION_PEDAL_THRESH:
            self.pedal_from_time = min(self.pedal_from_time, timestamp_in_sec)  # update pedal_from_time
            # if the pedal was pressed for enough time, update the state
            if timestamp_in_sec - self.pedal_from_time >= DRIVER_INITIATED_MOTION_PEDAL_TIME:
                self.active_from_time = timestamp_in_sec
        else:  # no pedal, set state inactive
            self.pedal_from_time = self.active_from_time = np.inf

    def is_active(self):
        return not np.isinf(self.active_from_time)
