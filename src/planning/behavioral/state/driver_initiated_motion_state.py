import numpy as np
from decision_making.src.global_constants import DRIVER_INITIATED_MOTION_PEDAL_THRESH, \
    DRIVER_INITIATED_MOTION_PEDAL_TIME


class DriverInitiatedMotionState:
    pedal_from_time = float
    active_from_time = float

    def __init__(self):
        self.pedal_from_time = np.inf
        self.active_from_time = np.inf

    def update(self, timestamp_in_sec: float, pedal_position: float):
        if pedal_position >= DRIVER_INITIATED_MOTION_PEDAL_THRESH:
            self.pedal_from_time = min(self.pedal_from_time, timestamp_in_sec)  # update pedal_from_time
            # if the pedal was pressed for enough time, update the state
            if timestamp_in_sec - self.pedal_from_time >= DRIVER_INITIATED_MOTION_PEDAL_TIME:
                self.active_from_time = timestamp_in_sec
        else:  # no pedal, set state inactive
            self.pedal_from_time = self.active_from_time = np.inf

    def is_active(self):
        return not np.isinf(self.active_from_time)
