import numpy as np
from decision_making.src.global_constants import DRIVER_INITIATED_MOTION_PEDAL_THRESH, \
    DRIVER_INITIATED_MOTION_PEDAL_TIME, DRIVER_INITIATED_MOTION_MESSAGE_TIMEOUT


class DriverInitiatedMotionState:
    pedal_from_time = float
    active_from_time = float
    stop_bar_lane_id = int
    stop_bar_lane_station = float

    def __init__(self):
        self.pedal_from_time = np.inf
        self.active_from_time = np.inf
        self.stop_bar_lane_id = None
        self.stop_bar_lane_station = None

    def update(self, timestamp_in_sec: float, pedal_position: float, stop_bar_lane_id: int, stop_bar_lane_station: float):
        if pedal_position >= DRIVER_INITIATED_MOTION_PEDAL_THRESH:
            self.pedal_from_time = min(self.pedal_from_time, timestamp_in_sec)  # update pedal_from_time
            # if the pedal was pressed for enough time, update the state
            if timestamp_in_sec - self.pedal_from_time >= DRIVER_INITIATED_MOTION_PEDAL_TIME:
                self.active_from_time = timestamp_in_sec
                self.stop_bar_lane_id = stop_bar_lane_id
                self.stop_bar_lane_station = stop_bar_lane_station
        else:  # no pedal, check for timeout
            self.pedal_from_time = np.inf
            if timestamp_in_sec - self.active_from_time > DRIVER_INITIATED_MOTION_MESSAGE_TIMEOUT:
                self.active_from_time = np.inf

    def is_active(self):
        return not np.isinf(self.active_from_time)
