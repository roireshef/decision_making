
import numpy as np
from decision_making.src.global_constants import PUBSUB_MSG_IMPL, DRIVER_INITIATED_MOTION_PEDAL_THRESH, \
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

    def update(self, timestamp_in_sec: float, pedal_rate: float, stop_bar_lane_id: int, stop_bar_lane_station: float):
        if pedal_rate >= DRIVER_INITIATED_MOTION_PEDAL_THRESH:
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


class DriverInitiatedMotionMessage(PUBSUB_MSG_IMPL):
    timestamp_in_sec = float
    pedal_rate = float

    def __init__(self, timestamp_in_sec: float, pedal_rate: float):
        self.timestamp_in_sec = timestamp_in_sec
        self.pedal_rate = pedal_rate

    def serialize(self) -> TsSYSDriverInitiatedMotionMessage:
        pubsub_msg = TsSYSDriverInitiatedMotionMessage()
        pubsub_msg.timestamp_in_sec = self.timestamp_in_sec
        pubsub_msg.pedal_rate = self.pedal_rate
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDriverInitiatedMotionMessage):
        return cls(pubsubMsg.timestamp_in_sec, pubsubMsg.pedal_rate)
