from enum import Enum
from typing import NamedTuple

# Observation breakdown
STATE = "state"
ACTION_MASK = "action_mask"
LAST_ACTION_IDX = "last_action_idx"


class LongitudinalDirection(Enum):
    BEHIND = -1
    AHEAD = 1

    def __int__(self):
        return self.value


class LateralDirection(Enum):
    RIGHT = -1
    SAME = 0
    LEFT = 1

    def __int__(self):
        return self.value


class RSSParams(NamedTuple):
    ro: float           # [sec] positive, delay
    accel_rear: float   # [m/sec^2] positive, rear vehicle accel while in delay
    decel_rear: float   # [m/sec^2] positive, rear vehicle decel after delay
    decel_front: float  # [m/sec^2] positive, front vehicle decel after delay

    def with_values(self, **kwargs):
        d = self._asdict()
        d.update(kwargs)
        return RSSParams(**d)


class InfoLevel(Enum):
    METRICS_ONLY = 0
    METRICS_AND_DEBUG = 1
