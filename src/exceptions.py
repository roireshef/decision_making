import six
from abc import ABCMeta

class MsgDeserializationError(Exception):
    pass


class MsgSerializationError(Exception):
    pass


class RoadNotFound(Exception):
    pass


class LongitudeOutOfRoad(Exception):
    pass


class MapCellNotFound(Exception):
    pass


class OutOfSegmentBack(Exception):
    pass


class OutOfSegmentFront(Exception):
    pass


# TRAJECTORY PLANNING


@six.add_metaclass(ABCMeta)
class TrjajectoryPlanningException(Exception):
    pass


class NoValidTrajectoriesFound(TrjajectoryPlanningException):
    pass


def raises(*e: Exception):
    """
    A decorator that determines that a function may raise a specific exception
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
