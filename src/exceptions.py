import six
from abc import ABCMeta


class PredictObjectInPastTimes(Exception):
    pass

class MsgDeserializationError(Exception):
    pass


class MsgSerializationError(Exception):
    pass


# TRAJECTORY PLANNING
@six.add_metaclass(ABCMeta)
class TrjajectoryPlanningException(Exception):
    pass


class NoValidTrajectoriesFound(TrjajectoryPlanningException):
    pass


# TRAJECTORY PLANNING
@six.add_metaclass(ABCMeta)
class BehavioralPlanningException(Exception):
    pass


class VehicleOutOfRoad(BehavioralPlanningException):
    pass


class NoValidLanesFound(BehavioralPlanningException):
    pass


def raises(*e):
    # type: (Exception)
    """
    A decorator that determines that a function may raise a specific exception
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
