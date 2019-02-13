import six
from abc import ABCMeta


class MultipleObjectsWithRequestedID(Exception):
    pass


class TimeAlignmentPredictionHorizonTooLong(Exception):
    pass


class ObjectHasNegativeVelocityError(Exception):
    pass


class ResourcesNotUpToDateException(Exception):
    pass


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


class CouldNotGenerateTrajectories(TrjajectoryPlanningException):
    pass


# BEHAVIORAL PLANNING
@six.add_metaclass(ABCMeta)
class BehavioralPlanningException(Exception):
    pass


class ActionOutOfSpec(BehavioralPlanningException):
    pass


class InvalidAction(BehavioralPlanningException):
    pass


class VehicleOutOfRoad(BehavioralPlanningException):
    pass


class NoValidLanesFound(BehavioralPlanningException):
    pass


class SceneModelIsEmpty(Exception):
    pass


# BEHAVIORAL PLANNING
@six.add_metaclass(ABCMeta)
class MappingException(Exception):
    pass


class UpstreamLaneNotFound(MappingException):
    pass


class DownstreamLaneNotFound(MappingException):
    pass


class NavigationPlanTooShort(MappingException):
    pass


class NavigationPlanDoesNotFitMap(MappingException):
    pass


class AmbiguousNavigationPlan(MappingException):
    pass


class IntersectionNotFound(MappingException):
    pass


class RoadNotFound(MappingException):
    pass


class LaneNotFound(MappingException):
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
