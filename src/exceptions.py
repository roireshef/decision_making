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


class StateHasNotArrivedYet(Warning):
    pass


# TRAJECTORY PLANNING
@six.add_metaclass(ABCMeta)
class TrajectoryPlanningException(Exception):
    pass


class CartesianLimitsViolated(TrajectoryPlanningException):
    pass


class FrenetLimitsViolated(TrajectoryPlanningException):
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


class RoadNotFound(MappingException):
    pass


class LaneNotFound(MappingException):
    pass

class EgoStationBeyondLaneLength(MappingException):
    pass

# ROUTE PLANNING
@six.add_metaclass(ABCMeta)
class RoutePlanningException(Exception):
    pass

class RepeatedRoadSegments(RoutePlanningException):
    pass

class EgoRoadSegmentNotFound(RoutePlanningException):
    pass

class RouteRoadSegmentNotFound(RoutePlanningException):
    pass

class RouteLaneSegmentNotFound(RoutePlanningException):
    pass

class EgoLaneOccupancyCostIncorrect(RoutePlanningException):
    pass

class RoadSegmentLaneSegmentMismatch(RoutePlanningException):
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
