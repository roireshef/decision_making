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


class NavigationPlanTooShort(MappingException):
    pass


class RoadNotFound(MappingException):
    pass


class LaneNotFound(MappingException):
    pass


class EquivalentStationNotFound(MappingException):
    pass


class IDAppearsMoreThanOnce(MappingException):
    pass


class StraightConnectionNotFound(MappingException):
    pass


class ConstraintFilterHaltWithValue(Exception):
    """
    This is raised internally within ConstraintFilter when halt (stopping the filter without completing the entire execution)
     is needed (with value)
    """
    def __init__(self, value: bool):
        self._value = value

    @property
    def value(self) -> bool:
        return self._value


class NoActionsLeftForBPError(Exception):
    pass


class OutOfSegmentBack(Exception):
    pass


class OutOfSegmentFront(Exception):
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


class EgoLaneOccupancyCostIncorrect(RoutePlanningException):
    pass


class RoadSegmentLaneSegmentMismatch(RoutePlanningException):
    pass


class MissingInputInformation(RoutePlanningException):
    pass


class NavigationSceneDataMismatch(RoutePlanningException):
    pass


class LaneSegmentDataNotFound(RoutePlanningException):
    pass


class RoadSegmentDataNotFound(RoutePlanningException):
    pass


class LaneAttributeNotFound(RoutePlanningException):
    pass


class DownstreamLaneDataNotFound(RoutePlanningException):
    pass


def raises(*e):
    """
    A decorator that determines that a function may raise a specific exception
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
