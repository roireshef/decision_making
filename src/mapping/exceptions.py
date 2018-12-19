from typing import Any


class RoadNotFound(Exception):
    pass


class LaneNotFound(Exception):
    pass


class NextRoadNotFound(Exception):
    pass


class LongitudeOutOfRoad(Exception):
    pass


class MapCellNotFound(Exception):
    pass


class OutOfSegmentBack(Exception):
    pass


class OutOfSegmentFront(Exception):
    pass


class MapServiceNotInitialized(Exception):
    pass


class WrongMapService(Exception):
    pass


def raises(*e):
    # type: (Exception) -> Any
    """
    A decorator that determines that a function may raise a specific exception
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
