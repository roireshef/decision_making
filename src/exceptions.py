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


def raises(*e: Exception):
    """
    A decorator that determines that a function may raise a specific exception
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
