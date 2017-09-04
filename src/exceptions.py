class MsgDeserializationError(Exception):
    pass


class MsgSerializationError(Exception):
    pass


class RoadNotFound(Exception):
    pass


class LongitudeOutOfRoad(Exception):
    pass


# a decorator that determines that a function may raise a specific exception
def raises(*e: Exception):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
