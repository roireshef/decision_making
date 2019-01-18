from abc import ABCMeta, abstractmethod

class RoutePlanner(metaclass=ABCMeta):
    """Add comments"""

    @abstractmethod
    def plan(self): # TODO: Set function annotaion
        """Add comments"""
        pass

    @abstractmethod
    def is_takeover_needed(self) -> bool:
        """Add comments"""
        pass
