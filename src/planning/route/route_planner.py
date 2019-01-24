from abc import ABCMeta, abstractmethod
from decision_making.src.messages.scene_static_lite_message import SceneStaticLite

class RoutePlanner(metaclass=ABCMeta):
    """Add comments"""

    @abstractmethod
    def plan(self): # TODO: Set function annotaion
        """Add comments"""
        pass
