from src.planning.messages.dds_message import DDSMessage
from abc import ABCMeta, abstractmethod


class BehavioralDebugData(DDSMessage, metaclass=ABCMeta):
    """
    The struct used for communicating the behavioral debug data to the debug modules.
    It is defined as abstract class, so every instance of behavioral planner would
    implement its' properties in a way that fits it's architecture
    """
    def __init__(self):
        super(BehavioralDebugData, self).__init__()
        self.reset()


    @abstractmethod
    def reset(self) -> None:
        """
        Reset all class members
        :return: void
        """

