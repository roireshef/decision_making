from abc import ABCMeta

from decision_making.src.planning.types import MinGlobalTimeStampInSec, GlobalTimeStampInSec
from decision_making.src.state.state import State


class TimeAlignmentPredictor(metaclass=ABCMeta):
    """
    Performs prediction for the purpose of short time alignment between ego and dynamic objects.
    """

    def __init__(self, logger):
        self._logger = logger

    def align_objects_to_most_recent_timestamp(self, state: State,
                                               current_timestamp: GlobalTimeStampInSec=MinGlobalTimeStampInSec) -> State:
        """
        Returns state with all objects aligned to the most recent timestamp.
        Most recent timestamp is taken as the max between the current_timestamp, and the most recent
        timestamp of all objects in the scene.
        :param state: state containing objects with different timestamps
        :param current_timestamp: current timestamp in global time in [sec]
        :return: new state with all objects aligned
        """

        pass
