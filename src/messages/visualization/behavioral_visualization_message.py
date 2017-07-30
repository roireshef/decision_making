import numpy as np

from src.messages.dds_message import DDSMessage


class BehavioralVisualizationMessage(DDSMessage):
    def __init__(self, _reference_route: np.ndarray):
        """
        The struct used for communicating the behavioral plan to the visualizer.
        :param _reference_route: of type np.ndarray, with rows of [(x ,y, theta)] where x, y, theta are floats
        """
        self._reference_route = _reference_route

    @property
    def reference_route(self): return self._reference_route

