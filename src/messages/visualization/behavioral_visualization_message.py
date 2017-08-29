import numpy as np

from decision_making.src.messages.dds_nontyped_message import DDSNonTypedMsg


class BehavioralVisualizationMsg(DDSNonTypedMsg):
    def __init__(self, reference_route):
        # type: (np.ndarray) -> None
        """
        The struct used for communicating the behavioral plan to the visualizer.
        :param reference_route: of type np.ndarray, with rows of [(x ,y, theta)] where x, y, theta are floats
        """
        self.reference_route = reference_route


