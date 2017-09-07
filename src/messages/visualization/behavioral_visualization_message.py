import numpy as np

from decision_making.src.messages.dds_typed_message import DDSTypedMsg


class BehavioralVisualizationMsg(DDSTypedMsg):
    def __init__(self, reference_route):
        # type: (np.ndarray) -> None
        """
        The struct used for communicating the behavioral plan to the visualizer.
        :param reference_route: of type np.ndarray, with rows of [(x ,y, theta)] where x, y, theta are floats
        """
        self.reference_route = reference_route


