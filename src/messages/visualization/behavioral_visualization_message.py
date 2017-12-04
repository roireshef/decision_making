import numpy as np

from common_data.lcm.generatedFiles.gm_lcm import LcmBehavioralVisualizationMsg
from common_data.lcm.generatedFiles.gm_lcm import LcmNonTypedNumpyArray

class BehavioralVisualizationMsg:
    def __init__(self, reference_route):
        # type: (np.ndarray) -> None
        """
        The struct used for communicating the behavioral plan to the visualizer.
        :param reference_route: of type np.ndarray, with rows of [(x ,y, theta)] where x, y, theta are floats
        """
        self.reference_route = reference_route

    def to_lcm(self) -> LcmBehavioralVisualizationMsg:
        lcm_msg = LcmBehavioralVisualizationMsg()

        lcm_msg.reference_route = LcmNonTypedNumpyArray()
        lcm_msg.reference_route.num_dimensions = len(self.reference_route.shape)
        lcm_msg.reference_route.shape = list(self.reference_route.shape)
        lcm_msg.reference_route.length = self.reference_route.size
        lcm_msg.reference_route.data = self.reference_route.flat.__array__().tolist()

        return lcm_msg

    @classmethod
    def from_lcm(cls, lcmMsg: LcmBehavioralVisualizationMsg):
        return cls(np.ndarray(shape = tuple(lcmMsg.reference_route.shape)
                            , buffer = np.array(lcmMsg.reference_route.data)
                            , dtype = float))


