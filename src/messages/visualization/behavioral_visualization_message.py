import numpy as np

from common_data.interface.py.idl_generated_files.dm import LcmBehavioralVisualizationMsg
from common_data.interface.py.idl_generated_files.dm.sub_structures.LcmNonTypedNumpyArray import LcmNonTypedNumpyArray
from decision_making.src.planning.types import CartesianPath2D


class BehavioralVisualizationMsg:
    def __init__(self, reference_route_points):
        # type: (CartesianPath2D) -> None
        """
        The struct used for communicating the behavioral plan to the visualizer.
        :param reference_route_points: of type CartesianPath2D
        """
        self.reference_route_points = reference_route_points

    def serialize(self) -> LcmBehavioralVisualizationMsg:
        lcm_msg = LcmBehavioralVisualizationMsg()

        lcm_msg.reference_route = LcmNonTypedNumpyArray()
        lcm_msg.reference_route.num_dimensions = len(self.reference_route_points.shape)
        lcm_msg.reference_route.shape = list(self.reference_route_points.shape)
        lcm_msg.reference_route.length = self.reference_route_points.size
        lcm_msg.reference_route.data = self.reference_route_points.flat.__array__().tolist()

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg: LcmBehavioralVisualizationMsg):
        return cls(np.ndarray(shape = tuple(lcmMsg.reference_route.shape[:lcmMsg.reference_route.num_dimensions])
                            , buffer = np.array(lcmMsg.reference_route.data)
                            , dtype = float))


