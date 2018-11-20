import numpy as np

from common_data.interface.py.idl_generated_files.dm import LcmBehavioralVisualizationMsg
from common_data.interface.py.utils.serialization_utils import SerializationUtils
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import CartesianPath2D


class BehavioralVisualizationMsg(PUBSUB_MSG_IMPL):
    def __init__(self, reference_route_points):
        # type: (CartesianPath2D) -> None
        """
        The struct used for communicating the behavioral plan to the visualizer.
        :param reference_route_points: of type CartesianPath2D
        """
        self.reference_route_points = reference_route_points

    def serialize(self) -> LcmBehavioralVisualizationMsg:
        lcm_msg = LcmBehavioralVisualizationMsg()
        lcm_msg.reference_route = SerializationUtils.serialize_non_typed_array(self.reference_route_points)
        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg: LcmBehavioralVisualizationMsg):
        return cls(SerializationUtils.deserialize_any_array(lcmMsg.reference_route))


