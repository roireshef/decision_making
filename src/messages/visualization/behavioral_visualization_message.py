from common_data.interface.Rte_Types.python.sub_structures.TsSYS_BehavioralVisualizationMsg import TsSYSBehavioralVisualizationMsg
from common_data.interface.py.utils.serialization_utils import SerializationUtils
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import CartesianPath2D


class BehavioralVisualizationMsg(PUBSUB_MSG_IMPL):
    def __init__(self, reference_route_points: CartesianPath2D) -> None:
        """
        The struct used for communicating the behavioral plan to the visualizer.
        :param reference_route_points: of type CartesianPath2D
        """
        self.s_ReferenceRoute = reference_route_points

    def serialize(self) -> TsSYSBehavioralVisualizationMsg:
        pubsub_msg = TsSYSBehavioralVisualizationMsg()
        pubsub_msg.s_ReferenceRoute = SerializationUtils.serialize_non_typed_array(self.s_ReferenceRoute)
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSBehavioralVisualizationMsg):
        return cls(SerializationUtils.deserialize_any_array(pubsubMsg.s_ReferenceRoute))


