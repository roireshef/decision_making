from decision_making.src.messages.scene_common_messages import Header, Timestamp
from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL
from interface.Rte_Types.python.sub_structures.TsSYS_DataSetSpeed import TsSYSDataSetSpeed
from interface.Rte_Types.python.sub_structures.TsSYS_SetSpeed import TsSYSSetSpeed


class DataSetSpeed(PUBSUB_MSG_IMPL):
    s_DataCreationTime = Timestamp
    s_PhysicalEventTime = Timestamp
    e_b_Valid = bool
    e_v_SpeedLimit = float
    e_i_lane_segment_id = int
    e_v_Offset = float
    e_v_SetSpeed = float

    def __init__(self, s_DataCreationTime: Timestamp, s_PhysicalEventTime: Timestamp, e_b_Valid: bool,
                 e_v_SpeedLimit: float, e_i_lane_segment_id: int, e_v_Offset: float, e_v_SetSpeed: float):
        """
        Data in SET_SPEED message
        :param s_DataCreationTime:
        :param s_PhysicalEventTime:
        :param e_b_Valid: Validity flag
        :param e_v_SpeedLimit:
        :param e_i_lane_segment_id:
        :param e_v_Offset:
        :param e_v_SetSpeed:
        """
        self.s_DataCreationTime = s_DataCreationTime
        self.s_PhysicalEventTime = s_PhysicalEventTime
        self.e_b_Valid = e_b_Valid
        self.e_v_SpeedLimit = e_v_SpeedLimit
        self.e_i_lane_segment_id = e_i_lane_segment_id
        self.e_v_Offset = e_v_Offset
        self.e_v_SetSpeed = e_v_SetSpeed

    def serialize(self) -> TsSYSDataSetSpeed:
        pubsub_msg = TsSYSDataSetSpeed()

        pubsub_msg.s_DataCreationTime = self.s_DataCreationTime.serialize()
        pubsub_msg.s_PhysicalEventTime = self.s_PhysicalEventTime.serialize()
        pubsub_msg.e_b_Valid = self.e_b_Valid
        pubsub_msg.e_v_SpeedLimit = self.e_v_SpeedLimit
        pubsub_msg.e_i_lane_segment_id = self.e_i_lane_segment_id
        pubsub_msg.e_v_Offset = self.e_v_Offset
        pubsub_msg.e_v_SetSpeed = self.e_v_SetSpeed

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataSetSpeed):
        return cls(Timestamp.deserialize(pubsubMsg.s_DataCreationTime),
                   Timestamp.deserialize(pubsubMsg.s_PhysicalEventTime),
                   pubsubMsg.e_b_Valid,
                   pubsubMsg.e_v_SpeedLimit,
                   pubsubMsg.e_i_lane_segment_id,
                   pubsubMsg.e_v_Offset,
                   pubsubMsg.e_v_SetSpeed)


class SetSpeed(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_Data = DataSetSpeed

    def __init__(self, s_Header: Header, s_Data: DataSetSpeed):
        """
        Class that represents the SET_SPEED topic
        :param s_Header: General Information
        :param s_Data: Message Data
        """
        self.s_Header = s_Header
        self.s_Data = s_Data

    def serialize(self) -> TsSYSSetSpeed:
        pubsub_msg = TsSYSSetSpeed()

        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSSetSpeed):
        return cls(Header.deserialize(pubsubMsg.s_Header),
                   DataSetSpeed.deserialize(pubsubMsg.s_Data))

