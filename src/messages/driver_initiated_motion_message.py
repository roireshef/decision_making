
from build.interface.Rte_Types.python.sub_structures.TsSYS_DataPedalPosition import TsSYSDataPedalPosition
from build.interface.Rte_Types.python.sub_structures.TsSYS_PedalPosition import TsSYSPedalPosition
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Header


class DataPedalPosition(PUBSUB_MSG_IMPL):
    e_Pct_BrakePedalPosition = float
    e_Pct_AcceleratorPedalPosition = float
    s_RecvTimestamp = float
    e_b_Valid = bool

    def __init__(self, s_RecvTimestamp: float, e_Pct_BrakePedalPosition: float, e_Pct_AcceleratorPedalPosition: float,
                 e_b_Valid: bool):
        """
        Initialize message containing brake and acceleration pedals position
        :param s_RecvTimestamp: NTP timestamp of when CAN message containing gear information reaches gateway sending this message
        :param e_Pct_BrakePedalPosition: Brake pedal position from 0.0 - 1.0; resolution: 0.00392157
        :param e_Pct_AcceleratorPedalPosition: Accelerator pedal position from 0.0 - 1.0; resolution: 0.00392157
        :param e_b_Valid: Validity of message, true if valid
        """
        self.e_Pct_BrakePedalPosition = e_Pct_BrakePedalPosition
        self.e_Pct_AcceleratorPedalPosition = e_Pct_AcceleratorPedalPosition
        self.s_RecvTimestamp = s_RecvTimestamp
        self.e_b_Valid = e_b_Valid

    def serialize(self) -> TsSYSDataPedalPosition:
        pubsub_msg = TsSYSDataPedalPosition()
        pubsub_msg.e_Pct_BrakePedalPosition = self.e_Pct_BrakePedalPosition
        pubsub_msg.e_Pct_AcceleratorPedalPosition = self.e_Pct_AcceleratorPedalPosition
        pubsub_msg.s_RecvTimestamp = self.s_RecvTimestamp
        pubsub_msg.e_b_Valid = self.e_b_Valid
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataPedalPosition):
        return cls(pubsubMsg.e_Pct_BrakePedalPosition, pubsubMsg.e_Pct_AcceleratorPedalPosition,
                   pubsubMsg.s_RecvTimestamp, pubsubMsg.e_b_Valid)


class PedalPosition(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_Data = DataPedalPosition

    def __init__(self, s_Header: Header, s_Data: DataPedalPosition):
        """
        Class that represents the ROUTE_PLAN topic
        :param s_Header: General Information
        :param s_Data: Message Data
        """
        self.s_Header = s_Header
        self.s_Data = s_Data

    def serialize(self) -> TsSYSPedalPosition:
        pubsub_msg = TsSYSPedalPosition()
        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSPedalPosition):
        return cls(pubsubMsg.s_Header, pubsubMsg.s_Data)
