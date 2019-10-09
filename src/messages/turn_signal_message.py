from enum import Enum

from common_data.interface.Rte_Types.python.sub_structures.TsSYS_TurnSignal import TsSYSTurnSignal
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_DataTurnSignal import TsSYSDataTurnSignal

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Timestamp, Header


class TurnSignalState(Enum):
    """Turn Signal State Enum"""
    CeSYS_e_TurnNoActivation = 0
    CeSYS_e_TurnLeft = 1
    CeSYS_e_TurnRight = 2
    CeSYS_e_TurnIndeterminate = 3

class DataTurnSignal(PUBSUB_MSG_IMPL):
    e_e_TurnSignalState = TurnSignalState
    s_RecvTimestamp = Timestamp
    e_b_Valid = bool

    def __init__(self, e_e_TurnSignalState: TurnSignalState, s_RecvTimestamp: Timestamp, e_b_Valid: bool):
        """
        Turn Signal State
        """
        self.e_e_TurnSignalState = e_e_TurnSignalState
        self.s_RecvTimestamp = s_RecvTimestamp
        self.e_b_Valid = e_b_Valid

    def serialize(self) -> TsSYSDataTurnSignal:
        pubsub_msg = TsSYSDataTurnSignal()

        pubsub_msg.e_e_TurnSignalState = self.e_e_TurnSignalState
        pubsub_msg.s_RecvTimestamp = self.s_RecvTimestamp
        pubsub_msg.e_b_Valid = self.e_b_Valid

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataTurnSignal):
        return cls(pubsubMsg.e_e_TurnSignalState,
                   pubsubMsg.s_RecvTimestamp,
                   pubsubMsg.e_b_Valid)


class TurnSignal(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_Data = DataTurnSignal

    def __init__(self, s_Header: Header, s_Data: DataTurnSignal):
        """
        Class that represents the TurnSignal topic
        :param s_Header: General Information
        :param s_Data: Message Data
        """
        self.s_Header = s_Header
        self.s_Data = s_Data

    def serialize(self) -> TsSYSTurnSignal:
        pubsub_msg = TsSYSTurnSignal()

        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSTurnSignal):
        return cls(Header.deserialize(pubsubMsg.s_Header),
                   DataTurnSignal.deserialize(pubsubMsg.s_Data))
