from enum import Enum
import numpy as np

from interface.Rte_Types.python.sub_structures.TsSYS_TurnSignal import TsSYSTurnSignal
from interface.Rte_Types.python.sub_structures.TsSYS_DataTurnSignal import TsSYSDataTurnSignal

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Header, Timestamp


class TurnSignalState(Enum):
    CeSYS_e_Unknown = 0
    CeSYS_e_Off = 1
    CeSYS_e_LeftTurnSignalOn = 2
    CeSYS_e_RightTurnSignalOn = 3


class DataTurnSignal(PUBSUB_MSG_IMPL):
    e_b_valid = bool
    s_RecvTimestamp = Timestamp
    s_time_changed = Timestamp
    e_e_turn_signal_state = TurnSignalState
    a_Reserved = np.ndarray

    def __init__(self, e_b_valid: bool, s_RecvTimestamp: Timestamp, s_time_changed: Timestamp,
                 e_e_turn_signal_state: TurnSignalState, a_Reserved: np.ndarray):
        """
        Data in TURN_SIGNAL message
        :param e_b_valid: Validity flag
        :param s_RecvTimestamp: Receive Timestamp representing when the data message frame was received by the driver, NTP format
        :param s_time_changed: Time that the turn signal changed state
        :param e_e_turn_signal_state: Turn signal state
        :param a_Reserved: Reserved
        """
        self.e_b_valid = e_b_valid
        self.s_RecvTimestamp = s_RecvTimestamp
        self.s_time_changed = s_time_changed
        self.e_e_turn_signal_state = e_e_turn_signal_state
        self.a_Reserved = a_Reserved

    def serialize(self) -> TsSYSDataTurnSignal:
        pubsub_msg = TsSYSDataTurnSignal()

        pubsub_msg.e_b_valid = self.e_b_valid
        pubsub_msg.s_RecvTimestamp = self.s_RecvTimestamp.serialize()
        pubsub_msg.s_time_changed = self.s_time_changed.serialize()
        pubsub_msg.e_e_turn_signal_state = self.e_e_turn_signal_state.value
        pubsub_msg.a_Reserved = self.a_Reserved

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataTurnSignal):
        return cls(pubsubMsg.e_b_valid,
                   Timestamp.deserialize(pubsubMsg.s_RecvTimestamp),
                   Timestamp.deserialize(pubsubMsg.s_time_changed),
                   TurnSignalState(pubsubMsg.e_e_turn_signal_state),
                   pubsubMsg.a_Reserved)


class TurnSignal(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_Data = DataTurnSignal

    def __init__(self, s_Header: Header, s_Data: DataTurnSignal):
        """
        Class that represents the TURN_SIGNAL topic
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

DEFAULT_MSG = TurnSignal(s_Header=None, s_Data=DataTurnSignal(e_b_valid=False, s_RecvTimestamp=None,
                                                              s_time_changed=None,
                                                              e_e_turn_signal_state=TurnSignalState.CeSYS_e_Off,
                                                              a_Reserved=None))
