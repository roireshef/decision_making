from enum import Enum
import numpy as np

from interface.Rte_Types.python.sub_structures.TsSYS_GapSetting import TsSYSGapSetting
from interface.Rte_Types.python.sub_structures.TsSYS_DataGapSetting import TsSYSDataGapSetting

from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Header, Timestamp


class GapSettingState(Enum):
    """
    Gap setting, to be set by the driver. Affects headway that will be kept to the car ahead.
    """
    CeSYS_e_Far = 0
    CeSYS_e_Medium = 1
    CeSYS_e_Close = 2


class DataGapSetting(PUBSUB_MSG_IMPL):
    e_b_valid = bool
    s_RecvTimestamp = Timestamp
    e_b_RawGapDownPressed = bool
    e_b_RawGapUpPressed = bool
    e_e_gap_setting_state = GapSettingState
    a_Reserved = np.ndarray

    def __init__(self, e_b_valid: bool, s_RecvTimestamp: Timestamp, e_b_RawGapDownPressed: bool,
                 e_b_RawGapUpPressed: bool, e_e_gap_setting_state:GapSettingState, a_Reserved: np.ndarray):
        """
        Data in GapSetting Message
        :param e_b_valid: Validity flag
        :param s_RecvTimestamp: Receive Timestamp representing when the data message frame was received by the driver, NTP format
        :param e_b_RawGapDownPressed: Raw switch data from CAN
        :param e_b_RawGapUpPressed: Raw switch data from CAN (not in current Escalades)
        :param e_e_gap_setting_state: Gap Setting State
        :param a_Reserved: Reserved
        """
        self.e_b_valid = e_b_valid
        self.s_RecvTimestamp = s_RecvTimestamp
        self.e_b_RawGapDownPressed = e_b_RawGapDownPressed
        self.e_b_RawGapUpPressed = e_b_RawGapUpPressed
        self.e_e_gap_setting_state = e_e_gap_setting_state
        self.a_Reserved = a_Reserved

    def serialize(self) -> TsSYSDataGapSetting:
        pubsub_msg = TsSYSDataGapSetting()

        pubsub_msg.e_b_valid = self.e_b_valid
        pubsub_msg.s_RecvTimestamp = self.s_RecvTimestamp.serialize()
        pubsub_msg.e_b_RawGapDownPressed = self.e_b_RawGapDownPressed
        pubsub_msg.e_b_RawGapUpPressed = self.e_b_RawGapUpPressed
        pubsub_msg.e_e_gap_setting_state = self.e_e_gap_setting_state.value
        pubsub_msg.a_Reserved = self.a_Reserved

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataGapSetting):
        return cls(pubsubMsg.e_b_valid,
                   Timestamp.deserialize(pubsubMsg.s_RecvTimestamp),
                   pubsubMsg.e_b_RawGapDownPressed,
                   pubsubMsg.e_b_RawGapUpPressed,
                   GapSettingState(pubsubMsg.e_e_gap_setting_state),
                   pubsubMsg.a_Reserved)


class GapSetting(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_Data = DataGapSetting

    def __init__(self, s_Header: Header, s_Data: DataGapSetting):
        """
        Class that represents the TURN_SIGNAL topic
        :param s_Header: General Information
        :param s_Data: Message Data
        """
        self.s_Header = s_Header
        self.s_Data = s_Data

    def serialize(self) -> TsSYSGapSetting:
        pubsub_msg = TsSYSGapSetting()

        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSGapSetting):
        return cls(Header.deserialize(pubsubMsg.s_Header),
                   TsSYSGapSetting.deserialize(pubsubMsg.s_Data))

DEFAULT_MSG = GapSetting(s_Header=None, s_Data=DataGapSetting(e_b_valid=False, s_RecvTimestamp=None,
                                                              e_b_RawGapDownPressed=False,
                                                              e_b_RawGapUpPressed=False,
                                                              e_e_gap_setting_state=GapSettingState.CeSYS_e_Medium,
                                                              a_Reserved=None))
