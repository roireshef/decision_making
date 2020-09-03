from enum import Enum

import numpy as np
from decision_making.src.messages.scene_common_messages import Header, Timestamp
from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL
from interface.Rte_Types.python.sub_structures.TsSYS_DataGapSetting import TsSYSDataGapSetting
from interface.Rte_Types.python.sub_structures.TsSYS_GapSetting import TsSYSGapSetting


class GapSettingState(Enum):
    CeSYS_e_Far = 0
    CeSYS_e_Medium = 1
    CeSYS_e_Close = 2


class DataGapSetting(PUBSUB_MSG_IMPL):
    e_b_valid = bool
    s_RecvTimestamp = Timestamp
    s_DataCreationTime = Timestamp
    s_PhysicalEventTime = Timestamp
    e_b_RawGapDownPressed = bool
    e_b_RawGapUpPressed = bool
    e_e_gap_setting_state = GapSettingState
    a_Reserved = np.ndarray

    def __init__(self, e_b_valid: bool, s_RecvTimestamp: Timestamp, s_DataCreationTime: Timestamp,
                 s_PhysicalEventTime: Timestamp, e_b_RawGapDownPressed: bool, e_b_RawGapUpPressed: bool,
                 e_e_gap_setting_state: GapSettingState, a_Reserved: np.ndarray):
        """
        Data in GAP_SETTING message
        :param e_b_valid: Validity flag
        :param s_RecvTimestamp: Receive Timestamp representing when the data message frame was received by the driver, NTP format
        :param s_DataCreationTime:
        :param s_PhysicalEventTime:
        :param e_b_RawGapDownPressed:
        :param e_b_RawGapUpPressed:
        :param e_e_gap_setting_state:
        :param a_Reserved: Reserved
        """
        self.e_b_valid = e_b_valid
        self.s_RecvTimestamp = s_RecvTimestamp
        self.s_DataCreationTime = s_DataCreationTime
        self.s_PhysicalEventTime = s_PhysicalEventTime
        self.e_b_RawGapDownPressed = e_b_RawGapDownPressed
        self.e_b_RawGapUpPressed = e_b_RawGapUpPressed
        self.e_e_gap_setting_state = e_e_gap_setting_state
        self.a_Reserved = a_Reserved

    def serialize(self) -> TsSYSDataGapSetting:
        pubsub_msg = TsSYSDataGapSetting()

        pubsub_msg.e_b_valid = self.e_b_valid
        pubsub_msg.s_RecvTimestamp = self.s_RecvTimestamp.serialize()
        pubsub_msg.s_DataCreationTime = self.s_DataCreationTime.serialize()
        pubsub_msg.s_PhysicalEventTime = self.s_PhysicalEventTime.serialize()
        pubsub_msg.e_b_RawGapDownPressed = self.e_b_RawGapDownPressed
        pubsub_msg.e_b_RawGapUpPressed = self.e_b_RawGapUpPressed
        pubsub_msg.e_e_gap_setting_state = self.e_e_gap_setting_state.value
        pubsub_msg.a_Reserved = self.a_Reserved

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataGapSetting):
        return cls(pubsubMsg.e_b_valid,
                   Timestamp.deserialize(pubsubMsg.s_RecvTimestamp),
                   Timestamp.deserialize(pubsubMsg.s_DataCreationTime),
                   Timestamp.deserialize(pubsubMsg.s_PhysicalEventTime),
                   pubsubMsg.e_b_RawGapDownPressed,
                   pubsubMsg.e_b_RawGapUpPressed,
                   GapSettingState(pubsubMsg.e_e_gap_setting_state),
                   pubsubMsg.a_Reserved)


class GapSetting(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_Data = DataGapSetting

    def __init__(self, s_Header: Header, s_Data: DataGapSetting):
        """
        Class that represents the Gap_SETTING topic
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
                   DataGapSetting.deserialize(pubsubMsg.s_Data))

