import numpy as np
from typing import List

from common_data.interface.py.idl_generated_files.Rte_Types.TsSYS_Takeover import TsSYSTakeover
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_DataTakeover import TsSYSDataTakeover

from decision_making.src.messages.scene_common_messages import Header
from decision_making.src.global_constants import PUBSUB_MSG_IMPL

class DataTakeover(PUBSUB_MSG_IMPL):
    """
    Takeover Flag

    Args:
        e_b_is_takeover_needed: true = takeover needed,
                                false = takeover not needed
    """
    e_b_is_takeover_needed = bool

    def __init__(self, e_b_is_takeover_needed: bool):
        self.e_b_is_takeover_needed = e_b_is_takeover_needed

    def serialize(self) -> TsSYSDataTakeover:
        pubsub_msg = TsSYSDataTakeover()

        pubsub_msg.e_b_is_takeover_needed = self.e_b_is_takeover_needed

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataTakeover):
        return cls(pubsubMsg.e_b_is_takeover_needed)

class Takeover(PUBSUB_MSG_IMPL):
    """
    Class that represents the TAKEOVER topic
    
    Args:
        s_Header: TODO: Add Comment
        s_Data: TODO: Add Comment
    """
    s_Header = Header
    s_Data = DataTakeover

    def __init__(self, s_Header: Header, s_Data: DataTakeover):
        self.s_Header = s_Header
        self.s_Data = s_Data

    def serialize(self) -> TsSYSTakeover:
        pubsub_msg = TsSYSTakeover()

        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSTakeover):
        return cls(Header.deserialize(pubsubMsg.s_Header),
                   DataTakeover.deserialize(pubsubMsg.s_Data))
