from interface.Rte_Types.python.sub_structures.TsSYS_Takeover import TsSYSTakeover
from interface.Rte_Types.python.sub_structures.TsSYS_DataTakeover import TsSYSDataTakeover

from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Header, Timestamp


class DataTakeover(PUBSUB_MSG_IMPL):
    e_b_is_takeover_needed = bool
    s_data_creation_time = Timestamp

    def __init__(self,
                 e_b_is_takeover_needed: bool,
                 s_data_creation_time: Timestamp):
        """
        Takeover Flag
        :param e_b_is_takeover_needed: true = takeover needed, false = takeover not needed
        :param s_data_creation_time:
        """
        self.e_b_is_takeover_needed = e_b_is_takeover_needed
        self.s_DataCreationTime = s_data_creation_time

    def serialize(self) -> TsSYSDataTakeover:
        pubsub_msg = TsSYSDataTakeover()

        pubsub_msg.e_b_is_takeover_needed = self.e_b_is_takeover_needed
        pubsub_msg.s_DataCreationTime = self.s_DataCreationTime.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataTakeover):
        return cls(
            e_b_is_takeover_needed=pubsubMsg.e_b_is_takeover_needed,
            s_data_creation_time=Timestamp.deserialize(pubsubMsg.s_DataCreationTime),
        )


class Takeover(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_Data = DataTakeover

    def __init__(self, s_Header: Header, s_Data: DataTakeover):
        """
        Class that represents the TAKEOVER topic
        :param s_Header: General Information
        :param s_Data: Message Data
        """
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
