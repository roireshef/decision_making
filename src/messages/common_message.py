from Rte_Types.sub_structures import TsSYSTimestamp, TsSYSHeader
from decision_making.src.global_constants import PUBSUB_MSG_IMPL


class Timestamp(PUBSUB_MSG_IMPL):
    e_Cnt_Secs = int
    # TODO: why fractions are int?
    e_Cnt_FractionSecs = int

    def __init__(self, e_Cnt_Secs, e_Cnt_FractionSecs):
        # type: (int, int)->None
        """
        A data class that corresponds to a parametrization of a sigmoid function
        :param e_Cnt_Secs: Seconds since 1 January 1900
        :param e_Cnt_FractionSecs: Fractional seconds
        """
        self.e_Cnt_Secs = e_Cnt_Secs
        self.e_Cnt_FractionSecs = e_Cnt_FractionSecs

    def serialize(self):
        # type: () -> TsSYSTimestamp
        pubsub_msg = TsSYSTimestamp()

        pubsub_msg.e_Cnt_Secs = self.e_Cnt_Secs
        pubsub_msg.e_Cnt_FractionSecs = self.e_Cnt_FractionSecs

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSTimestamp)->Timestamp
        return cls(pubsubMsg.e_Cnt_Secs, pubsubMsg.e_Cnt_FractionSecs)


class Header(PUBSUB_MSG_IMPL):
    e_Cnt_SeqNum = int
    s_Timestamp = Timestamp
    e_Cnt_version = int

    def __init__(self, e_Cnt_SeqNum, s_Timestamp, e_Cnt_version):
        # type: (int, Timestamp, int)->None
        """
        Header Information is controlled by Middleware
        :param e_Cnt_SeqNum: Starts from 0 and increments at every update of this data structure
        :param s_Timestamp: Timestamp in secs and nano seconds when the data was published
        :param e_Cnt_version: Version of the topic/service used to identify interface compatability
        :return:
        """
        self.e_Cnt_SeqNum = e_Cnt_SeqNum
        self.s_Timestamp = s_Timestamp
        self.e_Cnt_version = e_Cnt_version

    def serialize(self):
        # type: () -> TsSYSHeader
        pubsub_msg = TsSYSHeader()

        pubsub_msg.e_Cnt_SeqNum = self.e_Cnt_SeqNum
        pubsub_msg.s_Timestamp = self.s_Timestamp.serialize()
        pubsub_msg.e_Cnt_version = self.e_Cnt_version

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSHeader)->Header
        return cls(pubsubMsg.e_Cnt_SeqNum, Timestamp.deserialize(pubsubMsg.s_Timestamp), pubsubMsg.e_Cnt_version)
