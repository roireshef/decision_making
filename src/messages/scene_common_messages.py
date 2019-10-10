from interface.Rte_Types.python.sub_structures.TsSYS_Header import TsSYSHeader
from interface.Rte_Types.python.sub_structures.TsSYS_MapOrigin import TsSYSMapOrigin
from interface.Rte_Types.python.sub_structures.TsSYS_Timestamp import TsSYSTimestamp
from decision_making.src.global_constants import PUBSUB_MSG_IMPL


class Timestamp(PUBSUB_MSG_IMPL):
    e_Cnt_Secs = int
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

    @classmethod
    def from_seconds(cls, timestamp_in_sec: float):
        """
        wraps a timestamp (float, seconds) in a Timestamp message object
        :param timestamp_in_sec:
        :return:
        """
        timestamp_int = int(timestamp_in_sec)
        timestamp_frac = int((timestamp_in_sec-timestamp_int) * (1 << 32))
        return cls(e_Cnt_Secs=timestamp_int, e_Cnt_FractionSecs=timestamp_frac)

    @property
    def timestamp_in_seconds(self):
        return self.e_Cnt_Secs + self.e_Cnt_FractionSecs / (1 << 32)

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


class MapOrigin(PUBSUB_MSG_IMPL):
    e_phi_latitude = float
    e_phi_longitude = float
    e_l_altitude = float
    s_Timestamp = Timestamp

    def __init__(self, e_phi_latitude: float, e_phi_longitude: float, e_l_altitude: float, s_Timestamp: Timestamp) -> None:
        """
        All parameters are in ENU (east-north-up) coordinates, in [m] units
        """
        self.e_phi_latitude = e_phi_latitude
        self.e_phi_longitude = e_phi_longitude
        self.e_l_altitude = e_l_altitude
        self.s_Timestamp = s_Timestamp

    def serialize(self):
        # type: () -> TsSYSMapOrigin
        pubsub_msg = TsSYSMapOrigin()

        pubsub_msg.e_phi_latitude = self.e_phi_latitude
        pubsub_msg.e_phi_longitude = self.e_phi_longitude
        pubsub_msg.e_l_altitude = self.e_l_altitude
        pubsub_msg.s_Timestamp = self.s_Timestamp.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSMapOrigin)->MapOrigin
        return cls(pubsubMsg.e_phi_latitude, pubsubMsg.e_phi_longitude, pubsubMsg.e_l_altitude,
                   Timestamp.deserialize(pubsubMsg.s_Timestamp))


class Header(PUBSUB_MSG_IMPL):
    e_Cnt_SeqNum = int
    s_Timestamp = Timestamp
    e_Cnt_version = int

    def __init__(self, e_Cnt_SeqNum : int, s_Timestamp : Timestamp, e_Cnt_version : int) -> None:
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
