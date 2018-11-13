import numpy as np

from Rte_Types import TsSYSTrajectoryPlan
from Rte_Types.sub_structures import TsSYSHeader, TsSYSMapOrigin, TsSYSTimestamp, TsSYSDataTrajectoryPlan
from common_data.interface.py.utils.serialization_utils import SerializationUtils
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


class MapOrigin(PUBSUB_MSG_IMPL):
    e_phi_latitude = float
    e_phi_longitude = float
    e_l_altitude = float
    s_Timestamp = Timestamp

    def __init__(self, e_phi_latitude, e_phi_longitude, e_l_altitude, s_Timestamp):
        # type: (float, float, float, Timestamp) -> None
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


class DataTrajectoryPlan(PUBSUB_MSG_IMPL):
    s_Timestamp = Timestamp
    s_MapOrigin = MapOrigin
    a_TrajectoryWaypoints = np.ndarray
    e_Cnt_NumValidTrajectoryWaypoints = int

    def __init__(self, s_Timestamp: Timestamp, s_MapOrigin: MapOrigin, a_TrajectoryWaypoints: np.ndarray,
                 e_Cnt_NumValidTrajectoryWaypoints: int):
        """

        :param s_Timestamp: Scene time (sensor time) based on which the planner planned the trajectory
        :param s_MapOrigin: The map origin used to represent the trajectory points according to
        :param a_TrajectoryWaypoints: A 2d array of desired host-vehicle states (localizations) in the future (100ms difference in time),
               with the first state being the current state (t=0). The array is implicitly index-able via the enum
               TeSYS_TrajectoryWaypoint that reflect field names of the last dimension of the array
        :param e_Cnt_NumValidTrajectoryWaypoints: number of valid points from the former parameter
        """
        self.s_Timestamp = s_Timestamp
        self.s_MapOrigin = s_MapOrigin
        self.a_TrajectoryWaypoints = a_TrajectoryWaypoints
        self.e_Cnt_NumValidTrajectoryWaypoints = e_Cnt_NumValidTrajectoryWaypoints

    def serialize(self):
        # type: () -> TsSYSDataTrajectoryPlan
        pubsub_msg = TsSYSDataTrajectoryPlan()

        pubsub_msg.s_Timestamp = self.s_Timestamp.serialize()
        pubsub_msg.s_MapOrigin = self.s_MapOrigin.serialize()
        pubsub_msg.a_TrajectoryWaypoints = SerializationUtils.serialize_array(self.a_TrajectoryWaypoints)
        pubsub_msg.e_Cnt_NumValidTrajectoryWaypoints = self.e_Cnt_NumValidTrajectoryWaypoints

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSDataTrajectoryPlan)->DataTrajectoryPlan
        return cls(Timestamp.deserialize(pubsubMsg.s_Timestamp),
                   MapOrigin.deserialize(pubsubMsg.s_MapOrigin),
                   SerializationUtils.deserialize_any_array(pubsubMsg.a_TrajectoryWaypoints),
                   pubsubMsg.e_Cnt_NumValidTrajectoryWaypoints)


class TrajectoryPlan(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_Data = DataTrajectoryPlan

    def __init__(self, s_Header: Header, s_Data: DataTrajectoryPlan):
        """

        :param s_Header:
        :param s_Data:
        """
        self.s_Header = s_Header
        self.s_Data = s_Data

    def serialize(self):
        # type: () -> TsSYSTrajectoryPlan
        pubsub_msg = TsSYSTrajectoryPlan()

        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSTrajectoryPlan)->TrajectoryPlan
        return cls(Header.deserialize(pubsubMsg.s_Header),
                   DataTrajectoryPlan.deserialize(pubsubMsg.s_Data))































# class TrajectoryPlanMsg(PUBSUB_MSG_IMPL):
#     ''' Members annotations for python 2 compliant classes '''
#     timestamp = int
#     trajectory = np.ndarray
#     current_speed = float
#
#     def __init__(self, timestamp, trajectory, current_speed):
#         # type: (int, np.ndarray, float) -> None
#         """
#         A discrete representation of the trajectory to follow - passed from TrajectoryPlanner to Controller
#         :param timestamp: ego's timestamp on which the trajectory is based. Used for giving the controller a reference
#         between the planned trajectory at current time, and the part to be executed due to time delays in the system.
#         :param trajectory: numpy 2D array - 9 rows with each row containing <x, y, yaw, curvature, v> where x and y
#          values are given in global coordinate frame, yaw and curvature are currently only placeholders, and v is the
#          desired velocity of the vehicle on its actual longitudinal axis (the direction of its heading).
#          Note: the first points corresponds to the current pose of the ego-vehicle at the time of plan, and only the
#          following 8 points are points to follow further.
#         :param current_speed: the current speed of the ego vehicle at the time of plan
#         """
#         self.timestamp = timestamp
#         self.trajectory = trajectory
#         self.current_speed = current_speed
#
#     def serialize(self):
#         # type: () -> LcmTrajectoryData
#         lcm_msg = LcmTrajectoryData()
#
#         lcm_msg.timestamp = self.timestamp
#         lcm_msg.trajectory = LcmNumpyArray()
#         lcm_msg.trajectory.num_dimensions = len(self.trajectory.shape)
#         lcm_msg.trajectory.shape = list(self.trajectory.shape)
#         lcm_msg.trajectory.length = self.trajectory.size
#         lcm_msg.trajectory.data = self.trajectory.flat.__array__().tolist()
#
#         lcm_msg.current_speed = self.current_speed
#
#         return lcm_msg
#
#     @classmethod
#     def deserialize(cls, lcmMsg):
#         # type: (LcmTrajectoryData) -> TrajectoryPlanMsg
#
#         return cls(lcmMsg.timestamp,
#                    np.ndarray(shape=tuple(lcmMsg.trajectory.shape)
#                               , buffer=np.array(lcmMsg.trajectory.data)
#                               , dtype=float)
#                    , lcmMsg.current_speed)
