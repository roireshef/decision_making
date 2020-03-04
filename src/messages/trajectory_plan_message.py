import numpy as np

from interface.Rte_Types.python.sub_structures.TsSYS_TrajectoryPlan import TsSYSTrajectoryPlan
from interface.Rte_Types.python.sub_structures.TsSYS_DataTrajectoryPlan import TsSYSDataTrajectoryPlan
from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL
from decision_making.src.global_constants import TRAJECTORY_WAYPOINT_SIZE
from decision_making.src.messages.scene_common_messages import Timestamp, MapOrigin, Header


class DataTrajectoryPlan(PUBSUB_MSG_IMPL):
    s_Timestamp = Timestamp
    creation_time = Timestamp
    physical_time = Timestamp
    s_MapOrigin = MapOrigin
    a_TrajectoryWaypoints = np.ndarray
    e_Cnt_NumValidTrajectoryWaypoints = int

    def __init__(self, s_Timestamp: Timestamp, creation_time: Timestamp, physical_time: Timestamp,
                 s_MapOrigin: MapOrigin, a_TrajectoryWaypoints: np.ndarray, e_Cnt_NumValidTrajectoryWaypoints: int):
        """

        :param s_Timestamp: Scene time (sensor time) based on which the planner planned the trajectory
        :param s_MapOrigin: The map origin used to represent the trajectory points according to
        :param a_TrajectoryWaypoints: A 2d array of desired host-vehicle states (localizations) in the future (100ms difference in time),
               with the first state being the current state (t=0). The array is implicitly index-able via the enum
               TeSYS_TrajectoryWaypoint that reflect field names of the last dimension of the array
        :param e_Cnt_NumValidTrajectoryWaypoints: number of valid points from the former parameter
        """
        self.s_Timestamp = s_Timestamp
        self.creation_time = creation_time
        self.physical_time = physical_time
        self.s_MapOrigin = s_MapOrigin
        self.a_TrajectoryWaypoints = a_TrajectoryWaypoints
        self.e_Cnt_NumValidTrajectoryWaypoints = e_Cnt_NumValidTrajectoryWaypoints

    def serialize(self):
        # type: () -> TsSYSDataTrajectoryPlan
        pubsub_msg = TsSYSDataTrajectoryPlan()

        pubsub_msg.s_Timestamp = self.s_Timestamp.serialize()
        pubsub_msg.s_DataCreationTime = self.creation_time.serialize()
        pubsub_msg.s_PhysicalEventTime = self.physical_time.serialize()
        pubsub_msg.s_MapOrigin = self.s_MapOrigin.serialize()
        pubsub_msg.a_TrajectoryWaypoints = self.a_TrajectoryWaypoints
        pubsub_msg.e_Cnt_NumValidTrajectoryWaypoints = self.e_Cnt_NumValidTrajectoryWaypoints

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSDataTrajectoryPlan)->DataTrajectoryPlan
        return cls(Timestamp.deserialize(pubsubMsg.s_Timestamp),
                   Timestamp.deserialize(pubsubMsg.s_DataCreationTime),
                   Timestamp.deserialize(pubsubMsg.s_PhysicalEventTime),
                   MapOrigin.deserialize(pubsubMsg.s_MapOrigin),
                   pubsubMsg.a_TrajectoryWaypoints[:pubsubMsg.e_Cnt_NumValidTrajectoryWaypoints,:TRAJECTORY_WAYPOINT_SIZE],
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
