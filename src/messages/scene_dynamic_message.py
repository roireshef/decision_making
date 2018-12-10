from enum import Enum
from typing import List

import numpy as np

from common_data.interface.py.idl_generated_files.Rte_Types.TsSYS_SceneDynamic import TsSYSSceneDynamic
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_BoundingBoxSize import \
    TsSYSBoundingBoxSize
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_DataSceneDynamic import \
    TsSYSDataSceneDynamic
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_DataSceneHost import TsSYSDataSceneHost
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_HostLocalization import \
    TsSYSHostLocalization
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_ObjectHypothesis import \
    TsSYSObjectHypothesis
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_ObjectLocalization import \
    TsSYSObjectLocalization
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Timestamp, Header, MapOrigin

MAX_CARTESIANPOSE_FIELDS = 6
MAX_LANEFRENETPOSE_FIELDS = 6


class ObjectTrackDynamicProperty(Enum):
    """"
    Track Dynamic Property ENUM
    """
    CeSYS_e_ObjectTrackDynProp_Unknown = 0  # Unknown
    CeSYS_e_ObjectTrackDynProp_HasNeverMoved = 1  # Has Never Moved
    CeSYS_e_ObjectTrackDynProp_HasMovedButCurrentlyStopped = 2  # Has Moved but now stopped
    CeSYS_e_ObjectTrackDynProp_MovingInSameDirAsHost = 3  # Moving in same direction as host
    CeSYS_e_ObjectTrackDynProp_MovingInOppDir = 4  # Moving in opposite direction as host


class ObjectClassification(Enum):
    """"
    Detections type ENUM
    """
    CeSYS_e_ObjectClassification_Car = 0
    CeSYS_e_ObjectClassification_Truck = 1
    CeSYS_e_ObjectClassification_Bike = 2
    CeSYS_e_ObjectClassification_Bicycle = 3
    CeSYS_e_ObjectClassification_Pedestrian = 4
    CeSYS_e_ObjectClassification_GeneralObject = 5
    CeSYS_e_ObjectClassification_Animal = 6
    CeSYS_e_ObjectClassification_UNKNOWN = 7


class CartesianPose(Enum):
    CeSYS_CartesianPose_e_l_EastX = 0
    CeSYS_CartesianPose_e_l_NorthY = 1
    CeSYS_CartesianPose_e_phi_heading = 2
    CeSYS_CartesianPose_e_v_velocity = 3
    CeSYS_CartesianPose_e_a_acceleration = 4
    CeSYS_CartesianPose_e_il_curvature = 5


class LaneFrenetPose(Enum):
    CeSYS_LaneFrenetPose_e_l_s = 0
    CeSYS_LaneFrenetPose_e_v_s_dot = 1
    CeSYS_LaneFrenetPose_e_a_s_dotdot = 2
    CeSYS_LaneFrenetPose_e_l_d = 3
    CeSYS_LaneFrenetPose_e_v_d_dot = 4
    CeSYS_LaneFrenetPose_e_a_d_dotdot = 5


class HostLocalization(PUBSUB_MSG_IMPL):
    e_Cnt_road_segment_id = int
    e_Cnt_lane_segment_id = int
    a_cartesian_pose = np.ndarray
    a_lane_frenet_pose = np.ndarray

    def __init__(self, e_Cnt_road_segment_id, e_Cnt_lane_segment_id, a_cartesian_pose, a_lane_frenet_pose):
        # type: (int, int, np.ndarray, np.ndarray)->None
        """
        Host-localization information
        :param e_Cnt_road_segment_id: The ID of the road-segment that the host is in
        :param e_Cnt_lane_segment_id: The ID of the lane-segment that the host is in
        :param a_cartesian_pose: The host's pose, expressed in the Map (ENU) frame
        :param a_lane_frenet_pose: The host's pose, expressed in the the Frenet-Serret frame of the host's lane-segment
        """
        self.e_Cnt_road_segment_id = e_Cnt_road_segment_id
        self.e_Cnt_lane_segment_id = e_Cnt_lane_segment_id
        self.a_cartesian_pose = a_cartesian_pose
        self.a_lane_frenet_pose = a_lane_frenet_pose

    def serialize(self):
        # type: () -> TsSYSHostLocalization
        pubsub_msg = TsSYSHostLocalization()

        pubsub_msg.e_Cnt_road_segment_id = self.e_Cnt_road_segment_id
        pubsub_msg.e_Cnt_lane_segment_id = self.e_Cnt_lane_segment_id

        pubsub_msg.a_cartesian_pose = self.a_cartesian_pose
        pubsub_msg.a_lane_frenet_pose = self.a_lane_frenet_pose

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSHostLocalization)->HostLocalization
        return cls(pubsubMsg.e_Cnt_road_segment_id, pubsubMsg.e_Cnt_lane_segment_id,
                   pubsubMsg.a_cartesian_pose[:MAX_CARTESIANPOSE_FIELDS],
                   pubsubMsg.a_lane_frenet_pose[:MAX_LANEFRENETPOSE_FIELDS])


class DataSceneHost(PUBSUB_MSG_IMPL):

    e_b_Valid = bool
    s_ComputeTimestamp = Timestamp
    s_host_localization = HostLocalization

    def __init__(self, e_b_Valid, s_ComputeTimestamp, s_host_localization):
        # type: (bool, Timestamp, HostLocalization)->None
        """
        Scene provider's information on host vehicle pose
        :param e_b_Valid:
        :param s_ComputeTimestamp:
        :param s_host_localization:
        """
        self.e_b_Valid = e_b_Valid
        self.s_ComputeTimestamp = s_ComputeTimestamp
        self.s_host_localization = s_host_localization

    def serialize(self):
        # type: () -> TsSYSDataSceneHost
        pubsub_msg = TsSYSDataSceneHost()

        pubsub_msg.e_b_Valid = self.e_b_Valid
        pubsub_msg.s_ComputeTimestamp = self.s_ComputeTimestamp.serialize()
        pubsub_msg.s_host_localization = self.s_host_localization.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSDataSceneHost)->DataSceneHost
        return cls(pubsubMsg.e_b_Valid, Timestamp.deserialize(pubsubMsg.s_ComputeTimestamp),
                   HostLocalization.deserialize(pubsubMsg.s_host_localization))


class BoundingBoxSize(PUBSUB_MSG_IMPL):
    e_l_length = float
    e_l_width = float
    e_l_height = float

    def __init__(self, e_l_length, e_l_width, e_l_height):
        # type: (float, float, float)->None
        self.e_l_length = e_l_length
        self.e_l_width = e_l_width
        self.e_l_height = e_l_height

    def serialize(self):
        # type: () -> TsSYSBoundingBoxSize
        pubsub_msg = TsSYSBoundingBoxSize()

        pubsub_msg.e_l_length = self.e_l_length
        pubsub_msg.e_l_width = self.e_l_width
        pubsub_msg.e_l_height = self.e_l_height

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSBoundingBoxSize)->BoundingBoxSize
        return cls(pubsubMsg.e_l_length, pubsubMsg.e_l_width, pubsubMsg.e_l_height)


class ObjectHypothesis(PUBSUB_MSG_IMPL):
    e_r_probability = float
    e_Cnt_lane_segment_id = int
    e_e_dynamic_status = ObjectTrackDynamicProperty
    e_Pct_location_uncertainty_x = float
    e_Pct_location_uncertainty_y = float
    e_Pct_location_uncertainty_yaw = float
    e_Cnt_host_lane_frenet_id = int
    a_cartesian_pose = np.ndarray
    a_lane_frenet_pose = np.ndarray
    a_host_lane_frenet_pose = np.ndarray

    def __init__(self, e_r_probability, e_Cnt_lane_segment_id, e_e_dynamic_status, e_Pct_location_uncertainty_x,
                 e_Pct_location_uncertainty_y, e_Pct_location_uncertainty_yaw, e_Cnt_host_lane_frenet_id,
                 a_cartesian_pose, a_lane_frenet_pose, a_host_lane_frenet_pose):
        # type: (float, int, ObjectTrackDynamicProperty, float, float, float, int, np.ndarray, np.ndarray, np.ndarray) -> None
        """
        Actors-hypotheses information
        :param e_r_probability: Probability of this hypothesis (not relevant for M0)
        :param e_Cnt_lane_segment_id: The lane-segment ID that this actor-hypothesis is in
        :param e_e_dynamic_status:
        :param e_Pct_location_uncertainty_x: Not relevant for M0
        :param e_Pct_location_uncertainty_y: Not relevant for M0
        :param e_Pct_location_uncertainty_yaw: Not relevant for M0
        :param e_Cnt_host_lane_frenet_id: The ID of the lane-segment that the host is in
        :param a_cartesian_pose: The pose of this actor-hypothesis, expressed in the Map (ENU) frame
        :param a_lane_frenet_pose: The pose of this actor-hypothesis, expressed in the Frenet-Serret frame of its own lane-segment
        :param a_host_lane_frenet_pose: The pose of this actor-hypothesis, expressed in the Frenet-Serret frame of the host's lane-segment
        """
        self.e_r_probability = e_r_probability
        self.e_Cnt_lane_segment_id = e_Cnt_lane_segment_id
        self.e_e_dynamic_status = e_e_dynamic_status
        self.e_Pct_location_uncertainty_x = e_Pct_location_uncertainty_x
        self.e_Pct_location_uncertainty_y = e_Pct_location_uncertainty_y
        self.e_Pct_location_uncertainty_yaw = e_Pct_location_uncertainty_yaw
        self.e_Cnt_host_lane_frenet_id = e_Cnt_host_lane_frenet_id
        self.a_cartesian_pose = a_cartesian_pose
        self.a_lane_frenet_pose = a_lane_frenet_pose
        self.a_host_lane_frenet_pose = a_host_lane_frenet_pose

    def serialize(self):
        # type: () -> TsSYSObjectHypothesis
        pubsub_msg = TsSYSObjectHypothesis()

        pubsub_msg.e_r_probability = self.e_r_probability
        pubsub_msg.e_Cnt_lane_segment_id = self.e_Cnt_lane_segment_id
        pubsub_msg.e_e_dynamic_status = self.e_e_dynamic_status.value
        pubsub_msg.e_Pct_location_uncertainty_x = self.e_Pct_location_uncertainty_x
        pubsub_msg.e_Pct_location_uncertainty_y = self.e_Pct_location_uncertainty_y
        pubsub_msg.e_Pct_location_uncertainty_yaw = self.e_Pct_location_uncertainty_yaw
        pubsub_msg.e_Cnt_host_lane_frenet_id = self.e_Cnt_host_lane_frenet_id

        pubsub_msg.a_cartesian_pose = self.a_cartesian_pose
        pubsub_msg.a_lane_frenet_pose = self.a_lane_frenet_pose
        pubsub_msg.a_host_lane_frenet_pose = self.a_host_lane_frenet_pose

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSObjectHypothesis)->ObjectHypothesis
        return cls(pubsubMsg.e_r_probability, pubsubMsg.e_Cnt_lane_segment_id, ObjectTrackDynamicProperty(pubsubMsg.e_e_dynamic_status),
                   pubsubMsg.e_Pct_location_uncertainty_x, pubsubMsg.e_Pct_location_uncertainty_y,
                   pubsubMsg.e_Pct_location_uncertainty_yaw, pubsubMsg.e_Cnt_host_lane_frenet_id,
                   (pubsubMsg.a_cartesian_pose[:MAX_CARTESIANPOSE_FIELDS]),
                   pubsubMsg.a_lane_frenet_pose[:MAX_LANEFRENETPOSE_FIELDS],
                   pubsubMsg.a_host_lane_frenet_pose[:MAX_LANEFRENETPOSE_FIELDS])


class ObjectLocalization(PUBSUB_MSG_IMPL):
    e_Cnt_object_id = int
    e_e_object_type = ObjectClassification
    s_bounding_box = BoundingBoxSize
    e_Cnt_obj_hypothesis_count = int
    as_object_hypothesis = List[ObjectHypothesis]

    def __init__(self, e_Cnt_object_id, e_e_object_type, s_bounding_box, e_Cnt_obj_hypothesis_count, as_object_hypothesis):
        # type: (int, ObjectClassification, BoundingBoxSize, int, List[ObjectHypothesis]) -> None
        """
        Actors' localization information
        :param e_Cnt_object_id: Actor's id
        :param e_e_object_type:
        :param s_bounding_box:
        :param e_Cnt_obj_hypothesis_count: Total number of localization - hypotheses for this actor(Only one hypothesis for M0)
        :param as_object_hypothesis: Localization-hypotheses for this actor (Only one hypothesis for M0)
        """
        self.e_Cnt_object_id = e_Cnt_object_id
        self.e_e_object_type = e_e_object_type
        self.s_bounding_box = s_bounding_box
        self.e_Cnt_obj_hypothesis_count = e_Cnt_obj_hypothesis_count
        self.as_object_hypothesis = as_object_hypothesis

    def serialize(self):
        # type: () -> TsSYSObjectLocalization
        pubsub_msg = TsSYSObjectLocalization()

        pubsub_msg.e_Cnt_object_id = self.e_Cnt_object_id
        pubsub_msg.e_e_object_type = self.e_e_object_type.value
        pubsub_msg.s_bounding_box = self.s_bounding_box.serialize()
        pubsub_msg.e_Cnt_obj_hypothesis_count = self.e_Cnt_obj_hypothesis_count
        for i in range(pubsub_msg.e_Cnt_obj_hypothesis_count):
            pubsub_msg.as_object_hypothesis[i] = self.as_object_hypothesis[i].serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSObjectLocalization)->ObjectLocalization

        obj_hypotheses = list()
        for i in range(pubsubMsg.e_Cnt_obj_hypothesis_count):
            obj_hypotheses.append(ObjectHypothesis.deserialize(pubsubMsg.as_object_hypothesis[i]))

        return cls(pubsubMsg.e_Cnt_object_id, ObjectClassification(pubsubMsg.e_e_object_type), BoundingBoxSize.deserialize(pubsubMsg.s_bounding_box),
                   pubsubMsg.e_Cnt_obj_hypothesis_count, obj_hypotheses)


class DataSceneDynamic(PUBSUB_MSG_IMPL):
    e_b_Valid = bool
    s_RecvTimestamp = Timestamp
    s_ComputeTimestamp = Timestamp
    e_Cnt_num_objects = int
    as_object_localization = List[ObjectLocalization]
    s_host_localization = HostLocalization

    def __init__(self, e_b_Valid, s_RecvTimestamp, s_ComputeTimestamp, e_Cnt_num_objects, as_object_localization, s_host_localization):
        # type: (bool, Timestamp, Timestamp, int, List[ObjectLocalization], HostLocalization) -> None
        """

        :param e_b_Valid:
        :param s_RecvTimestamp:
        :param s_ComputeTimestamp:
        :param e_Cnt_num_objects: Total number of actors
        :param as_object_localization:
        :param s_host_localization:
        """
        self.e_b_Valid = e_b_Valid
        self.s_RecvTimestamp = s_RecvTimestamp
        self.s_ComputeTimestamp = s_ComputeTimestamp
        self.e_Cnt_num_objects = e_Cnt_num_objects
        self.as_object_localization = as_object_localization
        self.s_host_localization = s_host_localization

    def serialize(self):
        # type: () -> TsSYSDataSceneDynamic
        pubsub_msg = TsSYSDataSceneDynamic()
        pubsub_msg.e_b_Valid = self.e_b_Valid
        pubsub_msg.s_RecvTimestamp = self.s_RecvTimestamp.serialize()
        pubsub_msg.s_ComputeTimestamp = self.s_ComputeTimestamp.serialize()
        pubsub_msg.e_Cnt_num_objects = self.e_Cnt_num_objects

        for i in range(pubsub_msg.e_Cnt_num_objects):
            pubsub_msg.as_object_localization[i] = self.as_object_localization[i].serialize()

        pubsub_msg.s_host_localization = self.s_host_localization.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSDataSceneDynamic)->DataSceneDynamic

        obj_localizations = list()
        for i in range(pubsubMsg.e_Cnt_num_objects):
            obj_localizations.append(ObjectLocalization.deserialize(pubsubMsg.as_object_localization[i]))

        return cls(pubsubMsg.e_b_Valid, Timestamp.deserialize(pubsubMsg.s_RecvTimestamp), Timestamp.deserialize(pubsubMsg.s_ComputeTimestamp), pubsubMsg.e_Cnt_num_objects,
                   obj_localizations, HostLocalization.deserialize(pubsubMsg.s_host_localization))


class SceneDynamic(PUBSUB_MSG_IMPL):
    """
    PubSub topic=SCENE_DYNAMIC
    Contains localizations of Host and Actors (10Hz)
    """
    s_Header = Header
    s_Data = DataSceneDynamic
    s_MapOrigin = MapOrigin

    def __init__(self, s_Header, s_Data, s_MapOrigin):
        # type: (Header, DataSceneDynamic, MapOrigin) -> None
        self.s_Header = s_Header
        self.s_Data = s_Data
        self.s_MapOrigin = s_MapOrigin

    def serialize(self):
        # type: () -> TsSYSSceneDynamic
        pubsub_msg = TsSYSSceneDynamic()

        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()
        pubsub_msg.s_MapOrigin = self.s_MapOrigin.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSSceneDynamic)->SceneDynamic
        return cls(Header.deserialize(pubsubMsg.s_Header), DataSceneDynamic.deserialize(pubsubMsg.s_Data),
                   MapOrigin.deserialize(pubsubMsg.s_MapOrigin))
