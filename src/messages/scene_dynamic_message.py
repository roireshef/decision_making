from enum import Enum
from typing import List

from common_data.interface.py.idl_generated_files.Rte_Types.TsSYS_SceneDynamic import TsSYSSceneDynamic
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_BoundingBoxSize import \
    TsSYSBoundingBoxSize
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_CartesianPose import TsSYSCartesianPose
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_DataSceneDynamic import \
    TsSYSDataSceneDynamic
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_DataSceneHost import TsSYSDataSceneHost
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_Header import TsSYSHeader
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_HostLocalization import \
    TsSYSHostLocalization
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_LaneFrenetPose import \
    TsSYSLaneFrenetPose
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_MapOrigin import TsSYSMapOrigin
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_ObjectHypothesis import \
    TsSYSObjectHypothesis
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_ObjectLocalization import \
    TsSYSObjectLocalization
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_Timestamp import TsSYSTimestamp
from decision_making.src.global_constants import PUBSUB_MSG_IMPL


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


class CartesianPose(PUBSUB_MSG_IMPL):
    e_l_EastX = float
    e_l_NorthY = float
    e_phi_heading = float
    e_v_velocity = float
    e_a_acceleration = float
    e_il_curvature = float

    def __init__(self, e_l_EastX, e_l_NorthY, e_phi_heading, e_v_velocity, e_a_acceleration, e_il_curvature):
        # type: (float, float, float, float, float, float)->None
        """
        Pose information in Cartesian (ENU) frame
        :param e_l_EastX: East-X in the ENU Frame. Eastwards is positive.
        :param e_l_NorthY: North-Y in the ENU Frame. Northwards is positive.
        :param e_phi_heading: ISO-8855 heading (a.k.a yaw)
        The yaw angle is between the forward direction of the vehicle,projected in to the horizontal plane and the East.
        The range of the yaw angle is from -180 to +180 degrees, where positive rotation is from the East and
        counter-clockwise when looking from above, about the up direction axis of the vehicle.
        :param e_v_velocity: Ego. along the longitudinal-axis. Positive is forward.
        :param e_a_acceleration: Ego. along the longitudinal-axis. Positive is forward.
        :param e_il_curvature: Ego. Positive means vehicle is turning left
        """
        self.e_l_EastX = e_l_EastX
        self.e_l_NorthY = e_l_NorthY
        self.e_phi_heading = e_phi_heading
        self.e_v_velocity = e_v_velocity
        self.e_a_acceleration = e_a_acceleration
        self.e_il_curvature = e_il_curvature

    def serialize(self):
        # type: () -> TsSYSCartesianPose
        pubsub_msg = TsSYSCartesianPose()

        pubsub_msg.e_l_EastX = self.e_l_EastX
        pubsub_msg.e_l_NorthY = self.e_l_NorthY
        pubsub_msg.e_phi_heading = self.e_phi_heading
        pubsub_msg.e_v_velocity = self.e_v_velocity
        pubsub_msg.e_a_acceleration = self.e_a_acceleration
        pubsub_msg.e_il_curvature = self.e_il_curvature

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSCartesianPose)->CartesianPose
        return cls(pubsubMsg.e_l_EastX, pubsubMsg.e_l_NorthY, pubsubMsg.e_phi_heading,
                   pubsubMsg.e_v_velocity, pubsubMsg.e_a_acceleration, pubsubMsg.e_il_curvature)


class LaneFrenetPose(PUBSUB_MSG_IMPL):
    e_l_EastX = float
    e_l_NorthY = float
    e_phi_heading = float
    e_v_velocity = float
    e_a_acceleration = float
    e_il_curvature = float

    def __init__(self, e_l_s, e_v_s_dot, e_a_s_dotdot, e_l_d, e_v_d_dot, e_a_d_dotdot):
        # type: (float, float, float, float, float, float)->None
        """
        Pose information in Frenet-Serret frame.
        :param e_l_s: Pose information in Frenet-Serret frame
        s, station (a.k.a. arc-length, a.k.a. progress) along the curve of the Frenet-Serret frame.
        Forward (downstream) of the curve is positive.
        The curve's origin is located at the center and start of the lane-segment.
        The curve runs downstream along the center of the lane-segment.
        Each lane-segment has its own Frenet-Serret frame.
        :param e_v_s_dot: s-velocity
        :param e_a_s_dotdot: s-acceleration
        :param e_l_d: d, displacement, a.k.a. offset on the normal to the curve of the Frenet-Serret Frame.
        Leftside of the curve is positive. The curve's origin is located at the center and start of the lane-segment.
        The curve runs downstream along the center of the lane-segment.Each lane-segment has its own Frenet-Serret frame.
        :param e_v_d_dot: d-velocity
        :param e_a_d_dotdot: d-acceleration
        """
        self.e_l_s = e_l_s
        self.e_v_s_dot = e_v_s_dot
        self.e_a_s_dotdot = e_a_s_dotdot
        self.e_l_d = e_l_d
        self.e_v_d_dot = e_v_d_dot
        self.e_a_d_dotdot = e_a_d_dotdot

    def serialize(self):
        # type: () -> TsSYSLaneFrenetPose
        pubsub_msg = TsSYSLaneFrenetPose()

        pubsub_msg.e_l_s = self.e_l_s
        pubsub_msg.e_v_s_dot = self.e_v_s_dot
        pubsub_msg.e_a_s_dotdot = self.e_a_s_dotdot
        pubsub_msg.e_l_d = self.e_l_d
        pubsub_msg.e_v_d_dot = self.e_v_d_dot
        pubsub_msg.e_a_d_dotdot = self.e_a_d_dotdot

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSLaneFrenetPose)->LaneFrenetPose
        return cls(pubsubMsg.e_l_s, pubsubMsg.e_v_s_dot, pubsubMsg.e_a_s_dotdot,
                   pubsubMsg.e_l_d, pubsubMsg.e_v_d_dot, pubsubMsg.e_a_d_dotdot)


class HostLocalization(PUBSUB_MSG_IMPL):
    e_Cnt_road_segment_id = int
    e_Cnt_lane_segment_id = int
    s_cartesian_pose = CartesianPose
    s_lane_frenet_pose = LaneFrenetPose

    def __init__(self, e_Cnt_road_segment_id, e_Cnt_lane_segment_id, s_cartesian_pose, s_lane_frenet_pose):
        # type: (int, int, CartesianPose, LaneFrenetPose)->None
        """
        Host-localization information
        :param e_Cnt_road_segment_id: The ID of the road-segment that the host is in
        :param e_Cnt_lane_segment_id: The ID of the lane-segment that the host is in
        :param s_cartesian_pose: The host's pose, expressed in the Map (ENU) frame
        :param s_lane_frenet_pose: The host's pose, expressed in the the Frenet-Serret frame of the host's lane-segment
        """
        self.e_Cnt_road_segment_id = e_Cnt_road_segment_id
        self.e_Cnt_lane_segment_id = e_Cnt_lane_segment_id
        self.s_cartesian_pose = s_cartesian_pose
        self.s_lane_frenet_pose = s_lane_frenet_pose

    def serialize(self):
        # type: () -> TsSYSHostLocalization
        pubsub_msg = TsSYSHostLocalization()

        pubsub_msg.e_Cnt_road_segment_id = self.e_Cnt_road_segment_id
        pubsub_msg.e_Cnt_lane_segment_id = self.e_Cnt_lane_segment_id
        pubsub_msg.s_cartesian_pose = self.s_cartesian_pose.serialize()
        pubsub_msg.s_lane_frenet_pose = self.s_lane_frenet_pose.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSHostLocalization)->HostLocalization
        return cls(pubsubMsg.e_Cnt_road_segment_id, pubsubMsg.e_Cnt_lane_segment_id,
                   CartesianPose.deserialize(pubsubMsg.s_cartesian_pose),
                   LaneFrenetPose.deserialize(pubsubMsg.s_lane_frenet_pose))


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
    s_cartesian_pose = CartesianPose
    s_lane_frenet_pose = LaneFrenetPose
    s_host_lane_frenet_pose = LaneFrenetPose

    def __init__(self, e_r_probability, e_Cnt_lane_segment_id, e_e_dynamic_status, e_Pct_location_uncertainty_x,
                 e_Pct_location_uncertainty_y, e_Pct_location_uncertainty_yaw, e_Cnt_host_lane_frenet_id,
                 s_cartesian_pose, s_lane_frenet_pose, s_host_lane_frenet_pose):
        # type: (float, int, ObjectTrackDynamicProperty, float, float, float, int, CartesianPose, LaneFrenetPose, LaneFrenetPose) -> None
        """
        Actors-hypotheses information
        :param e_r_probability: Probability of this hypothesis (not relevant for M0)
        :param e_Cnt_lane_segment_id: The lane-segment ID that this actor-hypothesis is in
        :param e_e_dynamic_status:
        :param e_Pct_location_uncertainty_x: Not relevant for M0
        :param e_Pct_location_uncertainty_y: Not relevant for M0
        :param e_Pct_location_uncertainty_yaw: Not relevant for M0
        :param e_Cnt_host_lane_frenet_id: The ID of the lane-segment that the host is in
        :param s_cartesian_pose: The pose of this actor-hypothesis, expressed in the Map (ENU) frame
        :param s_lane_frenet_pose: The pose of this actor-hypothesis, expressed in the Frenet-Serret frame of its own lane-segment
        :param s_host_lane_frenet_pose: The pose of this actor-hypothesis, expressed in the Frenet-Serret frame of the host's lane-segment
        """
        self.e_r_probability = e_r_probability
        self.e_Cnt_lane_segment_id = e_Cnt_lane_segment_id
        self.e_e_dynamic_status = e_e_dynamic_status
        self.e_Pct_location_uncertainty_x = e_Pct_location_uncertainty_x
        self.e_Pct_location_uncertainty_y = e_Pct_location_uncertainty_y
        self.e_Pct_location_uncertainty_yaw = e_Pct_location_uncertainty_yaw
        self.e_Cnt_host_lane_frenet_id = e_Cnt_host_lane_frenet_id
        self.s_cartesian_pose = s_cartesian_pose
        self.s_lane_frenet_pose = s_lane_frenet_pose
        self.s_host_lane_frenet_pose = s_host_lane_frenet_pose

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
        pubsub_msg.s_cartesian_pose = self.s_cartesian_pose.serialize()
        pubsub_msg.s_lane_frenet_pose = self.s_lane_frenet_pose.serialize()
        pubsub_msg.s_host_lane_frenet_pose = self.s_host_lane_frenet_pose.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSObjectHypothesis)->ObjectHypothesis
        return cls(pubsubMsg.e_r_probability, pubsubMsg.e_Cnt_lane_segment_id, ObjectTrackDynamicProperty(pubsubMsg.e_e_dynamic_status),
                   pubsubMsg.e_Pct_location_uncertainty_x, pubsubMsg.e_Pct_location_uncertainty_y,
                   pubsubMsg.e_Pct_location_uncertainty_yaw, pubsubMsg.e_Cnt_host_lane_frenet_id,
                   CartesianPose.deserialize(pubsubMsg.s_cartesian_pose),
                   LaneFrenetPose.deserialize(pubsubMsg.s_lane_frenet_pose),
                   LaneFrenetPose.deserialize(pubsubMsg.s_host_lane_frenet_pose))


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
    s_ComputeTimestamp = Timestamp
    e_Cnt_num_objects = int
    as_object_localization = List[ObjectLocalization]
    s_host_localization = HostLocalization

    def __init__(self, e_b_Valid, s_ComputeTimestamp, e_Cnt_num_objects, as_object_localization, s_host_localization):
        # type: (bool, Timestamp, int, List[ObjectLocalization], HostLocalization) -> None
        """

        :param e_b_Valid:
        :param s_ComputeTimestamp:
        :param e_Cnt_num_objects: Total number of actors
        :param as_object_localization:
        :param s_host_localization:
        """
        self.e_b_Valid = e_b_Valid
        self.s_ComputeTimestamp = s_ComputeTimestamp
        self.e_Cnt_num_objects = e_Cnt_num_objects
        self.as_object_localization = as_object_localization
        self.s_host_localization = s_host_localization

    def serialize(self):
        # type: () -> TsSYSDataSceneDynamic
        pubsub_msg = TsSYSDataSceneDynamic()
        pubsub_msg.e_b_Valid = self.e_b_Valid
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

        return cls(pubsubMsg.e_b_Valid, Timestamp.deserialize(pubsubMsg.s_ComputeTimestamp), pubsubMsg.e_Cnt_num_objects,
                   obj_localizations, HostLocalization.deserialize(pubsubMsg.s_host_localization))


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
