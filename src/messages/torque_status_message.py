
from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Header, Timestamp
from interface.Rte_Types.python.enums import TeSYS_TorqueRequestStatus, TeSYS_BrakeStatus, TeSYS_SteeringStatus, \
    TeSYS_ACCATCSLmtgStatType
from interface.Rte_Types.python.sub_structures.TsSYS_ActuatorStatus import TsSYSActuatorStatus
from interface.Rte_Types.python.sub_structures.TsSYS_DataActuatorStatus import TsSYSDataActuatorStatus


class DataActuatorStatus(PUBSUB_MSG_IMPL):
    e_accel_pedal_pos = float
    e_brake_pedal_pos = float
    e_steering_wheel_angle = float
    e_actual_axle_torque = int
    e_brake_status = TeSYS_BrakeStatus
    e_torque_req_status = TeSYS_TorqueRequestStatus
    e_steering_status = TeSYS_SteeringStatus
    e_min_trq_avail = float
    e_max_trq_avail = float
    e_ACCATCSLmtgStat = TeSYS_ACCATCSLmtgStatType

    def __init__(self, e_accel_pedal_pos: float, e_brake_pedal_pos: float, e_steering_wheel_angle: float,
                 e_actual_axle_torque: int, e_brake_status: TeSYS_BrakeStatus,
                 e_torque_req_status: TeSYS_TorqueRequestStatus, e_steering_status: TeSYS_SteeringStatus,
                 e_min_trq_avail: float, e_max_trq_avail: float, e_ACCATCSLmtgStat: TeSYS_ACCATCSLmtgStatType):
        """
        Initialize message containing actuator status
        """
        self.e_accel_pedal_pos = e_accel_pedal_pos
        self.e_brake_pedal_pos = e_brake_pedal_pos
        self.e_steering_wheel_angle = e_steering_wheel_angle
        self.e_actual_axle_torque = e_actual_axle_torque
        self.e_brake_status = e_brake_status
        self.e_torque_req_status = e_torque_req_status
        self.e_steering_status = e_steering_status
        self.e_min_trq_avail = e_min_trq_avail
        self.e_max_trq_avail = e_max_trq_avail
        self.e_ACCATCSLmtgStat = e_ACCATCSLmtgStat

    def serialize(self) -> TsSYSDataActuatorStatus:
        pubsub_msg = TsSYSDataActuatorStatus()
        pubsub_msg.e_accel_pedal_pos = self.e_accel_pedal_pos
        pubsub_msg.e_brake_pedal_pos = self.e_brake_pedal_pos
        pubsub_msg.e_steering_wheel_angle = self.e_steering_wheel_angle
        pubsub_msg.e_actual_axle_torque = self.e_actual_axle_torque
        pubsub_msg.e_brake_status = self.e_brake_status
        pubsub_msg.e_torque_req_status = self.e_torque_req_status
        pubsub_msg.e_steering_status = self.e_steering_status
        pubsub_msg.e_min_trq_avail = self.e_min_trq_avail
        pubsub_msg.e_max_trq_avail = self.e_max_trq_avail
        pubsub_msg.e_ACCATCSLmtgStat = self.e_ACCATCSLmtgStat
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataActuatorStatus):
        return cls(pubsubMsg.e_accel_pedal_pos, pubsubMsg.e_brake_pedal_pos, pubsubMsg.e_steering_wheel_angle,
                   pubsubMsg.e_actual_axle_torque, pubsubMsg.e_brake_status,  pubsubMsg.e_torque_req_status,
                   pubsubMsg.e_steering_status, pubsubMsg.e_min_trq_avail, pubsubMsg.e_max_trq_avail,
                   pubsubMsg.e_ACCATCSLmtgStat)


class ActuatorStatus(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_Data = DataActuatorStatus

    def __init__(self, s_Header: Header, s_Data: DataActuatorStatus):
        """
        Class that represents the UC_SYSTEM_ACTUATOR_STATUS topic
        :param s_Header: General Information
        :param s_Data: Message Data
        """
        self.s_Header = s_Header
        self.s_Data = s_Data

    def serialize(self) -> TsSYSActuatorStatus:
        pubsub_msg = TsSYSActuatorStatus()
        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSActuatorStatus):
        return cls(Header.deserialize(pubsubMsg.s_Header), DataActuatorStatus.deserialize(pubsubMsg.s_Data))
