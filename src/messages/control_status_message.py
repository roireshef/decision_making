import numpy as np
from interface.Rte_Types.python.sub_structures.TsSYS_ControlStatus import TsSYSControlStatus
from interface.Rte_Types.python.sub_structures.TsSYS_DataControlStatus import TsSYSDataControlStatus
from interface.Rte_Types.python.sub_structures.TsSYS_InterpolatedTrajectory import TsSYSInterpolatedTrajectory

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Header


class InterpolatedTrajectory(PUBSUB_MSG_IMPL):
    a_latPosition = np.ndarray   # Interpolated trajectory - lateral position in m [15]
    a_lonPosition = np.ndarray   # Interpolated trajectory - longitudinal position in m [15]
    a_curvature = np.ndarray     # Interpolated trajectory - curvature in \f$ m^{-1} \f$  [15]
    a_heading = np.ndarray       # Interpolated trajectory - heading in rad [15]
    a_vx = np.ndarray            # Interpolated trajectory - Vx in \f$ \frac{m}{s} \f$ [15]
    a_ax = np.ndarray            # Interpolated trajectory - ax \f$ \frac{m}{s^2} \f$ [15]

    def __init__(self, a_latPosition: np.ndarray, a_lonPosition: np.ndarray, a_curvature: np.ndarray,
                 a_heading: np.ndarray, a_vx: np.ndarray, a_ax: np.ndarray):
        self.a_latPosition = a_latPosition
        self.a_lonPosition = a_lonPosition
        self.a_curvature = a_curvature
        self.a_heading = a_heading
        self.a_vx = a_vx
        self.a_ax = a_ax

    def serialize(self) -> TsSYSInterpolatedTrajectory:
        pubsub_msg = TsSYSInterpolatedTrajectory()

        pubsub_msg.a_latPosition = self.a_latPosition
        pubsub_msg.a_lonPosition = self.a_lonPosition
        pubsub_msg.a_curvature = self.a_curvature
        pubsub_msg.a_heading = self.a_heading
        pubsub_msg.a_vx = self.a_vx
        pubsub_msg.a_ax = self.a_ax

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSInterpolatedTrajectory):
        a_latPosition = pubsubMsg.a_latPosition
        a_lonPosition = pubsubMsg.a_lonPosition
        a_curvature = pubsubMsg.a_curvature
        a_heading = pubsubMsg.a_heading
        a_vx = pubsubMsg.a_vx
        a_ax = pubsubMsg.a_ax

        return cls(a_latPosition, a_lonPosition, a_curvature, a_heading, a_vx, a_ax)

class DataControlStatus(PUBSUB_MSG_IMPL):
    e_b_IsAlive = bool              # True if control is executing
    e_b_TTCEnabled = bool           # True if control is engaged
    e_b_HandsOffStrWh = bool        # True if hands are not on the steering wheel
    e_DisableReason = int           # bit encoded disable reason
    e_SteeringCmd = float           # Desired steering wheel angle in deg
    e_AxlTorqueCmd = float          # Desired axle torque in \f$ Nm \f$ *
    e_BrakeAccelCmd = float         # Desired brake acceleration in \frac{m}{s^2}
    e_BrakeType = int               # Brake type
    e_LatDev = float                # Lateral deviation in vehicle frame in m*
    e_LonDev = float                # Longitudinal deviation in vehicle frame in m
    e_VxDev = float                 # Vx deviation in vehicle frame in \f$ \frac{m}{s} \f$
    a_commit_id = np.ndarray        # Current commit id for when the model was built. Currently len 41, only 8 characters (short SHA)
    e_DrvAstdGoSt = int             # Driver assisted go state
    s_InterpTraj = InterpolatedTrajectory     # Interpolated trajectory used as MPC input
    e_vyEst = float                 # Vy Est \f$ \frac{m}{s} \f$
    e_RoadBankAccel = float         # Estimated acceleration due to road bank \f$ \frac{m}{s^2} \f$
    e_RoadInclinationAccel = float  # Estimated acceleration due to road inclination (a.k.a grade) in \f$ \frac{m}{s^2} \f$
    e_CCSwitchStatus = int          # CC Switch status


    def __init__(self, e_b_IsAlive: bool, e_b_TTCEnabled: bool, e_b_HandsOffStrWh: bool, e_DisableReason: int,
                 e_SteeringCmd: float, e_AxlTorqueCmd: float, e_BrakeAccelCmd: float, e_BrakeType: int,
                 e_LatDev: float, e_LonDev: float, e_VxDev: float, a_commit_id: np.ndarray,
                 e_DrvAstdGoSt: int, s_InterpTraj: InterpolatedTrajectory, e_vyEst: float,
                 e_RoadBankAccel: float, e_RoadInclinationAccel: float, e_CCSwitchStatus: int):
        """
        Takeover Flag
        :param e_b_is_takeover_needed: true = takeover needed,
                                       false = takeover not needed
        """
        self.e_b_IsAlive = e_b_IsAlive
        self.e_b_TTCEnabled = e_b_TTCEnabled
        self.e_b_HandsOffStrWh = e_b_HandsOffStrWh
        self.e_DisableReason = e_DisableReason
        self.e_SteeringCmd = e_SteeringCmd
        self.e_AxlTorqueCmd = e_AxlTorqueCmd
        self.e_BrakeAccelCmd = e_BrakeAccelCmd
        self.e_BrakeType = e_BrakeType
        self.e_LatDev = e_LatDev
        self.e_LonDev = e_LonDev
        self.e_VxDev = e_VxDev
        self.a_commit_id = a_commit_id
        self.e_DrvAstdGoSt = e_DrvAstdGoSt
        self.s_InterpTraj = s_InterpTraj
        self.e_vyEst = e_vyEst
        self.e_RoadBankAccel = e_RoadBankAccel
        self.e_RoadInclinationAccel = e_RoadInclinationAccel
        self.e_CCSwitchStatus = e_CCSwitchStatus

    def serialize(self) -> TsSYSDataControlStatus:
        pubsub_msg = TsSYSDataControlStatus()

        pubsub_msg.e_b_IsAlive = self.e_b_IsAlive
        pubsub_msg.e_b_TTCEnabled = self.e_b_TTCEnabled
        pubsub_msg.e_b_HandsOffStrWh = self.e_b_HandsOffStrWh
        pubsub_msg.e_DisableReason = self.e_DisableReason
        pubsub_msg.e_SteeringCmd = self.e_SteeringCmd
        pubsub_msg.e_AxlTorqueCmd = self.e_AxlTorqueCmd
        pubsub_msg.e_BrakeAccelCmd = self.e_BrakeAccelCmd
        pubsub_msg.e_BrakeType = self.e_BrakeType
        pubsub_msg.e_LatDev = self.e_LatDev
        pubsub_msg.e_LonDev = self.e_LonDev
        pubsub_msg.e_VxDev = self.e_VxDev
        pubsub_msg.a_commit_id = self.a_commit_id
        pubsub_msg.e_DrvAstdGoSt = self.e_DrvAstdGoSt
        pubsub_msg.s_InterpTraj = self.s_InterpTraj.serialize()
        pubsub_msg.e_vyEst = self.e_vyEst
        pubsub_msg.e_RoadBankAccel = self.e_RoadBankAccel
        pubsub_msg.e_RoadInclinationAccel = self.e_RoadInclinationAccel
        pubsub_msg.e_CCSwitchStatus = self.e_CCSwitchStatus

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataControlStatus):
        a_commit_id = pubsubMsg.a_commit_id
        s_InterpTraj = InterpolatedTrajectory.deserialize(pubsubMsg.s_InterpTraj)

        return cls(pubsubMsg.e_b_IsAlive, pubsubMsg.e_b_TTCEnabled, pubsubMsg.e_b_HandsOffStrWh,
                   pubsubMsg.e_DisableReason, pubsubMsg.e_SteeringCmd, pubsubMsg.e_AxlTorqueCmd,
                   pubsubMsg.e_BrakeAccelCmd, pubsubMsg.e_BrakeType, pubsubMsg.e_LatDev, pubsubMsg.e_LonDev,
                   pubsubMsg.e_VxDev, a_commit_id, pubsubMsg.e_DrvAstdGoSt, s_InterpTraj, pubsubMsg.e_vyEst,
                   pubsubMsg.e_RoadBankAccel, pubsubMsg.e_RoadInclinationAccel, pubsubMsg.e_CCSwitchStatus)


class ControlStatus(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_Data = DataControlStatus

    def __init__(self, s_Header: Header, s_Data: DataControlStatus):
        """
        Class that represents the CONTROL STATUS topic
        :param s_Header: General Information
        :param s_Data: Message Data
        """
        self.s_Header = s_Header
        self.s_Data = s_Data

    def serialize(self) -> TsSYSControlStatus:
        pubsub_msg = TsSYSControlStatus()

        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSControlStatus):
        return cls(Header.deserialize(pubsubMsg.s_Header),
                   DataControlStatus.deserialize(pubsubMsg.s_Data))
