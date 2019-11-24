from typing import List, Dict

from decision_making.src.messages.scene_static_enums import TrafficSignalState
from interface.Rte_Types.python.sub_structures.TsSYS_DynamicTrafficControlDeviceStatus import TsSYSDynamicTrafficControlDeviceStatus
from interface.Rte_Types.python.sub_structures.TsSYS_DataSceneTrafficControlDevices import TsSYSDataSceneTrafficControlDevices
from interface.Rte_Types.python.sub_structures.TsSYS_SceneTrafficControlDevices import TsSYSSceneTrafficControlDevices
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Timestamp, Header


class DynamicTrafficControlDeviceStatus(PUBSUB_MSG_IMPL):
    e_i_dynamic_traffic_control_device_id = int
    a_e_status = List[TrafficSignalState]
    a_Pct_status_confidence = List[float]

    def __init__(self, e_i_dynamic_traffic_control_device_id: int, a_e_status: List[TrafficSignalState],
                 a_Pct_status_confidence: List[float]):
        """
        Distribution over a Dynamic traffic-flow-control device's possible statuses, eg. red(20%)-yellow(70%)-green(10%)
        :param e_i_dynamic_traffic_control_device_id: ID of traffic control device
        :param a_e_status: status of dynamic TCD
        :param a_Pct_status_confidence: confidence distribution of status
        """
        self.e_i_dynamic_traffic_control_device_id = e_i_dynamic_traffic_control_device_id
        self.a_e_status = a_e_status
        self.a_Pct_status_confidence = a_Pct_status_confidence

    def serialize(self) -> TsSYSDynamicTrafficControlDeviceStatus:
        pubsub_msg = TsSYSDynamicTrafficControlDeviceStatus()

        pubsub_msg.e_i_dynamic_traffic_control_device_id = self.e_i_dynamic_traffic_control_device_id
        pubsub_msg.e_Cnt_status_count = len(self.a_e_status)
        for i in range(pubsub_msg.e_Cnt_status_count):
            pubsub_msg.a_e_status[i] = self.a_e_status[i].value
        for i in range(pubsub_msg.e_Cnt_status_count):
            pubsub_msg.a_Pct_status_confidence[i] = self.a_Pct_status_confidence[i]

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSDynamicTrafficControlDeviceStatus)->DynamicTrafficControlDeviceStatus

        a_e_status = list()
        for i in range(pubsubMsg.e_Cnt_status_count):
            a_e_status.append(TrafficSignalState(pubsubMsg.a_e_status[i]))  # convert uint8 to TrafficSignalState

        a_Pct_status_confidence = list()
        for i in range(pubsubMsg.e_Cnt_status_count):
            a_e_status.append(pubsubMsg.a_Pct_status_confidence[i])

        return cls(pubsubMsg.e_i_dynamic_traffic_control_device_id, a_e_status, a_Pct_status_confidence)


class DataSceneTrafficControlDevices(PUBSUB_MSG_IMPL):
    as_dynamic_traffic_control_device_status = Dict[int, DynamicTrafficControlDeviceStatus]
    s_RecvTimestamp = Timestamp
    s_ComputeTimestamp = Timestamp
    # TODO should that have a TCD id

    def __init__(self, s_RecvTimestamp: Timestamp, s_ComputeTimestamp: Timestamp, 
                 as_dynamic_traffic_control_device_status: Dict[int, DynamicTrafficControlDeviceStatus]):
        """

        :param as_dynamic_traffic_control_device_status: The status distribution for each dynamic TCD in the scene
        :param: s_RecvTimestamp
        :param: s_ComputeTimestamp
        """
        self.s_RecvTimestamp = s_RecvTimestamp
        self.s_ComputeTimestamp = s_ComputeTimestamp
        
        self.as_dynamic_traffic_control_device_status = as_dynamic_traffic_control_device_status

    def serialize(self) -> TsSYSDataSceneTrafficControlDevices:
        pubsub_msg = TsSYSDataSceneTrafficControlDevices()
        
        pubsub_msg.s_RecvTimestamp = Timestamp.serialize(self.s_RecvTimestamp)
        pubsub_msg.s_ComputeTimestamp = Timestamp.serialize(self.s_ComputeTimestamp)

        pubsub_msg.e_Cnt_dynamic_traffic_control_device_status_count = len(self.as_dynamic_traffic_control_device_status)
        for i, tcd_status in enumerate(self.as_dynamic_traffic_control_device_status.values()):
            pubsub_msg.as_dynamic_traffic_control_device_status[i] = \
                DynamicTrafficControlDeviceStatus.serialize(tcd_status)

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataSceneTrafficControlDevices):
        # returns DataSceneTrafficControlDevices
        s_RecvTimestamp = Timestamp.deserialize(pubsubMsg.s_RecvTimestamp)
        s_ComputeTimestamp = Timestamp.deserialize(pubsubMsg.s_ComputeTimestamp)
        as_dynamic_traffic_control_device_status = dict()
        for i in range(pubsubMsg.e_Cnt_dynamic_traffic_control_device_status_count):
            tcd_status = DynamicTrafficControlDeviceStatus.deserialize(pubsubMsg.as_dynamic_traffic_control_device_status[i])
            as_dynamic_traffic_control_device_status[tcd_status.e_i_dynamic_traffic_control_device_id] = tcd_status

        return cls(s_RecvTimestamp, s_ComputeTimestamp, as_dynamic_traffic_control_device_status)


class SceneTrafficControlDevices(PUBSUB_MSG_IMPL):
    """
    PubSub topic=  SCENE_TRAFFIC_CONTROL_DEVICES
    Contains information of Traffic Control Devices controlling Traffic Control Bars ( ??? Hz)
    """
    s_Header = Header
    s_Data = DataSceneTrafficControlDevices

    def __init__(self, s_Header: Header, s_Data: DataSceneTrafficControlDevices):
        self.s_Header = s_Header
        self.s_Data = s_Data

    def serialize(self) -> TsSYSSceneTrafficControlDevices:
        pubsub_msg = TsSYSSceneTrafficControlDevices()

        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSSceneTrafficControlDevices):
        # returns SceneTrafficControlDevices
        return cls(Header.deserialize(pubsubMsg.s_Header), DataSceneTrafficControlDevices.deserialize(pubsubMsg.s_Data))
