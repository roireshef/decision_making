from typing import Dict

from decision_making.src.exceptions import SceneModelIsEmpty
from decision_making.src.messages.scene_tcd_message import DynamicTrafficControlDeviceStatus


class SceneTrafficControlDevicesStatusModel:
    """
    Data layer. Holds the data from SceneTrafficControlDevicesStatus.
     A <<Singleton>>
    """
    __instance = None
    traffic_control_devices_status: Dict[int, DynamicTrafficControlDeviceStatus]

    def __init__(self) -> None:
        self._traffic_control_devices_status = None

    @classmethod
    def get_instance(cls) -> None:
        """
        :return: The instance of SceneTrafficControlDevicesStatusModel
        """
        if cls.__instance is None:
            cls.__instance = SceneTrafficControlDevicesStatusModel()
        return cls.__instance

    def set_traffic_control_devices_status(self, traffic_control_devices_status: Dict[int, DynamicTrafficControlDeviceStatus]) -> None:
        """
        Add a SceneTrafficControlDevicesStatus to the model. Currently this assumes there is only
        a single message
        :param traffic_control_devices_status:  The SceneTrafficControlDevicesStatus
        :return:
        """
        self._traffic_control_devices_status = traffic_control_devices_status

    def get_traffic_control_devices_status(self) -> Dict[int, DynamicTrafficControlDeviceStatus]:
        """
        Gets the last message in list
        :return:  The SceneTrafficControlDevicesStatus
        """
        if self._traffic_control_devices_status is None:
            raise SceneModelIsEmpty('Scene traffic control devices status model is empty')
        return self._traffic_control_devices_status


