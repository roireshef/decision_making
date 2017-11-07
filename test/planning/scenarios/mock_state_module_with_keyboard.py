from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import *
from decision_making.src.state.state import *
from decision_making.src.state.state_module import StateModule
from mapping.src.service.map_service import MapService
from pynput.keyboard import Key, Listener

class StateModuleMockWithKeyboard(StateModule):
    """
    Send periodic dummy state message
    Listen to keyboard to switch between the dummy messages in the input list
    """
    def __init__(self, dds: DdsPubSub, logger: Logger, state_list: List[State]):
        """

        :param dds: communication layer (DDS) instance
        :param logger: logger
        :param state: the state message to publish periodically
        """
        self._state_list=state_list
        self._state_idx=0
        self._state = state_list[self._state_idx]
        self._state_list_len = len(self._state_list)
        MapService.initialize()
        map_api = MapService.get_instance()
        super().__init__(dds, logger, map_api, None, None, None)
        self.listener=Listener(on_press=self.on_press)
        self.listener.start()

    def _periodic_action_impl(self):
        self.dds.publish(STATE_PUBLISH_TOPIC, self._state.serialize())

    def on_press(self,key):
        if key==Key.left:
            self._state_idx -= 1
        elif key==Key.right:
            self._state_idx += 1

        if self._state_idx >= self._state_list_len:
            self._state_idx = 0
        elif self._state_idx < 0:
            self._state_idx =self._state_list_len - 1

        self._state = self._state_list[self._state_idx]

    def __del__(self):
        self.listener.stop()