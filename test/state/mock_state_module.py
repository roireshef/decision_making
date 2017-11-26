from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import *
from decision_making.src.state.state import *
from decision_making.src.state.state_module import StateModule
from mapping.src.service.map_service import MapService
from logging import Logger


class StateModuleMock(StateModule):
    """
    Send periodic dummy state message
    """
    def __init__(self, dds: DdsPubSub, logger: Logger, state: State):
        """

        :param dds: communication layer (DDS) instance
        :param logger: logger
        :param state: the state message to publish periodically
        """
        self._state = state
        MapService.initialize()
        map_api = MapService.get_instance()
        super().__init__(dds, logger, None, None, None)

    def _periodic_action_impl(self):
        self.dds.publish(STATE_PUBLISH_TOPIC, self._state.serialize())
