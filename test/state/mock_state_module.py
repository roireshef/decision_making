from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import *
from decision_making.src.state.state import *
from decision_making.src.state.state_module import StateModule


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
        super().__init__(dds, logger, None, None, None, None)

    def _periodic_action_impl(self):
        self.__publish_state()

    # TODO: protected instead of private
    def __publish_state(self):

        self.dds.publish(STATE_PUBLISH_TOPIC, self._state.serialize())
