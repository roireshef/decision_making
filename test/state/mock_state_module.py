from logging import Logger

from common_data.interface.Rte_Types.python import Rte_Types_pubsub as pubsub_topics
from decision_making.src.state.state import State
from decision_making.src.state.state_module import StateModule


class StateModuleMock(StateModule):
    """
    Send periodic dummy state message
    """
    def __init__(self, logger: Logger, state: State):
        """

        :param logger: logger
        :param state: the state message to publish periodically
        """
        self._state = state
        super().__init__(logger=logger, scene_dynamic=None)

    def _periodic_action_impl(self):
        pubsub_topics.UC_SYSTEM_STATE_LCM.send(self._state.serialize())

