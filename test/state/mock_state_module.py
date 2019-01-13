from logging import Logger

from common_data.interface.py.pubsub import Rte_Types_pubsub_topics as pubsub_topics
from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.state.state import State
from decision_making.src.state.state_module import StateModule


class StateModuleMock(StateModule):
    """
    Send periodic dummy state message
    """
    def __init__(self, pubsub: PubSub, logger: Logger, state: State):
        """

        :param pubsub: communication layer (DDS/LCM/...) instance
        :param logger: logger
        :param state: the state message to publish periodically
        """
        self._state = state
        super().__init__(pubsub=pubsub, logger=logger, scene_dynamic=None)

    def _periodic_action_impl(self):
        self.pubsub.publish(pubsub_topics.STATE_LCM, self._state.serialize())

