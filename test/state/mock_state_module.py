from logging import Logger

from decision_making.src.infra.pubsub import PubSub
from common_data.interface.Rte_Types.python import Rte_Types_pubsub as pubsub_topics
from decision_making.src.state.state import State
from decision_making.src.state.state_module import StateModule


class StateModuleMock(StateModule):
    """
    Send periodic dummy state message
    """
    def __init__(self, pubsub: PubSub, logger: Logger, state: State):
        """

        :param logger: logger
        :param state: the state message to publish periodically
        """
        self._state = state
        super().__init__(pubsub=pubsub, logger=logger, scene_dynamic=None)

    def _periodic_action_impl(self):
        self.pubsub.publish(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_STATE"], self._state.serialize())

