from common_data.src.communication.pubsub.pubsub import PubSub
from common_data.interface.py.pubsub import Rte_Types_pubsub_topics as pubsub_topics
from decision_making.src.state.state import State
from decision_making.src.state.state_module import StateModule
from logging import Logger


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
        super().__init__(pubsub=pubsub, logger=logger, dynamic_objects=None, ego_state=None, occupancy_state=None)

    def _periodic_action_impl(self):
        self.pubsub.publish(pubsub_topics.STATE_LCM, self._state.serialize())

