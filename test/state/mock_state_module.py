from common_data.src.communication.pubsub.pubsub import PubSub
from common_data.lcm.config import pubsub_topics
from decision_making.src.global_constants import *
from decision_making.src.state.state import *
from decision_making.src.state.state_module import StateModule
from mapping.src.service.map_service import MapService
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
        super().__init__(pubsub, logger, None, None, None)

    def _periodic_action_impl(self):
        self.pubsub.publish(pubsub_topics.STATE_TOPIC, self._state.to_lcm())

