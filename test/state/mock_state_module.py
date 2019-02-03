from logging import Logger

from decision_making.src.infra.pubsub import PubSub
from Rte_Types.python.uc_system import uc_system_state_lcm
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
        self.pubsub.publish(uc_system_state_lcm, self._state.serialize())

