from logging import Logger

from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.navigation.navigation_facade import NavigationFacade


class NavigationFacadeMock(NavigationFacade):
    """
    Sends a periodic dummy navigation message
    """
    def __init__(self, pubsub: PubSub, logger: Logger, navigation_plan_msg: NavigationPlanMsg):
        """
        :param pubsub: communication layer (DDS/LCM/...) instance
        :param logger: logger
        :param navigation_plan_msg: the navigation plan message to publish periodically
        """
        self._navigation_plan_msg = navigation_plan_msg
        super().__init__(pubsub, logger, None)

    def _periodic_action_impl(self):
        """
        Publishes the received messages initialized in init
        :return: void
        """
        self._publish_navigation_plan(self._navigation_plan_msg)

