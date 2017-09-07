from logging import Logger

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.navigation.navigation_facade import NavigationFacade


class NavigationFacadeMock(NavigationFacade):
    """
    Sends a periodic dummy navigation message
    """
    def __init__(self, dds: DdsPubSub, logger: Logger, navigation_plan_msg: NavigationPlanMsg):
        """
        :param dds: communication layer (DDS) instance
        :param logger: logger
        :param navigation_plan_msg: the navigation plan message to publish periodically
        """
        self._navigation_plan_msg = navigation_plan_msg
        super().__init__(dds, logger, None)

    def _periodic_action_impl(self):
        """
        Publishes the received messages initialized in init
        :return: void
        """
        self._publish_navigation_plan(self._navigation_plan_msg)
