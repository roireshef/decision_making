from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import NAVIGATION_PLAN_PUBLISH_TOPIC
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.navigation.navigation_facade import NavigationFacade
from logging import Logger
import numpy as np
from decision_making.src.planning.navigation.preplanned_navigation_planner import PreplannedNavigationPlanner


class NavigationFacadeMock(NavigationFacade):
    """
    Sends a periodic dummy navigation message
    """
    def __init__(self, dds: DdsPubSub, logger: Logger):

        navigation_plan = NavigationPlanMsg(np.array([1, 2]))
        handler = PreplannedNavigationPlanner(navigation_plan)
        super().__init__(dds, logger, handler)

    def _periodic_action_impl(self):
        self.__publish_navigation_plan(self.handler.plan(None, None))

    def __publish_navigation_plan(self, plan: NavigationPlanMsg):
        self.dds.publish(NAVIGATION_PLAN_PUBLISH_TOPIC, plan)
