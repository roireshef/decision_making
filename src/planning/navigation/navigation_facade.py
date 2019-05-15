from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.navigation.navigation_planner import NavigationPlanner
from logging import Logger

from decision_making.src.infra.pubsub import PubSub
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_NAVIGATION_PLAN


# TODO - must think about what the input to the navigation computation is, and where it comes from
class NavigationFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger, handler: NavigationPlanner):
        super().__init__(pubsub, logger)
        self.handler = handler

    def _stop_impl(self):
        pass

    def _start_impl(self):
        pass

    def _periodic_action_impl(self):
        self._publish_navigation_plan(self.handler.plan())

    def _publish_navigation_plan(self, plan: NavigationPlanMsg):
        self.pubsub.publish(UC_SYSTEM_NAVIGATION_PLAN, plan.serialize())


