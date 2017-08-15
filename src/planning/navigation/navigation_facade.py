from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import NAVIGATION_PLAN_PUBLISH_TOPIC
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.navigation.navigation_planner import NavigationPlanner
from rte.python.logger.AV_logger import AV_Logger


# TODO - must think about what the input to the navigation computation is, and where it comes from
class NavigationFacade(DmModule):
    def __init__(self, dds: DdsPubSub, logger: AV_Logger, handler: NavigationPlanner):
        super().__init__(dds, logger)
        self.handler = handler

    def _stop_impl(self):
        pass

    def _start_impl(self):
        pass

    def _periodic_action_impl(self):
        self.__publish_navigation_plan(self.handler.plan())

    def __publish_navigation_plan(self, plan: NavigationPlanMsg):
        self.dds.publish(NAVIGATION_PLAN_PUBLISH_TOPIC, plan)
