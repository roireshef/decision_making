from logging import Logger

from interface.Rte_Types.python.uc_system import UC_SYSTEM_ROUTE_PLAN
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.route.route_planning_facade import RoutePlanningFacade


class RoutePlannerMock(RoutePlanningFacade):
    def __init__(self, pubsub: PubSub, logger: Logger, plan: RoutePlan):
        super().__init__(pubsub, logger, None)
        self.plan = plan

    def _periodic_action_impl(self) -> None:
        self.pubsub.publish(UC_SYSTEM_ROUTE_PLAN, self.plan.serialize())

    def _start_impl(self):
        pass

    def _stop_impl(self):
        pass