from src.messages.dds_message import DDSMessage
from src.planning.navigation.navigation_plan import NavigationPlan


class NavigationPlanMessage(DDSMessage):
    def __init__(self, _navigation_plan: NavigationPlan):
        self._navigation_plan = _navigation_plan

    @property
    def navigation_plan(self): return self._navigation_plan
