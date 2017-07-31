from src.messages.dds_message import DDSTypedMsg
from src.planning.navigation.navigation_plan import NavigationPlan


class NavigationPlanMsg(DDSTypedMsg):
    def __init__(self, _navigation_plan: NavigationPlan):
        self._navigation_plan = _navigation_plan

    @property
    def navigation_plan(self): return self._navigation_plan
