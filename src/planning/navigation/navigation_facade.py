from src.planning.navigation.constants import ONE_TWO_NAVIGATION_PLAN
from src.planning.navigation.navigation_plan import NavigationPlan


class NavigationFacade:
    def __init__(self, navigation_plan: NavigationPlan):
        self._navigation_plan = navigation_plan

    # TODO - must think about what the input to the navigation computation is, and where it comes from
    def update_navigation_plan(self) -> None:
        """
        Recompute navigation plan and update _navigation_plan field. For now, takes a constant plan from configuration.
        :return: Void
        """
        self._navigation_plan = NavigationPlan(ONE_TWO_NAVIGATION_PLAN)

    # TODO implement message passing
    def __publish_navigation_plan(self):
        pass