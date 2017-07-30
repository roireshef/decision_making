from src.planning.navigation.navigation_plan import NavigationPlan


class NavigationFacade:
    def __init__(self, navigation_plan: NavigationPlan):
        self._navigation_plan = navigation_plan

    # TODO - must think about what the input to the navigation computation is, and where it comes from
    def update_navigation_plan(self) -> None:
        """
        Recompute navigation plan and update _navigation_plan field
        :return: Void
        """
        pass

    # TODO implement message passing
    def __publish_navigation_plan(self):
        pass