from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.navigation.navigation_planner import NavigationPlanner
from decision_making.src.state.map_state import MapState


class PreplannedNavigationPlanner(NavigationPlanner):
    def __init__(self, navigation_plan: NavigationPlanMsg):
        self.navigation_plan = navigation_plan

    def plan(self, init_localization: MapState, target_localization: MapState) -> NavigationPlanMsg:
        return self.navigation_plan
