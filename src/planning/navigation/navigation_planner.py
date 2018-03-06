from abc import ABCMeta, abstractmethod

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.state.state import RoadLocalization


# TODO - must think about what the input to the navigation computation is, and where it comes from
class NavigationPlanner(metaclass=ABCMeta):
    @abstractmethod
    def plan(self, init_localization: RoadLocalization, target_localization: RoadLocalization) -> NavigationPlanMsg:
        pass

