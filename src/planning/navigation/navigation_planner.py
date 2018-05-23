from abc import ABCMeta, abstractmethod

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg

# TODO - must think about what the input to the navigation computation is, and where it comes from
from decision_making.src.state.state import MapState


class NavigationPlanner(metaclass=ABCMeta):
    @abstractmethod
    def plan(self, init_localization: MapState, target_localization: MapState) -> NavigationPlanMsg:
        pass

