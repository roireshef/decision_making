from abc import ABCMeta, abstractmethod

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.state.map_state import MapState

# TODO - to be replaced with Route Planner


class NavigationPlanner(metaclass=ABCMeta):
    @abstractmethod
    def plan(self, init_localization: MapState, target_localization: MapState) -> NavigationPlanMsg:
        pass

