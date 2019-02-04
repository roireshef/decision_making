from abc import ABCMeta, abstractmethod
from cost_based_route_planner import RoutePlannerInputData
from common_data.interface.Rte_Types.python.sub_structures import TsSYSDataRoutePlan


class RoutePlanner(metaclass=ABCMeta):
    """Add comments"""

    @abstractmethod
    def plan(self,RouteData:RoutePlannerInputData)->TsSYSDataRoutePlan: # TODO: Set function annotaion
        """Add comments"""
        pass
