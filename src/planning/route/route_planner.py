from abc import ABCMeta, abstractmethod
from cost_based_route_planner import RoutePlannerInputData
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures import TsSYS_DataRoutePlan


class RoutePlanner(metaclass=ABCMeta):
    """Add comments"""

    @abstractmethod
    def plan(self,RouteData:RoutePlannerInputData)->TsSYS_DataRoutePlan: # TODO: Set function annotaion
        """Add comments"""
        pass
