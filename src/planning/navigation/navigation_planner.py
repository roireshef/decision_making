from abc import ABCMeta, abstractmethod

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.state.map_state import MapState

# TODO - must think about what the input to the navigation computation is, and where it comes from

class RouteSegment:
    def __init__(self,road_segment_id: int, lanes_ids: List[int]):
    self.road_segment_id = road_segment_id
    self.lane_segment_ids = lanes_ids # List of lane segment IDS

class RoutePlanData(metaclass=ABCMeta):
    def __init__(self,RouteSegs: List[RouteSegment]):
    self.route_segments = RouteSegs # list: ordered RouteSegment's upto the variable size e_Cnt_num_road_segments
    self.LaneSegments = {} # dict : key - segment ID, value - LaneSegmentLite. The dict should contain all the lane segments listed in self.route_segment_ids. 

    @abstractmethod
    def Update_LaneSegmentData(self):
        pass

    @abstractmethod
    def Update_RoutePlan(self):
        pass

class NavigationPlanner(metaclass=ABCMeta):
    @abstractmethod
    def plan(self, init_localization: MapState, target_localization: MapState) -> NavigationPlanMsg:
        pass

