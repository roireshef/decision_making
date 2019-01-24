from decision_making.src.planning.route.route_planner import RoutePlanner


class RoutePlanData(metaclass=ABCMeta):
    def __init__(self,RouteSegs: List[RouteSegment]):
    self.route_segments = RouteSegs # list: ordered RouteSegment's upto the variable size e_Cnt_num_road_segments
    self.LaneSegments = {} # dict : key - segment ID, value - LaneSegmentLite. The dict should contain all the lane segments listed in self.route_segment_ids.

    @abstractmethod
    def Update_LaneSegmentData(self):
        pass

    @abstractmethod
    def Update_RoutePlanData(self):

        pass

class DualCostRoutePlanner(RoutePlanner):
    """Add comments"""

    def plan(self): # TODO: Set function annotaion


        pass


