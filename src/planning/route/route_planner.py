from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from decision_making.src.messages.route_plan_message import DataRoutePlan
from decision_making.src.messages.scene_static_message import SceneStatic,DataSceneStaticLite,DataNavigationPlan

class RoutePlannerInputData():
        
    def __init__(self, Scene: DataSceneStaticLite, Nav: DataNavigationPlan):
        self.route_roadsegments = [] # list: ordered RouteSegment's upto the variable size e_Cnt_num_road_segments
        self.route_lanesegments = OrderedDict() # dict:  key - road segment IDs (sorted as in routeplan) , value - list(LaneSegmentID)
        self.LaneSegmentDict = {} # dict : key - lane segment ID, value - LaneSegmentLite. 
        #The dict should contain all the lane segments listed in self.route_segment_ids. 
        self.RoadSegmentDict = {} # dict : key - road segment ID, value - Road Segments. 
        #The dict should contain all the lane segments listed in self.route_segment_ids. 
        self.Update_DictData(Scene)
        self.Update_RoutePlanData(Nav)
    
    def Update_DictData(self, Scene: DataSceneStaticLite):
        for i in range(Scene.e_Cnt_num_lane_segments):
            self.LaneSegmentDict[Scene.as_scene_lane_segments[i].e_i_lane_segment_id] = \
            Scene.as_scene_lane_segments[i]
        for i in range(Scene.e_Cnt_num_road_segments):
            self.RoadSegmentDict[Scene.as_scene_road_segment[i].e_i_road_segment_id] = \
            Scene.as_scene_road_segment[i]
    pass

    def Update_RoutePlanData(self, Nav: DataNavigationPlan):
        self.route_roadsegments = Nav.a_i_road_segment_ids
        for road_seg in self.route_roadsegments:
            lane_seg = []
            for j in range(self.RoadSegmentDict[road_seg].e_Cnt_lane_segment_id_count):
                lane_seg.append(self.RoadSegmentDict[road_seg].a_i_lane_segment_ids[j])
            self.RoadSegmentDict[road_seg]=lane_seg

class RoutePlanner(metaclass=ABCMeta):
    """Add comments"""

    @abstractmethod
    def plan(self, RouteData: RoutePlannerInputData) -> DataRoutePlan: # TODO: Set function annotaion
        """Add comments"""
        pass
