from abc import ABCMeta, abstractmethod
from decision_making.src.messages.scene_static_lite_message import SceneStaticLite


class RoutePlanner():

    def __init__(self):
    self.route_segments = [] # list: ordered RouteSegment's upto the variable size e_Cnt_num_road_segments
    self.LaneSegments = {} # dict : key - segment ID, value - LaneSegmentLite. The dict should contain all the lane segments listed in self.route_segment_ids. 


    def plan(self,SceneStaticLite Scene): # TODO: Set function annotaion
        # Create route plan
        pass

    def Update_LaneSegmentData(self,SceneStaticLite Scene):
        for i in range(pubsubMsg.e_Cnt_num_lane_segments):
            self.LaneSegments = lane_segments.append(Scene.DataSceneStatic.)
        pass

    def Update_RoutePlanData(self,SceneStaticLite Scene):
        pass

