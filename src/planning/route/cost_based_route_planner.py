from decision_making.src.messages.scene_static_lite_message import SceneStaticLite,DataSceneStaticLite

class RoutePlannerData():
    
    def __init__(self):
        self.route_segments = [] # list: ordered RouteSegment's upto the variable size e_Cnt_num_road_segments
        self.LaneSegments = {} # dict : key - segment ID, value - LaneSegmentLite. The dict should contain all the lane segments listed in self.route_segment_ids. 


    def plan(self,Scene:SceneStaticLite): # TODO: Set function annotaion
        # Create route plan
        pass

    def Update_LaneSegmentData(self,Scene:SceneStaticLite):
        for i in range(Scene.s_Data.e_Cnt_num_lane_segments):
            self.LaneSegments[Scene.s_Data.as_scene_lane_segment[i].e_i_lane_segment_id] = \
            Scene.s_Data.as_scene_lane_segment[i]
        pass

    def Update_RoutePlanData(self,Scene:SceneStaticLite):
        for i in range(Scene.s_Data.e_Cnt_num_road_segments):
            self.route_segments.append(Scene.s_Data.as_scene_lane_segment[i].e_i_lane_segment_id)
        pass


class CostBasedRoutePlanner(RoutePlanner):
    """Add comments"""

    def plan(self): # TODO: Set function annotaion
        """Add comments"""
        pass


