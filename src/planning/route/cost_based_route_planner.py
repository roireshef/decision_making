from decision_making.src.messages.scene_static_lite_message import SceneStaticLite,DataSceneStaticLite

class RoutePlannerData():

    def Update_DictData(self,Scene:SceneStaticLite):
        for i in range(Scene.s_SceneStaticData.e_Cnt_num_lane_segments):
            self.LaneSegmentDict[Scene.s_SceneStaticData.as_scene_lane_segment[i].e_i_lane_segment_id] = \
            Scene.s_SceneStaticData.as_scene_lane_segment[i]
        for i in range(Scene.s_SceneStaticData.e_Cnt_num_road_segments):
            self.RoadSegmentDict[Scene.s_SceneStaticData.as_scene_road_segment[i].e_Cnt_road_segment_id] = \
            Scene.s_SceneStaticData.as_scene_road_segment[i]
    pass

    def Update_RoutePlanData(self,Scene:SceneStaticLite):
        self.route_roadsegments = Scene.s_NavigationPlanData.a_i_road_segment_ids
        for road_seg in range(self.route_roadsegments):
            lane_seg = []
            for j in range(self.RoadSegmentDict[road_seg].e_Cnt_lane_segment_id_count):
                lane_seg.append(self.RoadSegmentDict[road_seg].a_Cnt_lane_segment_id[j])
            self.RoadSegmentDict[road_seg]=lane_seg

    def __init__(self,Scene:SceneStaticLite):
        self.route_roadsegments = [] # list: ordered RouteSegment's upto the variable size e_Cnt_num_road_segments
        self.route_lanesegments = {} # dict:  key - road segment ID, value - list(LaneSegmentID)
        self.LaneSegmentDict = {} # dict : key - road segment ID, value - LaneSegmentLite.
        #The dict should contain all the lane segments listed in self.route_segment_ids.
        self.RoadSegmentDict = {} # dict : key - road segment ID, value - Road Segments.
        #The dict should contain all the lane segments listed in self.route_segment_ids.
        self.Update_DictData(Scene)
        self.Update_RoutePlanData(Scene)





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

    def plan(self,RouteData:RoutePlannerData): # TODO: Set function annotaion
        """Add comments"""
        for seg in range(RouteData.route_lanesegments):
            for laneseg in range(seg):
                lanesegData = RouteData.LaneSegmentDict[laneseg]
                for Idx in range(lanesegData.e_Cnt_num_active_lane_attributes):
                    LaneAttrIdx = lanesegData.e_i_active_lane_attribute_indices[Idx]
                    LaneAttr = lanesegData.e_cmp_lane_attributes[LaneAttrIdx]
                    LaneAttrConf = lanesegData.e_cmp_lane_attribute_confidences[LaneAttrIdx]
                    if (LaneAttrConf<0.7):
                        continue
                    switch (LaneAttr){
                        case 0:

                        }
        pass


