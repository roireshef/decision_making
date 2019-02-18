from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from decision_making.src.messages.route_plan_message import DataRoutePlan
from decision_making.src.messages.scene_static_message import SceneStaticBase, NavigationPlan

class RoutePlannerInputData():
        
    def __init__(self, scene: SceneStaticBase, nav: NavigationPlan):
        self.route_roadsegments = [] # list: ordered RouteSegment's upto the variable size e_Cnt_num_road_segments
        self.route_lanesegments = OrderedDict() # dict:  key - road segment IDs (sorted as in routeplan) , value - list(LaneSegmentID)
        self.route_lanesegs_base_as_dict = {} # dict : key - lane segment ID, value - LaneSegmentBase. 
        #The dict should contain all the lane segments listed in self.route_segment_ids. 
        self.route_roadsegs_as_dict = {} # dict : key - road segment ID, value - Road Segments. 
        #The dict should contain all the lane segments listed in self.route_segment_ids. 
        self._update_dict_data(scene)
        self._update_routeplan_data(nav)
    
    def _update_dict_data(self, scene: SceneStaticBase)->None:
        for i in range(scene.e_Cnt_num_lane_segments):
            e_i_lane_segment_id = scene.as_scene_lane_segments[i].e_i_lane_segment_id
            self.route_lanesegs_base_as_dict[e_i_lane_segment_id] = scene.as_scene_lane_segments[i]
        for i in range(scene.e_Cnt_num_road_segments):
            e_i_road_segment_id = scene.as_scene_road_segment[i].e_i_road_segment_id
            self.route_roadsegs_as_dict[e_i_road_segment_id] = scene.as_scene_road_segment[i]
        
    def _update_routeplan_data(self, nav: NavigationPlan)->None:
        for road_seg_idx in range(nav.e_Cnt_num_road_segments):
            road_seg = nav.a_i_road_segment_ids[road_seg_idx]
            self.route_roadsegments.append(road_seg)
            laneseg_ids_in_this_roadseg = []
            #print("route_roadsegments",self.route_roadsegments)
            for j in range(self.route_roadsegs_as_dict[road_seg].e_Cnt_lane_segment_id_count):
                laneseg_ids_in_this_roadseg.append(self.route_roadsegs_as_dict[road_seg].a_i_lane_segment_ids[j])
            self.route_lanesegments[road_seg]=laneseg_ids_in_this_roadseg


    @abstractmethod
    def plan(self, RouteData: RoutePlannerInputData) -> DataRoutePlan: # TODO: Set function annotaion
        """Add comments"""
        pass

