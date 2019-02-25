from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from decision_making.src.messages.route_plan_message import DataRoutePlan
from decision_making.src.messages.scene_static_message import SceneStaticBase, NavigationPlan

class RoutePlannerInputData():

    def __init__(self, scene: SceneStaticBase, nav_plan: NavigationPlan):
        #self.route_roadsegments = [] # list: ordered RouteSegment's upto the variable size e_Cnt_num_road_segments
        self.route_lanesegments = OrderedDict() # dict:  key - road segment IDs (sorted as in routeplan) , value - list(LaneSegmentID)
        self.route_lanesegs_base_as_dict = {} # dict : key - lane segment ID, value - LaneSegmentBase.
        #The dict should contain all the lane segments listed in self.route_segment_ids.
        self.route_roadsegs_as_dict = {} # dict : key - road segment ID, value - Road Segments.
        #The dict should contain all the lane segments listed in self.route_segment_ids.
        self._update_dict_data(scene)
        self._update_routeplan_data(nav_plan)

        # This method updates route_lanesegs_base_as_dict : all the lanesegment base structures for lane in the route, as a dictionary for fast access
        #                     route_roadsegs_as_dict : all the roadsegments in the route as a dictionary for fast access
        # Input is Scene Static Base
    def _update_dict_data(self, scene: SceneStaticBase)->None:
        for i in range(scene.e_Cnt_num_lane_segments):
            e_i_lane_segment_id = scene.as_scene_lane_segments[i].e_i_lane_segment_id
            self.route_lanesegs_base_as_dict[e_i_lane_segment_id] = scene.as_scene_lane_segments[i]
        for i in range(scene.e_Cnt_num_road_segments):
            e_i_road_segment_id = scene.as_scene_road_segment[i].e_i_road_segment_id
            self.route_roadsegs_as_dict[e_i_road_segment_id] = scene.as_scene_road_segment[i]

    def _update_routeplan_data(self, nav_plan: NavigationPlan)->None:
        for road_seg_idx in range(nav_plan.e_Cnt_num_road_segments):
            road_seg = nav_plan.a_i_road_segment_ids[road_seg_idx]
            #self.route_roadsegments.append(road_seg)
            laneseg_ids_in_this_roadseg = []
            #print("route_roadsegments",self.route_roadsegments)
            for j in range(self.route_roadsegs_as_dict[road_seg].e_Cnt_lane_segment_id_count):
                laneseg_ids_in_this_roadseg.append(self.route_roadsegs_as_dict[road_seg].a_i_lane_segment_ids[j])
            self.route_lanesegments[road_seg]=laneseg_ids_in_this_roadseg

class RoutePlanner(metaclass=ABCMeta):
    """Abstract route planner class"""

    @abstractmethod
    def plan(self, RouteData: RoutePlannerInputData) -> DataRoutePlan: # TODO: Set function annotaion
        """Abstract route planner method. Implementation details will be in child class/methods """
        pass

