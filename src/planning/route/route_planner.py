from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from decision_making.src.messages.route_plan_message import DataRoutePlan
from decision_making.src.messages.scene_static_message import SceneStaticBase, NavigationPlan

class RoutePlannerInputData():

    def __init__(self):
        """
        TODO: Add Comments
        """
        self.route_lanesegments = OrderedDict() # dict:  key - road segment IDs (sorted as in routeplan), value - list(LaneSegmentID)
        self.route_lanesegs_base_as_dict = {}   # dict: key - lane segment ID, value - LaneSegmentBase.
                                                # Should contain all the lane segments listed in self.route_segment_ids
        self.route_roadsegs_as_dict = {}    # dict: key - road segment ID, value - Road Segments.
                                            # Should contain all the road segments listed in self.route_segment_ids TODO is this correct?

    def _update_dict_data(self, scene: SceneStaticBase, nav_plan: NavigationPlan)->None:
        """
        TODO: Add Comments
        """
        for scene_lane_segment in scene.as_scene_lane_segments:
            e_i_lane_segment_id = scene_lane_segment.e_i_lane_segment_id
            e_i_road_segment_id = scene_lane_segment.e_i_road_segment_id

            # Verify if these road segs are in route
            if e_i_road_segment_id in nav_plan.a_i_road_segment_ids:
                # At this point the assumption is a complete road segment is in/not in the route. It is not assumed only a partial list of
                # lane segments of a roadsegment could be in the route.
                self.route_lanesegs_base_as_dict[e_i_lane_segment_id] = scene_lane_segment

        for scene_road_segment in scene.as_scene_road_segment:
            e_i_road_segment_id = scene_road_segment.e_i_road_segment_id

            # Verify if these road segs are in route.
            if e_i_road_segment_id in nav_plan.a_i_road_segment_ids:
                self.route_roadsegs_as_dict[e_i_road_segment_id] = scene_road_segment


    def _update_routeplan_data(self, nav_plan: NavigationPlan)->None:
        """
        TODO: Add Comments
        """
        for road_segment_id in nav_plan.a_i_road_segment_ids:
            self.route_lanesegments[road_segment_id] = self.route_roadsegs_as_dict[road_segment_id].a_i_lane_segment_ids.tolist()
            # TODO replace tolist() with something appropriate

    def reformat_input_data(self, scene: SceneStaticBase, nav_plan: NavigationPlan):
        """
        TODO: Add Comments
        This method updates route_lanesegs_base_as_dict : all the lanesegment base structures for lane in the route, as a dictionary for fast access
                            route_roadsegs_as_dict : all the roadsegments in the route as a dictionary for fast access
                            route_lanesegments : TODO
         Input is Scene Static Base and Navigation Plan (to filter only the road and lane segments that are in the navigation route)
        """
        self._update_dict_data(scene, nav_plan) # maintain the following order
        self._update_routeplan_data(nav_plan)

class RoutePlanner(metaclass=ABCMeta):
    """Abstract route planner class"""

    @abstractmethod
    def plan(self, RouteData: RoutePlannerInputData) -> DataRoutePlan: # TODO: Set function annotaion
        """Abstract route planner method. Implementation details will be in child class/methods """
        pass

