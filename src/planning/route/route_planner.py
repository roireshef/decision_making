import numpy as np
from numpy import ndarray
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import List, Dict

from decision_making.src.messages.route_plan_message import DataRoutePlan
from decision_making.src.messages.scene_static_message import SceneStaticBase, NavigationPlan, SceneRoadSegment, SceneLaneSegmentBase
from decision_making.src.exceptions import  RouteRoadSegmentNotFound, RepeatedRoadSegments, raises

RoadSegmentDict = Dict[int,SceneRoadSegment]
LaneSegmentBaseDict = Dict[int,SceneLaneSegmentBase]
# RouteLaneSegmentOrderedDict = OrderedDict[int,ndarray]  Once typing.OrderedDict becomes availble (in python 3.7.2.)


class RoutePlannerInputData():

    """
        This class takes navigation and map (base) data and converts it to a more useful form of data which resembles the final
        route plan 2D lane sequences and also keeps relevant data in dictionary containers for faster access

    """


    def __init__(self):

        self.route_lanesegments = OrderedDict()                     # dict:  key - road segment IDs (ordered as in routeplan),
                                                                    #        value - ndarray(LaneSegmentID) (ordered as in the road segment structure )
        self.route_lanesegs_base_as_dict:LaneSegmentBaseDict = {}   # dict: key - lane segment ID,
                                                                    #       value - LaneSegmentBase.
                                                                    # Should contain all the lane segments listed in Nav route road segments
        self.route_roadsegs_as_dict: RoadSegmentDict = {}           # dict: key - road segment ID,
                                                                    #       value - Road Segments.
                                                                    # Should contain all the road segments listed in Nav route




    def _update_dict_data(self, scene: SceneStaticBase, nav_plan: NavigationPlan)->None:
        """
         This method updates route_lanesegs_base_as_dict : all the lanesegment base structures for lane in the route, as a dictionary for fast access
                             route_roadsegs_as_dict : all the roadsegments in the route as a dictionary for fast access
        """
        for scene_lane_segment in scene.as_scene_lane_segments:
            lane_segment_id = scene_lane_segment.e_i_lane_segment_id
            road_segment_id = scene_lane_segment.e_i_road_segment_id

            # Verify if these lane segs are in route
            if road_segment_id in nav_plan.a_i_road_segment_ids:
                self.route_lanesegs_base_as_dict[lane_segment_id] = scene_lane_segment

        for scene_road_segment in scene.as_scene_road_segment:
            road_segment_id = scene_road_segment.e_i_road_segment_id

            # Verify if these road segs are in route.
            if road_segment_id in nav_plan.a_i_road_segment_ids:
                self.route_roadsegs_as_dict[road_segment_id] = scene_road_segment




    @raises(RepeatedRoadSegments)
    @raises(RouteRoadSegmentNotFound)
    def _update_routeplan_data(self, nav_plan: NavigationPlan)->None:
        """
        This method updates route_lanesegments : an ordered dictionary: key -> road seg ids ordered as in route
                                                                        value -> ndaray of lane seg ids (ordered) as stored in the road seg structure        """
        enumerated_road_segment_ids = list(enumerate(nav_plan.a_i_road_segment_ids))
        for road_segment_idx,road_segment_id in enumerated_road_segment_ids:
            # Since _update_routeplan_data is executed after _update_dict_data we must have stored all the road segs in the current scene
            # and route in the dict container. But the insertion loop in _update_dict_data was
            # for all roadseg in scene:
            #       if roadseg is in NAV route :
            #           insert in the dict
            #
            #  So this fails to check if all the road segs in NAV route are reported in the scene. We are doing that check and exception raise here
            if road_segment_id in self.route_roadsegs_as_dict:
                self.route_lanesegments[road_segment_id] = self.route_roadsegs_as_dict[road_segment_id].a_i_lane_segment_ids
            else:
                raise RouteRoadSegmentNotFound("Route Planner Input Data Processing: Road segement reported in the NAV route not found in \
                    scene static base")

            # The same road segment can appear more than once in the the route indicating a loop, but should not appear consecutively
            if road_segment_idx > 0:
                downstream_road_segment_id, _ = enumerated_road_segment_ids[road_segment_idx-1]
                if (downstream_road_segment_id == road_segment_id):
                    raise RepeatedRoadSegments("Route Planner Input Data Processing: Repeated segement reported in the NAV route ")



    def reformat_input_data(self, scene: SceneStaticBase, nav_plan: NavigationPlan)->None:
        """
        This method updates route_lanesegs_base_as_dict : all the lanesegment base structures for lane in the route, as a dictionary for fast access
                            route_roadsegs_as_dict : all the roadsegments in the route as a dictionary for fast access
                            route_lanesegments : an ordered dictionary: key -> road seg ids ordered as in route
                                                                        value -> ndaray of lane seg ids (ordered) as stored in the road seg structure
                            by invoking _update_dict_data and _update_routeplan_data methods in that order. Look at the method comments for more
                            details.
         Input is Scene Static Base and Navigation Plan
        """
        self._update_dict_data(scene, nav_plan) # maintain the following order
        self._update_routeplan_data(nav_plan)

    def __str__(self)->str:

        """
         This method is a helper for pretty print of the RoutePlannerInputData(route_lanesegments only as the dictionaries are ususally not
         information we need to visualize for all road/lane segments at once ). 
        """
        print_route_planner_input_data = "\n"
        for road_segment_id in self.route_lanesegments:
            print_route_planner_input_data = print_route_planner_input_data + "roadseg:" + str(road_segment_id) + "\t"
        #    print_route_planner_input_data = print_route_planner_input_data + str(lane_segment_ids) + "\n"

        return print_route_planner_input_data






class RoutePlanner(metaclass=ABCMeta):
    """Abstract route planner class"""

    @abstractmethod
    def plan(self, RouteData: RoutePlannerInputData) -> DataRoutePlan:
        """Abstract route planner method. Implementation details will be in child class/methods """
        pass

