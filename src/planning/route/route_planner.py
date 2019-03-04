import numpy as np
from numpy import ndarray
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import List, Dict

from decision_making.src.messages.route_plan_message import DataRoutePlan
from decision_making.src.messages.scene_static_message import SceneStaticBase, NavigationPlan, SceneRoadSegment, SceneLaneSegmentBase
from decision_making.src.exceptions import  MissingInputInformation, RepeatedRoadSegments, raises

RoadSegmentDict = Dict[int,SceneRoadSegment]
LaneSegmentBaseDict = Dict[int,SceneLaneSegmentBase]
RouteLaneSegmentOrderedDict = Dict[int,ndarray] # Once typing.OrderedDict becomes availble (in python 3.7.2.) replace "Dict" with "OrderedDict" type



class RoutePlannerInputData():

    """
        This class takes navigation and map (base) data and converts it to a more useful form of data which resembles the final
        route plan 2D lane sequences and also keeps relevant data in dictionary containers for faster access

    """


    def __init__(self):

        self.route_lane_segments:RouteLaneSegmentOrderedDict = OrderedDict() # dict:  key - road segment IDs (ordered as in routeplan),
                                                                            #        value - ndarray(LaneSegmentID)
                                                                            #        (ordered as in the road segment structure )
        self.route_lane_segments_base_as_dict:LaneSegmentBaseDict = {}      # dict: key - lane segment ID,
                                                                            #       value - LaneSegmentBase.
                                                                            # Should contain all the lane segments listed in Nav route road segments
        self.route_roadsegs_as_dict: RoadSegmentDict = {}                   # dict: key - road segment ID,
                                                                            #       value - Road Segments.
                                                                            # Should contain all the road segments listed in Nav route


    @staticmethod # Made static method especially as this method doesn't access the classes states/variables
    @raises(MissingInputInformation)
    def check_scene_data_validity(scene: SceneStaticBase, nav_plan: NavigationPlan)->None:
        if not scene.as_scene_lane_segments:
            raise MissingInputInformation("Route Planner Input Data Processing:Empty scene.as_scene_lane_segments")

        if not scene.as_scene_road_segment:
            raise MissingInputInformation("Route Planner Input Data Processing:Empty scene.as_scene_road_segment")

        if not nav_plan.a_i_road_segment_ids.size: # ndarray type
            raise MissingInputInformation("Route Planner Input Data Processing:Empty NAV Plan")



    def _update_dict_data(self, scene: SceneStaticBase, nav_plan: NavigationPlan)->None:
        """
         This method updates route_lane_segments_base_as_dict : all the lanesegment base structures for lane in the route, as a dictionary for fast access
                             route_roadsegs_as_dict : all the roadsegments in the route as a dictionary for fast access
        """

        for scene_lane_segment in scene.as_scene_lane_segments:
            lane_segment_id = scene_lane_segment.e_i_lane_segment_id
            road_segment_id = scene_lane_segment.e_i_road_segment_id

            # Verify if these lane segs are in NAV route plan
            if road_segment_id in nav_plan.a_i_road_segment_ids:
                self.route_lane_segments_base_as_dict[lane_segment_id] = scene_lane_segment


        for scene_road_segment in scene.as_scene_road_segment:
            road_segment_id = scene_road_segment.e_i_road_segment_id

            # Verify if these road segs are in NAV route plan.
            if road_segment_id in nav_plan.a_i_road_segment_ids: # Empty NAV Plan error would have been caught earlier
                self.route_roadsegs_as_dict[road_segment_id] = scene_road_segment




    @raises(MissingInputInformation)
    @raises(RepeatedRoadSegments)
    @raises(KeyError)
    def _update_routeplan_data(self, nav_plan: NavigationPlan)->None:
        """
        This method updates route_lane_segments : an ordered dictionary: key -> road seg ids ordered as in route
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
                all_lane_segment_ids_in_this_road_segment = self.route_roadsegs_as_dict[road_segment_id].a_i_lane_segment_ids

                # Since this is a input validity check for which we have to loop over all road segs, it will cost O(n) extra if we do it upfront.
                if all_lane_segment_ids_in_this_road_segment.size:
                    self.route_lane_segments[road_segment_id] = all_lane_segment_ids_in_this_road_segment
                else:
                    raise MissingInputInformation("Route Planner Input Data Processing:Possible no lane segments in road segment ",road_segment_id)

            else:
                raise KeyError("Route Planner Input Data Processing: Road segement reported in the NAV route not found in \
                    scene static base for key:",road_segment_id)

            # The same road segment can appear more than once in the the route indicating a loop, but should not appear consecutively
            if road_segment_idx > 0:
                downstream_road_segment_id, _ = enumerated_road_segment_ids[road_segment_idx-1]
                if (downstream_road_segment_id == road_segment_id):
                    raise RepeatedRoadSegments("Route Planner Input Data Processing: Repeated segement reported in the NAV route ")



    def reformat_input_data(self, scene: SceneStaticBase, nav_plan: NavigationPlan)->None:
        """
        This method updates route_lane_segments_base_as_dict : all the lanesegment base structures for lane in the route, as a dictionary for fast access
                            route_roadsegs_as_dict : all the roadsegments in the route as a dictionary for fast access
                            route_lane_segments : an ordered dictionary: key -> road seg ids ordered as in route
                                                                        value -> ndaray of lane seg ids (ordered) as stored in the road seg structure
                            by invoking _update_dict_data and _update_routeplan_data methods in that order. Look at the method comments for more
                            details.
         Input is Scene Static Base and Navigation Plan
        """
        RoutePlannerInputData.check_scene_data_validity(scene,nav_plan)
        self._update_dict_data(scene, nav_plan) # maintain the following order
        self._update_routeplan_data(nav_plan)

    def __str__(self)->str:

        """
         This method is a helper for pretty print of the RoutePlannerInputData(route_lane_segments only as the dictionaries are ususally not
         information we need to visualize for all road/lane segments at once ).
        """
        print_route_planner_input_data = "\n"
        for road_segment_id in self.route_lane_segments:
            print_route_planner_input_data = print_route_planner_input_data + "roadseg:" + str(road_segment_id) + "\t"
        #    print_route_planner_input_data = print_route_planner_input_data + str(lane_segment_ids) + "\n"

        return print_route_planner_input_data




class RoutePlanner(metaclass=ABCMeta):
    """Abstract route planner class"""

    @abstractmethod
    def plan(self, RouteData: RoutePlannerInputData) -> DataRoutePlan:
        """Abstract route planner method. Implementation details will be in child class/methods """
        pass

