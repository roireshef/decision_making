import numpy as np
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import List, Dict, Optional
import rte.python.profiler as prof
from decision_making.src.messages.route_plan_message import DataRoutePlan
from decision_making.src.messages.scene_static_message import SceneStaticBase, NavigationPlan, \
    SceneRoadSegment, SceneLaneSegmentBase
from decision_making.src.exceptions import MissingInputInformation, RepeatedRoadSegments, raises,\
    NavigationSceneDataMismatch, LaneSegmentDataNotFound, RoadSegmentDataNotFound

RoadSegmentDict = Dict[int, SceneRoadSegment]
LaneSegmentBaseDict = Dict[int, SceneLaneSegmentBase]
RouteLaneSegmentOrderedDict = Dict[int, np.ndarray]   # Once typing.OrderedDict becomes availble (in python 3.7.2.) replace "Dict" with "OrderedDict" type


class RoutePlannerInputData():
    """
        This class takes navigation and map (base) data and converts it to a more useful form of data which resembles the final
        route plan 2D lane sequences and also keeps relevant data in dictionary containers for faster access
    """

    def __init__(self, route_lane_segment_ids: Optional[RouteLaneSegmentOrderedDict] = None,
                 route_lane_segments_base_as_dict: Optional[LaneSegmentBaseDict] = None,
                 route_road_segments_as_dict: Optional[RoadSegmentDict] = None,
                 next_road_segment_id: Optional[Dict[int, int]] = None,
                 prev_road_segment_id: Optional[Dict[int, int]] = None):

        """
        dict:   key - road segment IDs (ordered as in routeplan)
              value - np.ndarray(LaneSegmentID) (ordered as in the road segment structure in nav. plan)
        """
        self._route_lane_segment_ids = route_lane_segment_ids or OrderedDict()

        """
        dict:   key - lane segment ID
              value - LaneSegmentBase
        Should contain all the lane segments listed in nav. route road segments
        """
        self._route_lane_segments_base_as_dict = route_lane_segments_base_as_dict or {}

        """
        dict:   key - road segment ID
              value - Road Segments
        Should contain all the road segments listed in nav. route
        """
        self._route_road_segments_as_dict = route_road_segments_as_dict or {}

        """
        dict:   key - road segment ID,
              value - next road segment ID in nav. route
        Enables O(1) lookup of the next road segment.
        """
        self._next_road_segment_id = next_road_segment_id or {}

        """
        dict:   key - road segment ID,
              value - prev road segment ID in nav. route
        Enables O(1) lookup of the prev road segment
        """
        self._prev_road_segment_id = prev_road_segment_id or {}

    @staticmethod   # Made static method especially as this method doesn't access the classes states/variables
    @raises(MissingInputInformation)
    def check_scene_data_validity(scene: SceneStaticBase, nav_plan: NavigationPlan) -> None:
        if not scene.as_scene_lane_segments:
            raise MissingInputInformation("Route Planner Input Data Processing: Empty scene.as_scene_lane_segments")

        if not scene.as_scene_road_segment:
            raise MissingInputInformation("Route Planner Input Data Processing: Empty scene.as_scene_road_segment")

        if not nav_plan.a_i_road_segment_ids.size:  # np.ndarray type
            raise MissingInputInformation("Route Planner Input Data Processing: Empty NAV Plan")

    def _update_dict_data(self, scene: SceneStaticBase, nav_plan: NavigationPlan) -> None:
        """
         This method updates route_lane_segments_base_as_dict : all the lanesegment base structures for lane in the route, as a dictionary for fast access
                             route_road_segments_as_dict : all the roadsegments in the route as a dictionary for fast access
        """
        for scene_lane_segment in scene.as_scene_lane_segments:
            lane_segment_id = scene_lane_segment.e_i_lane_segment_id
            road_segment_id = scene_lane_segment.e_i_road_segment_id

            # Verify if these lane segs are in NAV route plan
            if road_segment_id in nav_plan.a_i_road_segment_ids:
                self._route_lane_segments_base_as_dict[lane_segment_id] = scene_lane_segment

        for scene_road_segment in scene.as_scene_road_segment:
            road_segment_id = scene_road_segment.e_i_road_segment_id

            # Verify if these road segs are in NAV route plan.
            if road_segment_id in nav_plan.a_i_road_segment_ids:    # Empty NAV Plan error would have been caught earlier
                self._route_road_segments_as_dict[road_segment_id] = scene_road_segment

    @raises(MissingInputInformation, RepeatedRoadSegments, NavigationSceneDataMismatch)
    def _update_routeplan_data(self, nav_plan: NavigationPlan) -> None:
        """
        This method updates route_lane_segments :
        an ordered dictionary: key -> road seg ids ordered as in route
                               value -> ndaray of lane seg ids (ordered) as stored in the road seg structure
        """
        # For the first road segment in the nav. plan, the previous road segment value will be "None"
        prev_road_segment_id = None

        """
        Loop over road segments, except for the last in the nav. plan, and assign the next and previous road segment IDs accordingly. Note that enumerate is configured to start counting the index (i) from 1 instead of 0. This was done so that no addition is needed each time the index is accessed.
        """
        for i, road_segment_id in enumerate(nav_plan.a_i_road_segment_ids[:-1], start=1):
            if road_segment_id in self._route_road_segments_as_dict:
                all_lane_segment_ids_in_this_road_segment = self._route_road_segments_as_dict[road_segment_id].a_i_lane_segment_ids

                # Since this is a input validity check for which we have to loop over all road segs, it will cost O(n) extra if we do it upfront.
                if all_lane_segment_ids_in_this_road_segment.size:
                    self._route_lane_segment_ids[road_segment_id] = all_lane_segment_ids_in_this_road_segment
                else:
                    raise MissingInputInformation('Route Planner Input Data Processing: no lane segments in road segment ID {0}'.format(road_segment_id))

            else:
                raise NavigationSceneDataMismatch('Road segement ID {0} reported in the NAV route not found in scene static base'.format(road_segment_id))

            self._next_road_segment_id[road_segment_id] = nav_plan.a_i_road_segment_ids[i]
            self._prev_road_segment_id[road_segment_id] = prev_road_segment_id

            prev_road_segment_id = road_segment_id

            # The same road segment can appear more than once in the the route indicating a loop, but should not appear consecutively
            if road_segment_id == nav_plan.a_i_road_segment_ids[i]:
                raise RepeatedRoadSegments("Route Planner Input Data Processing: Repeated segement reported in the NAV route ")

        # Assign values corresponding to the last road segment in the nav. plan
        last_road_segment_id = nav_plan.a_i_road_segment_ids[-1]

        if last_road_segment_id in self._route_road_segments_as_dict:
            all_lane_segment_ids_in_this_road_segment = self._route_road_segments_as_dict[last_road_segment_id].a_i_lane_segment_ids

            # Since this is a input validity check for which we have to loop over all road segs, it will cost O(n) extra if we do it upfront.
            if all_lane_segment_ids_in_this_road_segment.size:
                self._route_lane_segment_ids[last_road_segment_id] = all_lane_segment_ids_in_this_road_segment
            else:
                raise MissingInputInformation('Route Planner Input Data Processing: no lane segments in road segment ID {0}'.format(last_road_segment_id))

        else:
            raise NavigationSceneDataMismatch('Road segement ID {0} reported in the NAV route not found in scene static base'.format(last_road_segment_id))

        self._next_road_segment_id[last_road_segment_id] = None
        self._prev_road_segment_id[last_road_segment_id] = prev_road_segment_id

    @prof.ProfileFunction()
    def reformat_input_data(self, scene: SceneStaticBase, nav_plan: NavigationPlan) -> None:
        """
        This method updates route_lane_segments_base_as_dict : all the lanesegment base structures for lane in the route, as a dictionary for fast access
                            route_road_segments_as_dict : all the roadsegments in the route as a dictionary for fast access
                            route_lane_segments : an ordered dictionary: key -> road seg ids ordered as in route
                                                                         value -> ndaray of lane seg ids (ordered) as stored in the road seg structure
                            by invoking _update_dict_data and _update_routeplan_data methods in that order. Look at the method comments for more
                            details.
        """

        RoutePlannerInputData.check_scene_data_validity(scene, nav_plan)
        # maintain the following order
        self._update_dict_data(scene, nav_plan)
        self._update_routeplan_data(nav_plan)

    def get_nav_plan(self) -> List[int]:
        """
         This method returns List[road segment IDs] in the same sequence as that is in the NAV plan
        """
        return [road_segment_id for (road_segment_id, _) in self._route_lane_segment_ids.items()]

    @raises(LaneSegmentDataNotFound)
    def get_lane_segment_base(self, lane_segment_id: int) -> SceneLaneSegmentBase:
        """
         This method returns lane segment base given a lane segment ID
        """
        if lane_segment_id in self._route_lane_segments_base_as_dict:
            # Access all the lane segment lite data from lane segment dict
            current_lane_segment_base_data = self._route_lane_segments_base_as_dict[lane_segment_id]
        else:
            raise LaneSegmentDataNotFound('Cost Based Route Planner: Lane segment ID {0} not found in route_lane_segments_base_as_dict'.format(lane_segment_id))

        return current_lane_segment_base_data

    @raises(LaneSegmentDataNotFound)
    def get_lane_segment_ids_for_route(self) -> RouteLaneSegmentOrderedDict:
        """
         This method returns entire RouteLaneSegmentOrderedDict, which is an ordered Dict of
                        key - road segment IDs (ordered as in routeplan),
                        value - np.ndarray(LaneSegmentID) (ordered as in the road segment structure )
        """
        if not self._route_lane_segment_ids:
            raise LaneSegmentDataNotFound("Cost Based Route Planner: Trying to access empty route lane segment ids ")
        return self._route_lane_segment_ids

    @raises(RoadSegmentDataNotFound)
    def get_lane_segment_ids_for_road_segment(self, road_segment_id: int) -> np.ndarray:
        """
         This method returns np.ndarray(road_segment_id) (ordered as in the road segment structure )
        """
        if road_segment_id not in self._route_lane_segment_ids:
            raise RoadSegmentDataNotFound('Cost Based Route Planner: In _route_lane_segment_ids couldn\'t find road_segment_id {0}'.format(road_segment_id))

        return self._route_lane_segment_ids[road_segment_id]

    @raises(RoadSegmentDataNotFound)
    def get_next_road_segment(self, road_segment_id: int) -> int:
        """
         This method returns next road segment id of a given road segment id
        """
        if road_segment_id not in self._next_road_segment_id:
            raise RoadSegmentDataNotFound('Cost Based Route Planner: No entry for next road segment found for road segment ID {0}'.format(road_segment_id))


    @raises(RoadSegmentDataNotFound)
    def get_prev_road_segment(self, road_segment_id: int) -> int:
        """
         This method returns next road segment id of a given road segment id
        """
        if road_segment_id not in self._prev_road_segment_id:
            raise RoadSegmentDataNotFound('Cost Based Route Planner: No entry for previous road segment found for road segment ID {0}'.format(road_segment_id))

        self._prev_road_segment_id[road_segment_id]

    def __str__(self) -> str:
        """
         This method is a helper for pretty print of the RoutePlannerInputData(route_lane_segments only as the dictionaries are ususally not
         information we need to visualize for all road/lane segments at once ).
        """
        print_route_planner_input_data = "\n"
        for road_segment_id in self._route_lane_segment_ids:
            print_route_planner_input_data = print_route_planner_input_data + "roadseg:" + str(road_segment_id) + "\t"

        return print_route_planner_input_data


class RoutePlanner(metaclass=ABCMeta):
    """Abstract route planner class"""
    @abstractmethod
    def plan(self, route_plan_input_data: RoutePlannerInputData) -> DataRoutePlan:
        """Abstract route planner method. Implementation details will be in child class/methods """
        pass
