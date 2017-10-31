from logging import Logger
from typing import List

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.state.state import State, EgoState, DynamicObject, RelativeRoadLocalization
from mapping.src.model.map_api import MapAPI


class DynamicObjectOnRoad(DynamicObject):
    def __init__(self, dynamic_object_properties: DynamicObject, relative_road_localization: RelativeRoadLocalization):
        """
        This object hold the dynamic object and it's relative (to ego) localization on road
        :param dynamic_object_properties: the dynamic object state
        :param relative_road_localization: a relative road localization (relative to ego)
        """
        super().__init__(**dynamic_object_properties.__dict__)
        self.relative_road_localization = relative_road_localization


class BehavioralState:
    def __init__(self, logger: Logger, map_api: MapAPI, navigation_plan: NavigationPlanMsg, ego_state: EgoState,
                 dynamic_objects_on_road: List[DynamicObjectOnRoad]) -> None:
        """
        Behavioral state generates and stores relevant state features that will be used for planning
        :param logger: logger
        :param map_api: map API
        :param navigation_plan: car's navigation plan
        :param ego_state: updated ego state
        """
        pass


    def update_behavioral_state(self, state: State, navigation_plan: NavigationPlanMsg):
        """
        This method updates the behavioral state according to the new world state and navigation plan.
         It fetches relevant features that will be used for the decision-making process.
        :param navigation_plan: new navigation plan of vehicle
        :param state: new world state
        :return: a new and updated BehavioralState
        """
        pass


class MarginInfo:
    def __init__(self, right_width: float, right_clear: bool, left_width: float, left_clear: bool) -> None:
        """
        All information regarding the margins of the road. used to differentiate between the shoulder and the end of the
         road.
        :param right_width: width of right margin, in meters
        :param right_clear: is the right margin clear
        :param left_width: width of left margin, in meters
        :param left_clear: is the left margin clear
        """
        self.right_width = right_width
        self.right_clear = right_clear
        self.left_width = left_width
        self.left_clear = left_clear


class LaneObjectInfo:
    def __init__(self, relative_velocity_of_closest_object: float, time_distance_of_closest_object: float,
                 confidence: float = None) -> None:
        """
        object describing the relevant information for a drivable lane.
        :param relative_velocity_of_closest_object: looking forward, what is the relative velocity of the closest object
         in the lane.
        :param time_distance_of_closest_object: looking forward, what is the time it would take to reach the closest
        object in the lane (given ego's current speed)
        :param confidence:
        """
        self.relative_velocity_of_closest_object = relative_velocity_of_closest_object
        self.time_distance_of_closest_object = time_distance_of_closest_object
        self.confidence = confidence
