from logging import Logger
from typing import List

import numpy as np

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.constants import MIN_DISTANCE_OF_OBJECT_FROM_EGO_FOR_FILTERING, \
    MAX_DISTANCE_OF_OBJECT_FROM_EGO_FOR_FILTERING
from decision_making.src.state.state import State, EgoState, DynamicObject, RelativeRoadLocalization
from mapping.src.model.map_api import MapAPI
from mapping.src.transformations.geometry_utils import CartesianFrame


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

        self.logger = logger

        # Navigation
        self.map = map_api
        self.navigation_plan = navigation_plan

        # Ego state features
        self.ego_timestamp = ego_state.timestamp
        self.ego_state = ego_state

        self.ego_yaw = ego_state.yaw
        self.ego_position = np.array([ego_state.x, ego_state.y, ego_state.z])
        self.ego_orientation = np.array(CartesianFrame.convert_yaw_to_quaternion(ego_state.yaw))
        self.ego_velocity = np.linalg.norm([ego_state.v_x, ego_state.v_y])
        self.ego_road_id = ego_state.road_localization.road_id
        self.ego_on_road = ego_state.road_localization.road_id is not None
        if not self.ego_on_road:
            self.logger.warning("Car is off road.")

        # Dynamic objects and their relative locations
        self.dynamic_objects_on_road = dynamic_objects_on_road

    def update_behavioral_state(self, state: State, navigation_plan: NavigationPlanMsg):
        """
        This method updates the behavioral state according to the new world state and navigation plan.
         It fetches relevant features that will be used for the decision-making process.
        :param navigation_plan: new navigation plan of vehicle
        :param state: new world state
        :return: a new and updated BehavioralState
        """
        ego_state = state.ego_state

        # Filter static & dynamic objects that are relevant to car's navigation
        dynamic_objects_on_road = []
        for dynamic_obj in state.dynamic_objects:
            # Get object's relative road localization
            relative_road_localization = dynamic_obj.get_relative_road_localization(
                ego_road_localization=ego_state.road_localization, ego_nav_plan=navigation_plan,
                map_api=self.map, logger=self.logger)

            # filter objects with out of decision-making range
            if MAX_DISTANCE_OF_OBJECT_FROM_EGO_FOR_FILTERING > \
                    relative_road_localization.rel_lon > \
                    MIN_DISTANCE_OF_OBJECT_FROM_EGO_FOR_FILTERING:
                dynamic_object_on_road = DynamicObjectOnRoad(dynamic_object_properties=dynamic_obj,
                                                             relative_road_localization=relative_road_localization)
                dynamic_objects_on_road.append(dynamic_object_on_road)

        return BehavioralState(logger=self.logger, map_api=self.map, navigation_plan=navigation_plan,
                               ego_state=ego_state, dynamic_objects_on_road=dynamic_objects_on_road)


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
