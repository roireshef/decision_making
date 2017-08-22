from logging import Logger
from typing import List
import numpy as np
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.utils.geometry_utils import CartesianFrame
from decision_making.src.state.state import State, EgoState, DynamicObject
from decision_making.src.map.map_api import MapAPI


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


class BehavioralState:
    def __init__(self, logger: Logger, map_api: MapAPI, navigation_plan: NavigationPlanMsg, ego_state: EgoState = None,
                 dynamic_objects: List[DynamicObject] = None) -> None:
        """
        initialization of behavioral state. default values are None and empty list, because the logic for actual updates
        (coming from messages) is done in the update_behavioral_state method.
        :param map_api: cached map of type MapAPI
        :param navigation_plan:
        :param ego_state: ego_state of car
        :param dynamic_objects: list of dynamic objects from state
        """

        self.logger = logger

        # initiate behavioral state with cached map and initial navigation plan
        self.map = map_api
        self._navigation_plan = navigation_plan

        # private members, will be updated when new state arrives
        self._margin_info = None
        self._lane_object_information = None  # Array of LaneObjectInfo's

        # public members defining internal state, will be used by policy to choose action
        self.ego_state = ego_state
        self.last_ego_state_on_road = ego_state
        self.dynamic_objects = dynamic_objects

        self.current_timestamp = None
        self.current_position = None
        self.current_orientation = None
        self.current_velocity = None
        self.ego_off_road = None
        self.road_data = None

    def update_behavioral_state(self, state: State, navigation_plan: NavigationPlanMsg) -> None:
        """
        updating the behavioral state from the raw input state. This includes only direct processing without complex
        logic. This is implemented separately from initialization in order to potentially use differences for more
        efficient processing.
        :param state: the state coming as a message from perception via DDS.
        :param navigation_plan: will be used for processing the behavioral state, as well as for PolicyFeatures
        :return: void
        """

        # Update navigation plan
        self._navigation_plan = navigation_plan

        # Process ego_state
        self.ego_state = state.ego_state
        self.current_timestamp = self.ego_state._timestamp
        self.current_yaw = self.ego_state.yaw
        self.current_position = np.array([self.ego_state.x, self.ego_state.y, self.ego_state.z])
        self.current_orientation = np.array(CartesianFrame.convert_yaw_to_quaternion(self.ego_state.yaw))
        self.current_velocity = np.linalg.norm([self.ego_state.v_x, self.ego_state.v_y])

        # Save last known road localization if car is off road
        self.ego_off_road = self.ego_state.road_localization.road_id is None

        if self.ego_off_road:
            self.logger.warning("Car is off road! Keeping previous valid Behavioral state")
        else:
            self.last_ego_state_on_road = self.ego_state

        # Filter static & dynamic objects that are relevant to car's navigation
        self.dynamic_objects = []
        for dynamic_obj in state.dynamic_objects:
            # ignoring everything not in our path looking forward
            if dynamic_obj.rel_road_localization.rel_lon > 0:
                self.dynamic_objects.append(dynamic_obj)
