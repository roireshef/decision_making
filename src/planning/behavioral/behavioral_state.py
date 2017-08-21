from logging import Logger
from typing import Union, List
import numpy as np
from pygments.unistring import Lo

from decision_making.src.global_constants import BEHAVIORAL_STATE_NAME_FOR_LOGGING
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.state.state import State, EgoState, DynamicObject, ObjectSize
from decision_making.src import global_constants
from decision_making.src.map.map_api import MapAPI
from rte.python.logger.AV_logger import AV_Logger


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
    def __init__(self, logger: Logger, cached_map: MapAPI, navigation_plan: NavigationPlanMsg) -> None:
        """
        initialization of behavioral state. default values are None and empty list, because the logic for actual updates
        (coming from messages) is done in the update_behavioral_state method.
        :param cached_map: cached map of type MapAPI
        :param navigation_plan:
        """

        self.logger = logger

        # initiate behavioral state with cached map and initial navigation plan
        self.map = cached_map
        self._navigation_plan = navigation_plan

        # private members, will be updated when new state arrives
        self._margin_info = None
        self._lane_object_information = None  # Array of LaneObjectInfo's

        # public members defining internal state, will be used by policy to choose action
        self.ego_state = EgoState(obj_id=0, timestamp=0, x=0.0, y=0.0, z=0.0, yaw=0.0,
                                  size=ObjectSize(length=0.0, width=0.0, height=0.0), confidence=0.0, v_x=0, v_y=0,
                                  acceleration_lon=0.0, yaw_deriv=0.0, steering_angle=0.0, map_api=cached_map)

        self.current_timestamp = None
        self.current_yaw = None
        self.current_position = None
        self.current_orientation = None
        self.current_velocity = None
        self.current_road_id = None
        self.current_lane = None
        self.current_lat = None
        self.current_long = None
        self.ego_off_road = None
        self.road_data = None
        self.dynamic_objects = [DynamicObject(obj_id=0, timestamp=0, x=0.0, y=0.0, z=0.0, yaw=0.0,
                                              size=ObjectSize(length=0.0, width=0.0, height=0.0), confidence=0.0, v_x=0,
                                              v_y=0, acceleration_lon=0.0, yaw_deriv=0.0, map_api=cached_map)]

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
        self.current_orientation = np.array(self.ego_state.getOrientationQuaternion())
        self.current_velocity = np.sqrt(self.ego_state.v_x ** 2 + self.ego_state.v_y ** 2)

        # Process map-related localization
        ego_road_id, ego_lane, ego_full_lat, _, ego_long, _ = self.map.convert_world_to_lat_lon(
            x=self.current_position[0], y=self.current_position[1], z=0.0, yaw=0.0)

        ego_off_road = ego_road_id is None

        if ego_off_road:
            self.ego_off_road = True
            self.logger.info("Car is off road! Keeping previous valid Behavioral state")
        else:
            self.current_road_id = ego_road_id
            self.current_lane = ego_lane
            self.current_lat = ego_full_lat
            self.current_long = ego_long
            self.ego_off_road = False
            self.road_data = self.map.get_road_details(self.current_road_id)

        # TODO: get actual length and width, relative to objects's orientation
        OBJECT_CONST_WIDTH_IN_METERS = 1.2
        OBJECT_CONST_LENGTH_IN_METERS = 1.7

        # Filter static & dynamic objects that are relevant to car's navigation
        self.dynamic_objects = []
        for dynamic_obj in state.dynamic_objects:
            if dynamic_obj.rel_road_localization.rel_lon is not None:  # ignoring everything not in our path looking forward
                if dynamic_obj.rel_road_localization.rel_lon > 0:
                    self.dynamic_objects.append(dynamic_obj)
