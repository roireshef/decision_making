from logging import Logger
from typing import List, Union
import numpy as np
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.utils.geometry_utils import CartesianFrame
from decision_making.src.state.state import State, EgoState, DynamicObject, RelativeRoadLocalization
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
    def __init__(self, logger: Logger, map_api: MapAPI, navigation_plan: NavigationPlanMsg, ego_state: EgoState,
                 timestamp: int, ego_position: np.array, ego_orientation: np.array, ego_yaw: float, ego_velocity: float,
                 ego_off_road: bool, dynamic_objects: List[DynamicObject],
                 dynamic_objects_relative_localization: List[RelativeRoadLocalization]) -> None:
        """
        Behavioral state generates and stores relevant state features that will be used for planning
        :param logger:
        :param map_api:
        :param navigation_plan:
        :param ego_state:
        :param timestamp:
        :param ego_position:
        :param ego_orientation:
        :param ego_yaw:
        :param ego_velocity:
        :param ego_off_road:
        """

        self.logger = logger

        # Navigation
        self.map = map_api
        self.navigation_plan = navigation_plan

        # Ego state features
        self.timestamp = timestamp
        self.ego_state = ego_state
        self.ego_position = ego_position
        self.ego_orientation = ego_orientation
        self.ego_yaw = ego_yaw
        self.ego_velocity = ego_velocity
        self.ego_off_road = ego_off_road
        self.dynamic_objects = dynamic_objects
        self.dynamic_objects_relative_localization = dynamic_objects_relative_localization

    @classmethod
    def update_behavioral_state(cls, logger: Logger, map_api: MapAPI, navigation_plan: NavigationPlanMsg, state: State):
        ego_state = state.ego_state
        timestamp = ego_state._timestamp
        ego_yaw = ego_state.yaw
        ego_position = np.array([ego_state.x, ego_state.y, ego_state.z])
        ego_orientation = np.array(CartesianFrame.convert_yaw_to_quaternion(ego_state.yaw))
        ego_velocity = np.linalg.norm([ego_state.v_x, ego_state.v_y])

        # Save last known road localization if car is off road
        ego_off_road = ego_state.road_localization.road_id is None
        if ego_off_road:
            logger.warning("Car is off road! Keeping previous valid Behavioral state")

        # Filter static & dynamic objects that are relevant to car's navigation
        dynamic_objects = []
        for dynamic_obj in state.dynamic_objects:
            # Get object's relative road localization
            object_relative_road_localization = dynamic_obj.
            # ignoring everything not in our path looking forward
            if dynamic_obj.rel_road_localization.rel_lon > 0:
                dynamic_objects.append(dynamic_obj)

        return cls(logger=logger, map_api=map_api, navigation_plan=navigation_plan,
                   ego_state=ego_state, timestamp=timestamp, ego_position=ego_position,
                   ego_orientation=ego_orientation, ego_yaw=ego_yaw, ego_velocity=ego_velocity,
                   ego_off_road=ego_off_road, dynamic_objects_relative_localization=dynamic_objects_relative_localization)
