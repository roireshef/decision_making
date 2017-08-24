from logging import Logger
from typing import List, Union
import numpy as np
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.constants import MIN_DISTANCE_OF_OBJECT_FROM_EGO_FOR_FILTERING, \
    MAX_DISTANCE_OF_OBJECT_FROM_EGO_FOR_FILTERING
from decision_making.src.planning.utils.geometry_utils import CartesianFrame
from decision_making.src.state.state import State, EgoState, DynamicObject, RelativeRoadLocalization
from decision_making.src.map.map_api import MapAPI


class BehavioralState:
    def __init__(self, logger: Logger, map_api: MapAPI, navigation_plan: NavigationPlanMsg, ego_state: EgoState,
                 timestamp: int, ego_position: np.array, ego_orientation: np.array, ego_yaw: float, ego_velocity: float,
                 ego_on_road: bool, dynamic_objects: List[DynamicObject],
                 dynamic_objects_relative_localization: List[RelativeRoadLocalization]) -> None:
        """
        Behavioral state generates and stores relevant state features that will be used for planning
        :param logger: logger
        :param map_api: map API
        :param navigation_plan: car's navigation plan
        :param ego_state: updated ego state
        :param timestamp: of ego
        :param ego_position: np.array of (x,y,z) location of ego in [m]
        :param ego_orientation: np.array of (x,y,z,w) quaternion orientation og ego
        :param ego_yaw: yaw of ego in [rad]
        :param ego_velocity: velocity of ego in [m/s]
        :param ego_on_road: boolean flat that states whether car is located on road
        """

        self.logger = logger

        # Navigation
        self.map = map_api
        self.navigation_plan = navigation_plan

        # Ego state features
        self.ego_timestamp = timestamp
        self.ego_state = ego_state
        self.ego_position = ego_position
        self.ego_orientation = ego_orientation
        self.ego_yaw = ego_yaw
        self.ego_velocity = ego_velocity
        self.ego_off_road = ego_on_road

        # Dynamic objects and their relative locations
        self.dynamic_objects = dynamic_objects
        self.dynamic_objects_relative_localization = dynamic_objects_relative_localization

    def update_behavioral_state(self, state: State, navigation_plan: NavigationPlanMsg):
        """
        This method updates the behavioral state according to the new world state and navigation plan.
         It fetches relevant features that will be used for the decision-making process.
        :param navigation_plan: new navigation plan of vehicle
        :param state: new world state
        :return: a new and updated BehavioralState
        """
        ego_state = state.ego_state
        timestamp = ego_state._timestamp
        ego_yaw = ego_state.yaw
        ego_position = np.array([ego_state.x, ego_state.y, ego_state.z])
        ego_orientation = np.array(CartesianFrame.convert_yaw_to_quaternion(ego_state.yaw))
        ego_velocity = np.linalg.norm([ego_state.v_x, ego_state.v_y])

        # Save last known road localization if car is off road
        ego_on_road = ego_state.road_localization.road_id is not None
        if not ego_on_road:
            self.logger.warning("Car is off road.")

        # Filter static & dynamic objects that are relevant to car's navigation
        dynamic_objects = []
        dynamic_objects_relative_localization = []
        for dynamic_obj in state.dynamic_objects:
            # Get object's relative road localization
            relative_road_localization = dynamic_obj.get_relative_road_localization(
                ego_road_localization=ego_state.road_localization, ego_navigation_plan=navigation_plan,
                map_api=self.map, max_lookahead_dist=MAX_DISTANCE_OF_OBJECT_FROM_EGO_FOR_FILTERING)

            # filter objects with out of decision-making range
            if MAX_DISTANCE_OF_OBJECT_FROM_EGO_FOR_FILTERING > \
                    relative_road_localization.rel_lon > \
                    MIN_DISTANCE_OF_OBJECT_FROM_EGO_FOR_FILTERING:
                dynamic_objects.append(dynamic_obj)
                dynamic_objects_relative_localization.append(relative_road_localization)

        return BehavioralState(logger=self.logger, map_api=self.map, navigation_plan=navigation_plan,
                               ego_state=ego_state, timestamp=timestamp, ego_position=ego_position,
                               ego_orientation=ego_orientation, ego_yaw=ego_yaw, ego_velocity=ego_velocity,
                               ego_on_road=ego_on_road, dynamic_objects=dynamic_objects,
                               dynamic_objects_relative_localization=dynamic_objects_relative_localization)


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
