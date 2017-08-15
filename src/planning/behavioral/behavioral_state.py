from typing import Union

from decision_making.src import global_constants
from decision_making.src.state.enriched_state import EnrichedState, EnrichedDynamicObject, EnrichedObjectState
import numpy as np


class ObjectOnRoad():
    def __init__(self, enriched_object: Union[EnrichedDynamicObject, EnrichedObjectState], road_latitude: float,
                 lane: int, relative_longitude_to_ego: float):
        """
        This class is an extension of EnrichedDynamicObject. It enriches each object on the road with additional
        parameters relating to the objects location relative to the ego state. It therefore depends on the navigation
        plan.
        :param enriched_object: the original object
        :param road_latitude: full latitude on road [m]
        :param lane: lane number on road [integer]
        :param relative_longitude_to_ego: relative longitudinal distance in [m] from ego to object\
                (according to driving route)
        """
        v_x = 0.0
        v_y = 0.0
        acceleration_x = 0.0
        turn_radius = None

        if isinstance(enriched_object, EnrichedDynamicObject):
            v_x = enriched_object.v_x
            v_y = enriched_object.v_y
            acceleration_x = enriched_object.acceleration_x
            turn_radius = enriched_object.turn_radius

        EnrichedDynamicObject.__init__(self, enriched_object.obj_id, enriched_object.timestamp,
                                       enriched_object.x, enriched_object.y, enriched_object.z,
                                       enriched_object.yaw, enriched_object.size,
                                       enriched_object.road_localization, enriched_object.confidence,
                                       enriched_object.localization_confidence,
                                       v_x, v_y,
                                       acceleration_x, turn_radius)

        self.road_latitude = road_latitude
        self.lane = lane
        self.relative_longitude_to_ego = relative_longitude_to_ego

    def predict(self, timestamp: int, map: MapAPI, navigation_plan: NavigationPlan):
        """
        This prediction will estimate object's location according to map and our navigation plan
        :param timestamp:
        :return:
        """
        pass


class MarginInfo:
    def __init__(self, right_width: float, right_clear: bool, left_width: float, left_clear: bool):
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
                 confidence: float = None):
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
    # TODO add type hint for navigation plan, once implemented
    def __init__(self, map: MapAPI, navigation_plan: NavigationPlan):
        """
        initialization of behavioral state. default values are None and empty list, because the logic for actual updates
        (coming from messages) is done in the update_behavioral_state method.
        :param map: cached map of type MapAPI
        :param navigation_plan:
        """

        # initiate behavioral state with cached map and initial navigation plan
        self.map = map
        self._navigation_plan = navigation_plan

        # private members, will be updated when new state arrives
        self._margin_info = None
        self._lane_object_information = None  # Array of LaneObjectInfo's

        # public members defining internal state, will be used by policy to choose action
        self.current_yaw = None
        self.current_position = None
        self.current_orientation = None
        self.current_velocity = None
        self.current_road_id = None
        self.current_lane = None
        self.current_lat = None
        self.current_long = None
        self.ego_off_road = None
        self.road_data = None  # of type (lanes_num, width, length, points)
        # each element in this list is of type (object_road_id, object_lane, object_full_lat, lon_distance_relative_to_ego)
        self.static_objects = None
        self.dynamic_objects = None

    def get_object_road_localization_relative_to_ego(self, target_object: Union[
        EnrichedDynamicObject, EnrichedObjectState]) -> (
            bool, float, float, int):
        object_road_id, lane, road_latitude, object_long, object_on_road = self.map.get_point_in_road_coordinates(
            X=target_object.x, Y=target_object.y, Z=0.0)

        if not object_on_road:
            return False, None, None, None

        lon_distance_relative_to_ego, found_connection = self.map.get_point_relative_longitude(
            to_road_id=object_road_id, to_lon_in_road=object_long, from_road_id=self.current_road_id,
            from_lon_in_road=self.current_long,
            max_lookahead_distance=global_constants.BEHAVIORAL_PLANNING_LOOKAHEAD_DIST)
        return found_connection, lon_distance_relative_to_ego, road_latitude, lane

    def update_behavioral_state(self, state: EnrichedState, navigation_plan: NavigationPlan) -> None:
        """
        updating the behavioral state from the raw input state. This includes only direct processing without complex
        logic. This is implemented separately from initialization in order to potentially use differences for more
        efficient processing.
        :param state: the enriched state coming as a message from perception via DDS.
        :param navigation_plan: will be used for processing the behavioral state, as well as for PolicyFeatures
        :return: void
        """

        # Update navigation plan
        self._navigation_plan = navigation_plan

        # Process ego_state
        ego_state = state.ego_state
        self.current_yaw = ego_state.yaw
        self.current_position = np.array([ego_state.x, ego_state.y, ego_state.z])
        self.current_orientation = np.array(ego_state.getOrientationQuaternion())
        self.current_velocity = np.sqrt(ego_state.v_x * ego_state.v_x + ego_state.v_y * ego_state.v_y)

        # Process map-related localization
        ego_road_id, ego_lane, ego_full_lat, ego_long, is_on_road = self.map.get_point_in_road_coordinates(
            X=self.current_position[0], Y=self.current_position[1], Z=self.current_position[2])
        ego_off_road = not is_on_road
        if ego_off_road:
            ego_road_id = navigation_plan.get_current_road_id()

        self.current_road_id = ego_road_id
        self.current_lane = ego_lane
        self.current_lat = ego_full_lat
        self.current_long = ego_long
        self.ego_off_road = ego_off_road
        self.road_data = self.map.get_road_details(self.current_road_id)

        # TODO: get actual length and width, relative to objects's orientation
        OBJECT_CONST_WIDTH_IN_METERS = 1.2
        OBJECT_CONST_LENGTH_IN_METERS = 1.7

        # Process static and dynamic objects
        self.static_objects = []
        for static_obj in state.static_objects:
            found_connection, lon_distance_relative_to_ego, road_latitude, lane = self.get_object_road_localization_relative_to_ego(
                state.map, navigation_plan, static_obj)
            if found_connection:  # ignoring everything not in our path looking forward
                self.static_objects.append(
                    ObjectOnRoad(enriched_object=static_obj, road_latitude=object_full_lat, lane=object_lane,
                                 relative_longitude_to_ego=lon_distance_relative_to_ego))

        self.dynamic_objects = []
        for dynamic_obj in state.dynamic_objects:
            found_connection, lon_distance_relative_to_ego, road_latitude, lane = self.get_object_road_localization_relative_to_ego(
                state.map, navigation_plan, dynamic_obj)
            if found_connection:  # ignoring everything not in our path looking forward
                self.dynamic_objects.append(
                    ObjectOnRoad(enriched_object=dynamic_obj, road_latitude=object_full_lat, lane=object_lane,
                                 relative_longitude_to_ego=lon_distance_relative_to_ego))
