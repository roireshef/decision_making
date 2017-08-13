from decision_making.src.state.enriched_state import EnrichedState, EgoState
import numpy as np


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
    def __init__(self, ego_state: EgoState = None, margin_info: MarginInfo = None,
                 lane_object_information: list = [], navigation_plan=None):
        """
        initialization of behavioral state. default values are None and empty list, because the logic for actual updates
        (coming from messages) is done in the update_behavioral_state method.
        :param ego_state: state of our ego vehicle, coming from EnrichedState
        :param margin_info: of type MarginInfo
        :param lane_object_information: list containing information regarding all the lanes of our current road. Each
        element is of type LaneObjectInfo
        :param navigation_plan:
        """
        self._ego_state = ego_state  # taken from the enriched state
        self._margin_info = margin_info
        self._lane_object_information = lane_object_information  # Array of LaneObjectInfo's
        self._navigation_plan = navigation_plan

    def update_behavioral_state(self, state: EnrichedState, navigation_plan) -> None:
        """
        updating the behavioral state from the raw input state. This includes only direct processing without complex
        logic. This is implemented separately from initialization in order to potentially use differences for more
        efficient processing.
        :param state: the enriched state coming as a message from perception via DDS.
        :param navigation_plan: will be used for processing the behavioral state, as well as for PolicyFeatures
        :return: void
        """
        if state is None:
            # Only happens on init
            return

        # updating information from ego_state
        ego_state = state.ego_state
        self._ego_state = ego_state
        self._current_yaw = ego_state.yaw
        self._current_position = np.array([ego_state.x, ego_state.y, ego_state.z])
        self._current_orientation = np.array(ego_state.getOrientationQuaternion())
        self._current_velocity = np.sqrt(ego_state.v_x * ego_state.v_x + ego_state.v_y * ego_state.v_y)

        semantic_db = state.semantic_db
        ###################
        # getting relevant information about our car from semantic DB
        ###################
        ego_road_id, ego_lane, ego_full_lat, ego_long, is_on_road = semantic_db.get_point_in_road_coordinates(
            X=self.current_position[0], Y=self.current_position[1], Z=self.current_position[2])
        ego_off_road = not is_on_road
        if ego_off_road:
            ego_road_id = state.semantic_db.navigation_plan.get_current_road_id()

        self.current_road_id = ego_road_id
        self.current_lane = ego_lane
        self.current_lat = ego_full_lat
        self.current_long = ego_long
        self.ego_off_road = ego_off_road
        self.road_data = semantic_db.get_road_details(self.current_road_id)

        ###################
        # getting relevant information about objects (using semantic DB)
        ###################
        self.static_objects = []
        for obj in state.perception_state.static_objects:
            obj_state = obj.getState()
            object_road_id, object_lane, object_full_lat, object_long, object_on_road = semantic_db.get_point_in_road_coordinates(
                X=obj_state.x, Y=obj_state.y, Z=0.0)
            if object_on_road:
                lon_distance_relative_to_ego, found_connection = semantic_db.get_point_relative_longitude(
                    to_road_id=object_road_id, to_lon_in_road=object_long, from_road_id=self.current_road_id,
                    from_lon_in_road=self.current_long, max_lookahead_distance=Constants.MAX_LOOKAHEAD_DISTANCE)
                if found_connection:    # ignoring everything not in our path looking forward
                    # TODO: get actual length and width, relative to objects's orientation
                    OBJECT_CONST_WIDTH_IN_METERS = 1.2
                    OBJECT_CONST_LENGTH_IN_METERS = 1.7
                    self.static_objects.append(
                        {'object_id': obj.unique_id, 'road_id': object_road_id, 'lane': object_lane,
                         'full_lat': object_full_lat, 'relative_lon': lon_distance_relative_to_ego,
                         'width': OBJECT_CONST_WIDTH_IN_METERS,
                         'length': OBJECT_CONST_LENGTH_IN_METERS})


