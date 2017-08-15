from decision_making.src.state.enriched_state import State, EgoState


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

    def update_behavioral_state(self, state: State, navigation_plan) -> None:
        """
        updating the behavioral state from the raw input state. This includes only direct processing without complex
        logic. This is implemented separately from initialization in order to potentially use differences for more
        efficient processing.
        :param state: the enriched state coming as a message from perception via DDS.
        :param navigation_plan: will be used for processing the behavioral state, as well as for PolicyFeatures
        :return: void
        """
        pass
