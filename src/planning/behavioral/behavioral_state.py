from src.state.enriched_state import State, EgoState


class BehavioralState:
    # TODO add type hint for navigation plan, once implemented
    def __init__(self, ego_state: EgoState = None, margin_info: MarginInfo = None,
                 lane_object_information: list[LaneObjectInfo] = [], navigation_plan=None):
        self._ego_state = ego_state  # taken from the enriched state
        self._margin_info = margin_info
        self._lane_object_information = lane_object_information  # Array of LaneObjectInfo's
        self._navigation_plan = navigation_plan

    def update_behavioral_state(self, state: State, navigation_plan) -> None:
        """
        updating the behavioral state from the raw input state. This includes only direct processing without complex
        logic. This is implemented separately from initialization in order to potentially use differences for more
        efficient processing.
        :param self:
        :param state: the enriched state coming as a message from perception via DDS.
        :param navigation_plan: will be used for processing the behavioral state, as well as for PolicyFeatures
        :return: void
        """
        pass


class MarginInfo:
    def __init__(self, right_width: float, right_clear: bool, left_width: float, left_clear: bool):
        self.right_width = right_width
        self.right_clear = right_clear
        self.left_width = left_width
        self.left_clear = left_clear


class LaneObjectInfo:
    def __init__(self, relative_velocity_of_closest_object: float, time_distance_of_closest_object: float,
                 confidence: float = None):
        self.relative_velocity_of_closest_object = relative_velocity_of_closest_object
        self.time_distance_of_closest_object = time_distance_of_closest_object
        self.confidence = confidence
