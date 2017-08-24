import numpy as np

from decision_making.src.planning.behavioral.constants import LATERAL_MARGIN_FROM_OBJECT_TO_ASSUME_OUT_OF_WAY
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.default_policy import DefaultPolicyConfig

'''
Static methods for computing complex features, e.g., ACDA speed.
'''


class DefaultPolicyFeatures:
    @staticmethod
    def compute_acda_speed(behavioral_state: BehavioralState):
        pass

    @staticmethod
    def get_preferred_lane(behavioral_state: BehavioralState):
        """
        Navigation/traffic laws based function to determine the optimal lane for our current route. For example, default
         should be rightmost lane, but when nearing a left turn, should return the left lane.
        :param behavioral_state:
        :return: Integer representing the lane index 0 is right most lane.
        """
        pass

    @staticmethod
    def get_closest_object_on_path(policy_config: DefaultPolicyConfig, behavioral_state: BehavioralState,
                                   lat_options: np.array) -> np.array:
        """
        Gets the closest object on lane per each lateral offset in lat_options
        :param policy_config: policy parameters configuration
        :param behavioral_state:
        :param lat_options: grid of lateral offsets on road [m]
        :return: array with closest object per each lateral location options
        """

        # Fetch each latitude offset attributes (free-space, blocking objects, etc.)
        num_of_lat_options = len(lat_options)
        closest_blocking_object_per_option = np.inf * np.ones(shape=[num_of_lat_options, 1])

        # Assign object to optional lanes
        for blocking_object in behavioral_state.dynamic_objects:
            relative_lon = blocking_object.rel_road_localization.rel_lon
            if relative_lon < policy_config.assume_blocking_object_at_rear_if_distance_less_than:
                # If we passed an obstacle, treat it as at inf
                relative_lon = np.inf

            # get leftmost and rightmost edge of object
            object_leftmost_edge = blocking_object.road_localization.full_lat + 0.5 * blocking_object.size.width
            object_leftmost_edge_dilated = object_leftmost_edge + (
                LATERAL_MARGIN_FROM_OBJECT_TO_ASSUME_OUT_OF_WAY + 0.5 * behavioral_state.ego_state.size.width)

            object_rightmost_edge = blocking_object.road_localization.full_lat - 0.5 * blocking_object.size.width
            object_rightmost_edge_dilated = object_rightmost_edge - (
                LATERAL_MARGIN_FROM_OBJECT_TO_ASSUME_OUT_OF_WAY + 0.5 * behavioral_state.ego_state.size.width)

            # check which lateral offsets are affected
            affected_lanes = np.where((lat_options < object_leftmost_edge_dilated) & (
                lat_options > object_rightmost_edge_dilated))[0]

            # assign closest object to each lateral offset
            closest_blocking_object_per_option[affected_lanes] = np.minimum(
                closest_blocking_object_per_option[affected_lanes],
                relative_lon)

        return closest_blocking_object_per_option
