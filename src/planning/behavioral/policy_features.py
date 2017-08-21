import numpy as np
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.default_policy import DefaultPolicyConfig
from decision_making.src.planning.behavioral.policy import PolicyConfig

'''
Static methods for computing complex features, e.g., ACDA speed.
'''


def compute_acda_speed(behavioral_state: BehavioralState):
    pass


def get_preferred_lane(behavioral_state: BehavioralState):
    """
    Navigation/traffic laws based function to determine the optimal lane for our current route. For example, default
     should be rightmost lane, but when nearing a left turn, should return the left lane.
    :param behavioral_state:
    :return: Integer representing the lane index 0 is right most lane.
    """
    pass


def get_closest_object_on_lane(policy_config: DefaultPolicyConfig, behavioral_state: BehavioralState, lat_options: np.array):
    """
    Gets the closest object on lane per each lateral offset in lat_options
    :param policy_config: policy parameters configuration
    :param behavioral_state:
    :param lat_options: grid of lateral offsets on road to be considered [m]
    :return: array with closest object per each lateral location options
    """

    # Fetch each latitude offset attributes (free-space, blocking objects, etc.)
    num_of_lat_options = len(lat_options)
    closest_blocking_object_in_lane = np.inf * np.ones(shape=[num_of_lat_options, 1])

    # Assign object to optional lanes
    # TODO: set actual dilation
    CAR_WIDTH_DILATION_IN_LANES = 0.4
    CAR_LENGTH_DILATION_IN_METERS = 2
    for blocking_object in behavioral_state.dynamic_objects:
        relative_lon = blocking_object.rel_road_localization.rel_lon
        if relative_lon < policy_config.assume_blocking_object_at_rear_if_distance_less_than:
            # If we passed an obstacle, treat it as at inf
            relative_lon = np.inf
        object_latitude_in_lanes = behavioral_state.map.convert_lat_in_meters_to_lat_in_lanes(
            road_id=blocking_object.road_localization.road_id,
            lat_in_meters=blocking_object.road_localization.full_lat)
        object_width_in_lanes = behavioral_state.map.convert_lat_in_meters_to_lat_in_lanes(
            road_id=blocking_object.road_localization.road_id, lat_in_meters=blocking_object.size.width)
        leftmost_edge_in_lanes = object_latitude_in_lanes + 0.5 * object_width_in_lanes
        leftmost_edge_in_lanes_dilated = leftmost_edge_in_lanes + 0.5 * CAR_WIDTH_DILATION_IN_LANES

        rightmost_edge_lanes = object_latitude_in_lanes - 0.5 * object_width_in_lanes
        rightmost_edge_lanes_dilated = rightmost_edge_lanes - 0.5 * CAR_WIDTH_DILATION_IN_LANES

        affected_lanes = np.where((lat_options < leftmost_edge_in_lanes_dilated) & (
            lat_options > rightmost_edge_lanes_dilated))[0]
        closest_blocking_object_in_lane[affected_lanes] = np.minimum(
            closest_blocking_object_in_lane[affected_lanes],
            relative_lon)

    return closest_blocking_object_in_lane
