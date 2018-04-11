from decision_making.src.global_constants import BEHAVIORAL_PLANNING_HORIZON, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, LATERAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.state.state import ObjectSize


class SemanticActionsUtils:
    @staticmethod
    def compute_distance_by_velocity_diff(current_velocity: float,
                                          desired_velocity: float = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED,
                                          time: float = BEHAVIORAL_PLANNING_HORIZON) -> float:
        """
        Given a time-window and difference in velocities (current and desired), return the distance travelled by the
        mean velocity (assuming linear velocity profile)
        :param current_velocity: [m/sec]
        :param desired_velocity: [m/sec]
        :param time: [sec]
        :return: distance to travel [m]
        """
        return time * (current_velocity + desired_velocity) / 2

    @staticmethod
    def get_ego_lon_margin(ego_size: ObjectSize) -> float:
        """
        calculate margin for a safe longitudinal distance between ego and another car based on ego length
        :param ego_size: the size of ego (length, width, height)
        :return: safe longitudinal margin
        """
        return ego_size.length / 2 + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT

    @staticmethod
    def get_ego_lat_margin(ego_size: ObjectSize) -> float:
        """
        calculate margin for a safe lateral distance between ego and another car based on ego width
        :param ego_size: the size of ego (length, width, height)
        :return: safe lateral margin
        """
        return ego_size.width / 2 + LATERAL_SAFETY_MARGIN_FROM_OBJECT
