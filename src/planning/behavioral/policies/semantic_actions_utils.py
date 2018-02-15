from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_HORIZON


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
