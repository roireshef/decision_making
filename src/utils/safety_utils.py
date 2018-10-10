import numpy as np
from decision_making.src.global_constants import LON_ACC_LIMITS, SAFETY_MARGIN_TIME_DELAY, \
    SPECIFICATION_MARGIN_TIME_DELAY, LON_SAFETY_ACCEL_DURING_RESPONSE
from decision_making.src.planning.types import FrenetTrajectories2D, FS_SX, FS_SV, LIMIT_MIN


class SafetyUtils:

    @staticmethod
    def _get_lon_safety(ego_trajectories: FrenetTrajectories2D, ego_response_time: float,
                        obj_trajectories: FrenetTrajectories2D, obj_response_time: float,
                        margin: float, max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate longitudinal safety between ego and another object for all timestamps.
        Longitudinal safety between two objects considers only their longitudinal data: longitude and longitudinal velocity.
        Longitudinal RSS formula considers distance reduction during the reaction time and difference between
        objects' braking distances.
        An object is defined safe if it's safe either longitudinally OR laterally.
        :param ego_trajectories: ego frenet trajectories: 3D tensor of shape: traj_num x timestamps_num x 6
        :param ego_response_time: [sec] ego response time
        :param obj_trajectories: object's frenet trajectories: 2D matrix: timestamps_num x 6
        :param obj_response_time: [sec] object's response time
        :param margin: [m] cars' lengths half sum
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] longitudinal safety per timestamp. 2D matrix shape: traj_num x timestamps_num
        """
        # extract the relevant longitudinal data from the trajectories
        ego_lon, ego_vel = ego_trajectories[..., FS_SX], ego_trajectories[..., FS_SV]
        obj_lon, obj_vel = obj_trajectories[..., FS_SX], obj_trajectories[..., FS_SV]

        # determine which object is in front (per trajectory and timestamp)
        lon_relative_to_obj = ego_lon - obj_lon
        sign_of_lon_relative_to_obj = np.sign(lon_relative_to_obj)
        ego_ahead = (sign_of_lon_relative_to_obj > 0).astype(int)

        # The worst-case velocity of the rear object (either ego or another object) may increase during its reaction
        # time, since it may accelerate before it starts to brake.
        ego_vel_after_reaction_time = ego_vel + (1-ego_ahead) * ego_response_time * LON_SAFETY_ACCEL_DURING_RESPONSE
        obj_vel_after_reaction_time = obj_vel + ego_ahead * obj_response_time * LON_SAFETY_ACCEL_DURING_RESPONSE

        # longitudinal RSS formula considers distance reduction during the reaction time and difference between
        # objects' braking distances
        ego_acceleration_dist = 0.5 * LON_SAFETY_ACCEL_DURING_RESPONSE * SAFETY_MARGIN_TIME_DELAY ** 2
        obj_acceleration_dist = 0.5 * LON_SAFETY_ACCEL_DURING_RESPONSE * SPECIFICATION_MARGIN_TIME_DELAY ** 2
        safe_dist = np.maximum(np.divide(sign_of_lon_relative_to_obj * (obj_vel_after_reaction_time ** 2 -
                                                                        ego_vel_after_reaction_time ** 2),
                                         2 * max_brake), 0) + \
                    (1 - ego_ahead) * (ego_vel * ego_response_time + ego_acceleration_dist) + \
                    ego_ahead * (obj_vel * obj_response_time + obj_acceleration_dist) + margin

        return sign_of_lon_relative_to_obj * lon_relative_to_obj > safe_dist
