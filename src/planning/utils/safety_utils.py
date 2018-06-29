import numpy as np

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS, LAT_ACC_LIMITS, LATERAL_SAFETY_MU, TINY_LAT_VELOCITY, VELOCITY_LIMITS
from decision_making.src.planning.types import LIMIT_MIN, FS_SV, FS_SX, FS_DX, FS_DV, FrenetTrajectories2D, \
    FrenetState2D


class SafetyUtils:
    @staticmethod
    def calc_safety_for_trajectories(ego_ftraj: FrenetTrajectories2D, ego_size: np.array,
                                     obj_ftraj: np.array, obj_sizes: np.array,
                                     both_dimensions_flag: bool=True) -> np.array:
        """
        Calculate safety boolean tensor for different ego Frenet trajectories and objects' Frenet trajectories.
        :param ego_ftraj: ego Frenet trajectories: tensor of shape: traj_num x timestamps_num x Frenet state size
        :param ego_size: array of size 2: ego length, ego width
        :param obj_ftraj: one or array of objects Frenet trajectories: tensor of shape: objects_num x timestamps_num x Frenet state size
        :param obj_sizes: one or array of arrays of size 2: i-th row is i-th object's size
        :param both_dimensions_flag: if False then only longitudinal dimension is considered
        :return: [bool] safety per [ego trajectory, object, timestamp]. Tensor of shape: traj_num x objects_num x timestamps_num
        """
        (ego_traj_num, times_num, fstate_size) = ego_ftraj.shape
        if obj_ftraj.ndim > 2:  # multiple objects
            objects_num = obj_ftraj.shape[0]
            # duplicate ego_ftraj to the following dimensions: ego_traj_num, objects_num, timestamps_num, fstate (6)
            ego_ftraj_dup = np.tile(ego_ftraj, objects_num).reshape(ego_traj_num, times_num, objects_num,
                                                                    fstate_size).swapaxes(1, 2)
            obj_lengths = np.repeat(obj_sizes[:, 0], times_num).reshape(objects_num, times_num)
            obj_widths = np.repeat(obj_sizes[:, 1], times_num).reshape(objects_num, times_num)
        else:  # a single object, don't duplicate ego_ftraj and obj_sizes
            ego_ftraj_dup = ego_ftraj
            obj_lengths = obj_sizes[0]
            obj_widths = obj_sizes[1]

        # split the trajectories to 6 fstate components for the duplicated ego and the object
        ego = np.array(np.split(ego_ftraj_dup, 6, axis=-1))[..., 0]
        obj = np.array(np.split(obj_ftraj, 6, axis=-1))[..., 0]

        # calculate longitudinal safety
        lon_safe_times = SafetyUtils.get_lon_safety_full(ego, SAFETY_MARGIN_TIME_DELAY,
                                                         obj, SPECIFICATION_MARGIN_TIME_DELAY,
                                                         0.5 * (ego_size[0] + obj_lengths))

        if not both_dimensions_flag:  # if only longitudinal safety
            return lon_safe_times

        # calculate lateral safety
        lat_safe_times = SafetyUtils.get_lat_safety(ego, SAFETY_MARGIN_TIME_DELAY,
                                                    obj, SPECIFICATION_MARGIN_TIME_DELAY,
                                                    0.5 * (ego_size[1] + obj_widths) + LATERAL_SAFETY_MU)

        return np.logical_or(lon_safe_times, lat_safe_times)

    @staticmethod
    def get_lon_safety_full(ego: np.array, ego_time_delay: float, obj: np.array, obj_time_delay: float,
                            margins: np.array, max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        # TODO: consider to remove in the future this function, which has the following logic:
        # if ego does not move laterally toward the REAR object, it's safe w.r.t. to it
        """
        Calculate longitudinal safety between ego and another object for all timestamps.
        This function considers also lateral movement of ego w.r.t. to the REAR object.
        :param ego: ego fstate components: tensor of shape: 6 x traj_num x objects_num x timestamps_num
        :param ego_time_delay: [sec] ego time delay
        :param obj: object's fstate components: tensor of any shape that compatible with the shape of ego
        :param obj_time_delay: [sec] object's time delay
        :param margins: [m] lengths half sum: matrix of size objects_num x timestamps_num
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] longitudinal safety per timestamp. Tensor of the same shape as object1 or object2
        """
        (ego_lon, ego_vel, _, ego_lat, ego_lat_vel, _) = ego
        (obj_lon, obj_vel, _, obj_lat, _, _) = obj

        lon_safe_times = SafetyUtils.get_lon_safety(ego, ego_time_delay, obj, obj_time_delay, margins, max_brake)
        # if ego does not move laterally to the REAR object, it's always safe
        lon_sign = np.sign(ego_lon - obj_lon - margins)
        lat_sign = np.sign(ego_lat - obj_lat)
        lat_vel_from_obj = ego_lat_vel * lat_sign
        ego_is_ahead_and_zero_lat_vel = np.logical_and(-TINY_LAT_VELOCITY < lat_vel_from_obj,
                                                       lat_vel_from_obj < VELOCITY_LIMITS[1] * lon_sign)

        return np.logical_or(lon_safe_times, ego_is_ahead_and_zero_lat_vel)

    @staticmethod
    def get_lon_safety(ego: np.array, ego_time_delay: float, obj: np.array, obj_time_delay: float,
                       margins: np.array, max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate longitudinal safety between ego and another object for all timestamps.
        :param ego: ego fstate components: tensor of shape: 6 x traj_num x objects_num x timestamps_num
        :param ego_time_delay: [sec] ego time delay
        :param obj: object's fstate components: tensor of any shape that compatible with the shape of ego
        :param obj_time_delay: [sec] object's time delay
        :param margins: [m] lengths half sum: matrix of size objects_num x timestamps_num
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] longitudinal safety per timestamp. Tensor of the same shape as object1 or object2
        """
        (ego_lon, ego_vel, _, _, _, _) = ego
        (obj_lon, obj_vel, _, _, _, _) = obj

        dist = ego_lon - obj_lon
        sign = np.sign(dist)
        ego_ahead = 0.5 * (sign + 1)
        safe_dist = np.clip(np.divide(sign * (obj_vel ** 2 - ego_vel ** 2), 2 * max_brake), 0, None) + \
                    (1 - ego_ahead) * ego_vel * ego_time_delay + ego_ahead * obj_vel * obj_time_delay + margins
        return sign * dist > safe_dist

    @staticmethod
    def get_lat_safety(ego: np.array, ego_time_delay: float, obj: np.array, obj_time_delay: float,
                       margins: np.array, max_brake: float=-LAT_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate lateral safety between ego and another object for all timestamps.
        :param ego: ego fstate components: tensor of shape: 6 x traj_num x objects_num x timestamps_num
        :param ego_time_delay: [sec] object1 time delay
        :param obj: object's fstate components: tensor of any shape that compatible with the shape of ego
        :param obj_time_delay: [sec] object2 time delay
        :param margins: [m] objects' widths + mu: matrix of size objects_num x timestamps_num
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] lateral safety per timestamp. Tensor of the same shape as object1 or object2
        """
        (_, _, _, ego_lat, ego_lat_vel, _) = ego
        (_, _, _, obj_lat, obj_lat_vel, _) = obj

        dist = ego_lat - obj_lat
        sign = np.sign(dist)
        safe_dist = np.clip(np.divide(sign * (obj_lat_vel * np.abs(obj_lat_vel) - ego_lat_vel * np.abs(ego_lat_vel)),
                                      2 * max_brake),
                            0, None) + \
                    np.clip(-sign * ego_lat_vel, 0, None) * ego_time_delay + \
                    np.clip( sign * obj_lat_vel, 0, None) * obj_time_delay + \
                    margins
        return sign * dist > safe_dist
