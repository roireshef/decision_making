import numpy as np

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS, LAT_ACC_LIMITS, LATERAL_SAFETY_MU
from decision_making.src.planning.types import LIMIT_MIN, FS_SV, FS_SX, FS_DX, FS_DV, FrenetTrajectories2D


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
        ego_traj_num = ego_ftraj.shape[0]
        times_num = ego_ftraj.shape[1]
        fstate_size = ego_ftraj.shape[2]
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

        # calculate longitudinal safety
        lon_safe_times = SafetyUtils.get_lon_safety(
            ego_ftraj_dup[..., FS_SX], ego_ftraj_dup[..., FS_SV], SAFETY_MARGIN_TIME_DELAY,
            obj_ftraj[..., FS_SX], obj_ftraj[..., FS_SV], SPECIFICATION_MARGIN_TIME_DELAY,
            0.5 * (ego_size[0] + obj_lengths))

        if not both_dimensions_flag:  # if only longitudinal safety
            return lon_safe_times

        # calculate lateral safety
        lat_safe_times = SafetyUtils.get_lat_safety(
            ego_ftraj_dup[..., FS_DX], ego_ftraj_dup[..., FS_DV], SAFETY_MARGIN_TIME_DELAY,
            obj_ftraj[..., FS_DX], obj_ftraj[..., FS_DV], SPECIFICATION_MARGIN_TIME_DELAY,
            0.5 * (ego_size[1] + obj_widths + LATERAL_SAFETY_MU))

        return np.logical_or(lon_safe_times, lat_safe_times)

    @staticmethod
    def get_lon_safety(x1: np.array, v1: np.array, td1: float, x2: np.array, v2: np.array, td2: float,
                       margins: np.array, max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate longitudinal safety between two objects for all timestamps
        :param x1: [m] object1 longitudes: tensor of shape: traj_num x objects_num x timestamps_num
        :param v1: [m/s] object1 velocities: tensor of shape: traj_num x objects_num x timestamps_num
        :param td1: [sec] object1 time delay
        :param x2: [m] object2 longitudes: tensor of any shape that compatible with the shape of object1
        :param v2: [m/s] object2 velocities: tensor of any shape that compatible with the shape of object1
        :param td2: [sec] object2 time delay
        :param margins: [m] objects' lengths: matrix of size objects_num x timestamps_num
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] longitudinal safety per timestamp. Tensor of the same shape as object1 or object2
        """
        dist = x1 - x2
        sign = np.sign(dist)
        switch = 0.5 * (sign + 1)
        one_over_a = 1. / (2 * max_brake)
        safe_dist = np.clip(sign * (v2 ** 2 - v1 ** 2) * one_over_a, 0, None) + \
                    (1 - switch) * v1 * td1 + switch * v2 * td2 + margins
        return sign * dist > safe_dist

    @staticmethod
    def get_lat_safety(x1: np.array, v1: np.array, td1: float, x2: np.array, v2: np.array, td2: float,
                       margins: np.array, max_brake: float=-LAT_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate lateral safety between two objects for all timestamps
        :param x1: [m] object1 longitudes: tensor of any shape
        :param v1: [m/s] object1 velocities: tensor of any shape
        :param td1: [sec] object1 time delay
        :param x2: [m] object2 longitudes: tensor of any shape that compatible with the shape of object1
        :param v2: [m/s] object2 velocities: tensor of any shape that compatible with the shape of object1
        :param td2: [sec] object2 time delay
        :param margins: [m] objects' widths + mu: matrix of size objects_num x timestamps_num
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] lateral safety per timestamp. Tensor of the same shape as object1 or object2
        """
        dist = x1 - x2
        sign = np.sign(dist)
        one_over_a = 1. / (2 * max_brake)
        safe_dist = np.clip(sign * (v2 * np.abs(v2) - v1 * np.abs(v2)) * one_over_a, 0, None) + \
                    np.clip(-sign * v1, 0, None) * td1 + np.clip(sign * v2, 0, None) * td2 + margins
        return sign * dist > safe_dist
