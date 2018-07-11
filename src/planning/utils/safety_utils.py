import numpy as np
import copy

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS, LAT_ACC_LIMITS, LATERAL_SAFETY_MU, LAT_VEL_BLAME_THRESH, LON_SAFETY_ACCEL_DURING_DELAY, \
    LAT_SAFETY_ACCEL_DURING_DELAY
from decision_making.src.planning.types import LIMIT_MIN, FrenetTrajectories2D, FS_SX

EGO_ACCEL_DIST = 0.5 * LON_SAFETY_ACCEL_DURING_DELAY * SAFETY_MARGIN_TIME_DELAY * SAFETY_MARGIN_TIME_DELAY
OBJ_ACCEL_DIST = 0.5 * LON_SAFETY_ACCEL_DURING_DELAY * SPECIFICATION_MARGIN_TIME_DELAY * SPECIFICATION_MARGIN_TIME_DELAY


class SafetyUtils:
    @staticmethod
    def get_blame_times(ego_ftraj: FrenetTrajectories2D, ego_size: np.array,
                        obj_ftraj: np.array, obj_sizes: np.array) -> np.array:
        """
        Calculate safety boolean tensor for different ego Frenet trajectories and objects' Frenet trajectories.
        :param ego_ftraj: ego Frenet trajectories: 3D tensor of shape: traj_num x timestamps_num x Frenet state size
        :param ego_size: array of size 2: ego length, ego width
        :param obj_ftraj: 3D array of ftrajectories of objects: shape: objects_num x timestamps_num x 6 (Frenet state size)
        :param obj_sizes: 2D array of arrays of size 2: i-th row is i-th object's size
        :return: [bool] safety per [ego trajectory, object, timestamp]. Tensor of shape: traj_num x objects_num x timestamps_num
        """
        # split the trajectories to 6 fstate components for the duplicated ego and the object
        ego = np.transpose(ego_ftraj, (2, 0, 1))
        obj = np.transpose(obj_ftraj, (2, 0, 1))

        # calculate blame times for every object
        blame_times = np.ones((ego_ftraj.shape[0], obj_ftraj.shape[0], obj_ftraj.shape[1]))
        for i, _ in enumerate(obj_ftraj):  # loop over objects
            blame_times[:, i] = SafetyUtils._get_blame_times_per_obj(ego, ego_size, obj[:, i], obj_sizes[i])
        return blame_times

    @staticmethod
    def _get_blame_times_per_obj(ego: np.array, ego_size: np.array, obj: np.array, obj_size: np.array) -> np.array:
        """
        Calculate safety boolean tensor for different ego Frenet trajectories and a single object.
        :param ego: ego fstate components: 3D tensor of shape: 6 x traj_num x timestamps_num
        :param ego_size: array of size 2: ego length, ego width
        :param obj: object's fstate components: 2D matrix of shape 6 x timestamps_num
        :param obj_size: one or array of arrays of size 2: i-th row is i-th object's size
        :return: [bool] ego blame per [ego trajectory, timestamp]. 2D matrix of shape: traj_num x timestamps_num
        """
        lon_margins = 0.5 * (ego_size[0] + obj_size[0])
        lat_margins = 0.5 * (ego_size[1] + obj_size[1]) + LATERAL_SAFETY_MU

        # calculate longitudinal safety
        lon_safe_times = SafetyUtils._get_lon_safety(ego, SAFETY_MARGIN_TIME_DELAY, obj, SPECIFICATION_MARGIN_TIME_DELAY,
                                                     lon_margins)
        # calculate lateral safety
        lat_safe_times, lat_vel_blame = SafetyUtils._get_lat_safety(ego, SAFETY_MARGIN_TIME_DELAY, obj,
                                                                    SPECIFICATION_MARGIN_TIME_DELAY, lat_margins)

        # if the current state is unsafe wrt any object (except the rear), then all times are unsafe
        if not lon_safe_times[0, 0] and not lat_safe_times[0, 0] and ego[FS_SX, 0, 0] < obj[FS_SX, 0] + lon_margins:
            return np.ones((ego.shape[1], ego.shape[2]))

        # calculate and return blame times
        blame_times = SafetyUtils._calc_blame(ego[FS_SX], obj[FS_SX], lon_safe_times, lat_safe_times, lat_vel_blame)
        return blame_times

    @staticmethod
    def _get_lon_safety(ego: np.array, ego_time_delay: float, obj: np.array, obj_time_delay: float,
                        margins: float, max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate longitudinal safety between ego and another object for all timestamps.
        :param ego: ego fstate components: tensor of shape: 6 x traj_num x objects_num x timestamps_num
        :param ego_time_delay: [sec] ego time delay
        :param obj: object's fstate components: tensor of any shape that compatible with the shape of ego
        :param obj_time_delay: [sec] object's time delay
        :param margins: [m] cars' lengths half sum
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] longitudinal safety per timestamp. Tensor of the same shape as object1 or object2
        """
        (ego_lon, ego_vel, _, _, _, _) = ego
        (obj_lon, obj_vel, _, _, _, _) = obj

        dist = ego_lon - obj_lon
        sign = np.sign(dist)
        ego_ahead = (sign > 0).astype(int)
        delayed_ego_vel = ego_vel + (1-ego_ahead) * ego_time_delay * LON_SAFETY_ACCEL_DURING_DELAY
        delayed_obj_vel = obj_vel + ego_ahead * obj_time_delay * LON_SAFETY_ACCEL_DURING_DELAY

        safe_dist = np.maximum(np.divide(sign * (delayed_obj_vel ** 2 - delayed_ego_vel ** 2), 2 * max_brake), 0) + \
                    (1 - ego_ahead) * (ego_vel * ego_time_delay + EGO_ACCEL_DIST) + \
                    ego_ahead * (obj_vel * obj_time_delay + OBJ_ACCEL_DIST) + margins
        return sign * dist > safe_dist

    @staticmethod
    def _get_lat_safety(ego: np.array, ego_time_delay: float, obj: np.array, obj_time_delay: float,
                        margins: float, max_brake: float=-LAT_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate lateral safety between ego and another object for all timestamps.
        :param ego: ego fstate components: tensor of shape: 6 x traj_num x objects_num x timestamps_num
        :param ego_time_delay: [sec] object1 time delay
        :param obj: object's fstate components: tensor of any shape that compatible with the shape of ego
        :param obj_time_delay: [sec] object2 time delay
        :param margins: [m] half sum of objects' widths + mu (mu is from Mobileye's paper)
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] 1. lateral safety per timestamp. Shape: 3D or 2D traj_num x (objects_num x) timestamps_num
                        2. lateral velocity blame: True if ego moves laterally towards object. The same shape.
        """
        (_, _, _, ego_lat, ego_lat_vel, _) = ego
        (_, _, _, obj_lat, obj_lat_vel, _) = obj

        dist = ego_lat - obj_lat
        sign = np.sign(dist)

        delayed_ego_vel = ego_lat_vel - sign * ego_time_delay * LAT_SAFETY_ACCEL_DURING_DELAY
        delayed_obj_vel = obj_lat_vel + sign * obj_time_delay * LAT_SAFETY_ACCEL_DURING_DELAY
        avg_ego_vel = 0.5 * (ego_lat_vel + delayed_ego_vel)
        avg_obj_vel = 0.5 * (obj_lat_vel + delayed_obj_vel)

        reaction_dist = sign * (avg_obj_vel * obj_time_delay - avg_ego_vel * ego_time_delay)

        safe_dist = np.maximum(np.divide(sign * (delayed_obj_vel * np.abs(delayed_obj_vel) -
                                                 delayed_ego_vel * np.abs(delayed_ego_vel)),
                                         2 * max_brake) + reaction_dist,
                               0) + \
                    margins

        # lateral velocity blame is true if one of the 3 following conditions holds:
        #   ego moves laterally towards object faster than small thresh or
        #   ego moves laterally towards object faster than the object moves laterally towards ego or
        #   the action is towards the object
        action_lat_dir = (ego_lat[:, -1] - ego_lat[:, 0])[:, np.newaxis]  # last lat - initial lat
        lat_vel_blame = np.logical_or(-sign * ego_lat_vel > np.clip(sign * obj_lat_vel, 0, LAT_VEL_BLAME_THRESH),
                                      -sign * action_lat_dir > 0)
        return sign * dist > safe_dist, lat_vel_blame

    @staticmethod
    def _calc_blame(ego_lon: np.array, obj_lon: np.array, lon_safe_times: np.array, lat_safe_times: np.array,
                    lat_vel_blame: np.array) -> np.array:
        """
        Calculate all times, for which ego becomes unsafe on its blame.
        :param ego_lon: ego longitudes: 2D matrix of shape: traj_num x timestamps_num
        :param obj_lon: object longitudes: 1D array of size timestamps_num
        :param lon_safe_times: longitudinally safe times; shape as ego_lon
        :param lat_safe_times: laterally safe times; shape as ego_lon
        :param lat_vel_blame: times for which ego lat_vel towards object is larger than object's lat_vel; shape as ego_lon
        :return: 2D boolean matrix (shape as ego_lon) of times, when ego is accountable for unsafe situation.
        """
        # find points, for which longitudinal safety changes from true to false, while unsafe laterally,
        # and the object is not behind ego
        lon_blame = np.logical_and(lon_safe_times[:, ..., :-1], np.logical_not(lon_safe_times[:, ..., 1:]))  # become unsafe
        lon_blame = np.insert(lon_blame, 0, np.logical_not(lon_safe_times[:, ..., 0]), axis=-1)  # dup first column
        lon_blame = np.logical_and(lon_blame, np.logical_not(lat_safe_times))  # becomes fully unsafe
        lon_blame = np.logical_and(lon_blame, (ego_lon < obj_lon))  # don't blame for unsafe rear object

        # find points, for which lateral safety changes from true to false, while the unsafe longitudinally,
        # and lateral velocity of ego towards the object is larger than of the object towards ego
        lat_blame = np.logical_and(lat_safe_times[:, ..., :-1], np.logical_not(lat_safe_times[:, ..., 1:]))  # become unsafe
        # blame in time=0: laterally and longitudinally unsafe at time=0 AND ego is behind object
        init_blame = np.logical_and(np.logical_not(lat_safe_times[:, ..., 0]), lon_blame[:, ..., 0])
        lat_blame = np.insert(lat_blame, 0, init_blame, axis=-1)  # dup first column
        lat_blame = np.logical_and(lat_blame, np.logical_not(lon_safe_times))  # becomes fully unsafe
        lat_blame = np.logical_and(lat_blame, lat_vel_blame)  # blame according to lateral velocity
        return np.logical_or(lon_blame, lat_blame)
