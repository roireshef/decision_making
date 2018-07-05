import numpy as np

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS, LAT_ACC_LIMITS, LATERAL_SAFETY_MU, LAT_VEL_BLAME_THRESH, LON_SAFETY_ACCEL_DURING_DELAY, \
    LAT_SAFETY_ACCEL_DURING_DELAY
from decision_making.src.planning.types import LIMIT_MIN, FrenetTrajectories2D, FS_SX

EGO_ACCEL_DIST = 0.5 * LON_SAFETY_ACCEL_DURING_DELAY * SAFETY_MARGIN_TIME_DELAY * SAFETY_MARGIN_TIME_DELAY
OBJ_ACCEL_DIST = 0.5 * LON_SAFETY_ACCEL_DURING_DELAY * SPECIFICATION_MARGIN_TIME_DELAY * SPECIFICATION_MARGIN_TIME_DELAY


class SafetyUtils:
    @staticmethod
    def calc_blame_times(ego_ftraj: FrenetTrajectories2D, ego_size: np.array,
                         obj_ftraj: np.array, obj_sizes: np.array,
                         both_dimensions_flag: bool=True) -> np.array:
        """
        Calculate safety boolean tensor for different ego Frenet trajectories and objects' Frenet trajectories.
        :param ego_ftraj: ego Frenet trajectories: tensor of shape: traj_num x timestamps_num x Frenet state size
        :param ego_size: array of size 2: ego length, ego width
        :param obj_ftraj: single ftrajectory (2D array) or array of ftrajectories of objects (3D array):
                shape: objects_num x timestamps_num x 6 (Frenet state size)
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
        else:  # a single object, don't duplicate ego_ftraj and obj_sizes
            ego_ftraj_dup = ego_ftraj

        # split the trajectories to 6 fstate components for the duplicated ego and the object
        ego = np.array(np.split(ego_ftraj_dup, 6, axis=-1))[..., 0]
        obj = np.array(np.split(obj_ftraj, 6, axis=-1))[..., 0]
        lon_margins = 0.5 * (ego_size[0] + obj_sizes[..., 0])[..., np.newaxis]
        lat_margins = 0.5 * (ego_size[1] + obj_sizes[..., 1])[..., np.newaxis] + LATERAL_SAFETY_MU

        # calculate longitudinal safety
        lon_safe_times = SafetyUtils.get_lon_safety(ego, SAFETY_MARGIN_TIME_DELAY, obj, SPECIFICATION_MARGIN_TIME_DELAY,
                                                    lon_margins)

        if not both_dimensions_flag:  # if only longitudinal safety
            return lon_safe_times

        # calculate lateral safety
        lat_safe_times, lat_vel_blame = SafetyUtils.get_lat_safety(ego, SAFETY_MARGIN_TIME_DELAY, obj,
                                                                   SPECIFICATION_MARGIN_TIME_DELAY, lat_margins)
        # calculate and return blame times
        return SafetyUtils._get_blame_times(ego[FS_SX], obj[FS_SX], lon_safe_times, lat_safe_times, lat_vel_blame)

    @staticmethod
    def get_lon_safety(ego: np.array, ego_time_delay: float, obj: np.array, obj_time_delay: float,
                       margins: np.array, max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate longitudinal safety between ego and another object for all timestamps.
        :param ego: ego fstate components: tensor of shape: 6 x traj_num x objects_num x timestamps_num
        :param ego_time_delay: [sec] ego time delay
        :param obj: object's fstate components: tensor of any shape that compatible with the shape of ego
        :param obj_time_delay: [sec] object's time delay
        :param margins: [m] lengths half sum: matrix of size objects_num x 1
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] longitudinal safety per timestamp. Tensor of the same shape as object1 or object2
        """
        (ego_lon, ego_vel, _, _, _, _) = ego
        (obj_lon, obj_vel, _, _, _, _) = obj

        dist = ego_lon - obj_lon
        sign = np.sign(dist)
        ego_ahead = 0.5 * (sign + 1)
        delayed_ego_vel = ego_vel + (1-ego_ahead) * ego_time_delay * LON_SAFETY_ACCEL_DURING_DELAY
        delayed_obj_vel = obj_vel + ego_ahead * obj_time_delay * LON_SAFETY_ACCEL_DURING_DELAY

        safe_dist = np.maximum(np.divide(sign * (delayed_obj_vel ** 2 - delayed_ego_vel ** 2), 2 * max_brake), 0) + \
                    (1 - ego_ahead) * (ego_vel * ego_time_delay + EGO_ACCEL_DIST) + \
                    ego_ahead * (obj_vel * obj_time_delay + OBJ_ACCEL_DIST) + margins
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
        :param margins: [m] objects' widths + mu: matrix of size objects_num x 1
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

        delay_dist = sign * (avg_obj_vel * obj_time_delay - avg_ego_vel * ego_time_delay)

        safe_dist = np.maximum(np.divide(sign * (delayed_obj_vel * np.abs(delayed_obj_vel) -
                                                 delayed_ego_vel * np.abs(delayed_ego_vel)),
                                         2 * max_brake) + delay_dist,
                               0) + \
                    margins

        return sign * dist > safe_dist, -sign * ego_lat_vel > np.clip(sign * obj_lat_vel, 0, LAT_VEL_BLAME_THRESH)

    @staticmethod
    def _get_blame_times(ego_lon: np.array, obj_lon: np.array, lon_safe_times: np.array, lat_safe_times: np.array,
                         lat_vel_blame: np.array) -> np.array:
        """
        Calculate all times, for which ego becomes unsafe on its blame.
        :param ego_lon: ego longitudes: tensor of shape: 3D traj_num x objects_num x timestamps_num or
                        2D traj_num x timestamps_num
        :param obj_lon: object longitudes: tensor of shape: 2D objects_num x timestamps_num or just 1D timestamps_num
        :param lon_safe_times: longitudinally safe times; tensor of shape like ego_lon
        :param lat_safe_times: laterally safe times; tensor of shape like ego_lon
        :param lat_vel_blame: times for which ego lat_vel towards object is larger than object's lat_vel
        :return: 3D or 2D boolean tensor (shape as ego_lon) of times, when ego is accountable for unsafe situation.
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
