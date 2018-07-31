from typing import List

import numpy as np

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS, LAT_ACC_LIMITS, LATERAL_SAFETY_MU, LAT_VEL_BLAME_THRESH, LON_SAFETY_ACCEL_DURING_DELAY, \
    LAT_SAFETY_ACCEL_DURING_DELAY
from decision_making.src.planning.types import LIMIT_MIN, FrenetTrajectories2D, FS_SX, FS_SV, FS_DX, FS_DV, \
    ObjectSizeArray, OBJ_LENGTH, OBJ_WIDTH
from decision_making.src.state.state import ObjectSize


class SafetyUtils:
    """
    This class is an implementation of RSS safety from the paper of Mobileye https://arxiv.org/abs/1708.06374
    RSS safety promises avoidance of accident on ego's blame.

    Our changes with respect to the original RSS:

    Definitions
        Blame time: the first time, for which two objects become unsafe (both laterally and longitudinally).
        Lateral blame: if before the blame time the objects were safe laterally.
        Longitudinal blame: if before the blame time the objects were safe longitudinally.

    Lateral blame
        Mobileye:
            In the last version of Mobileye the blame is on that vehicle, which does not brake laterally after
            time_delay after blame time, with deceleration max_lateral_brake. They don't consider lateral velocities
            at blame time.
        Our version:
            Ego is blamed if:
                ego_lat_vel towards the object at blame time is faster than obj_lat_vel towards ego  OR
                ego_lat_vel > thresh (0.2 m/s)

    RSS formulas
    Lateral
        In the RSS lateral formula Mobileye consider only the case, when the objects move laterally one towards another,
        while I consider all cases.
        In our version, if the objects move laterally to the same direction, then the formula becomes quite similar to
        the longitudinal case, but it is bi-directional.

    Longitudinal
        In their RSS formula max_brake of two objects may be different, while in my implementation they are same.

    Mobileye's evasive efforts
        If a rear objects is not safe (its longitudinal blame), ego should brake laterally with some deceleration
        (a constant) and keep lat_vel = 0.
        If another object unsafely cuts-in ego (its lateral blame), ego should brake longitudinally with some given
        deceleration (another constant).
        We did not implement it, since we don't deal with policies in safety module.
    """
    # TODO: treat cases of occlusions (Mobileye / ACDA), road intersection/merge/split, stohastic predictions.

    @staticmethod
    def get_safe_times(ego_trajectories: FrenetTrajectories2D, ego_size: ObjectSize,
                       obj_trajectories: np.array, obj_sizes: List[ObjectSize]) -> \
            np.array:
        """
        For every ego Frenet trajectory, every predicted object's Frenet trajectory and every timestamp (3D tensor)
        calculate RSS safety: a boolean indicating whether this tensor element is safe.
        An exception from the pure RSS: If
            the unsafe object is rear wrt ego AND
            the unsafe situation was caused by the rear object AND
            the rear object's predicted trajectory does not create an accident with ego trajectory
        then the function returns True (safe).
        :param ego_trajectories: ego Frenet trajectories: 3D tensor of shape: traj_num x timestamps_num x Frenet state size
        :param ego_size: ego size
        :param obj_trajectories: 3D array of ftrajectories of objects: shape: objects_num x timestamps_num x 6 (Frenet state size)
        :param obj_sizes: list of objects' sizes
        :return: [bool] safety per [ego trajectory, object, timestamp].
                Tensor of shape: traj_num x objects_num x timestamps_num
        """
        objects_num = obj_trajectories.shape[0]
        # calculate safe times for every object
        safe_times = np.ones((ego_trajectories.shape[0], objects_num, ego_trajectories.shape[1])).astype(bool)
        ego_size_arr = np.array([ego_size.length, ego_size.width])
        for i in range(objects_num):  # loop over objects
            obj_size_arr = np.array([obj_sizes[i].length, obj_sizes[i].width])
            safe_times[:, i] = SafetyUtils._get_safe_times_per_obj(ego_trajectories, ego_size_arr,
                                                                   obj_trajectories[i], obj_size_arr)
        return safe_times

    @staticmethod
    def _get_safe_times_per_obj(ego_trajectories: FrenetTrajectories2D, ego_size: ObjectSizeArray,
                                obj_trajectories: FrenetTrajectories2D, obj_size: ObjectSizeArray) -> np.array:
        """
        Calculate safety boolean tensor for different ego Frenet trajectories and a SINGLE object.
        Safety definition by the RSS paper: an object is defined safe if it's safe either longitudinally OR laterally.
        Do it by calculation of
            (1) longitudinal safety,
            (2) lateral safety,
            (3) blame determination using the combination of the above results.
        :param ego_trajectories: ego frenet trajectories: 3D tensor of shape: trajectories_num x timestamps_num x 6
        :param ego_size: array of size 2: ego length, ego width
        :param obj_trajectories: object's frenet trajectories: 2D matrix of shape timestamps_num x 6
        :param obj_size: one or array of arrays of size 2: i-th row is i-th object's size
        :return: [bool] safe times. 2D matrix of shape: trajectories_num x timestamps_num
        """
        lon_margin = 0.5 * (ego_size[OBJ_LENGTH] + obj_size[OBJ_LENGTH])
        lat_margin = 0.5 * (ego_size[OBJ_WIDTH] + obj_size[OBJ_WIDTH]) + LATERAL_SAFETY_MU

        # calculate longitudinal safety
        lon_safe_times = SafetyUtils._get_lon_safety(ego_trajectories, SAFETY_MARGIN_TIME_DELAY,
                                                     obj_trajectories, SPECIFICATION_MARGIN_TIME_DELAY, lon_margin)
        # calculate lateral safety
        lat_safe_times, lat_vel_blame = SafetyUtils._get_lat_safety(ego_trajectories, SAFETY_MARGIN_TIME_DELAY,
                                                                    obj_trajectories, SPECIFICATION_MARGIN_TIME_DELAY,
                                                                    lat_margin)
        # calculate and return blame times
        safe_times = SafetyUtils._calc_fully_safe_times(ego_trajectories[..., FS_SX], obj_trajectories[..., FS_SX],
                                                        lon_safe_times, lat_safe_times, lat_vel_blame, lon_margin)
        return safe_times

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
        ego_vel_after_reaction_time = ego_vel + (1-ego_ahead) * ego_response_time * LON_SAFETY_ACCEL_DURING_DELAY
        obj_vel_after_reaction_time = obj_vel + ego_ahead * obj_response_time * LON_SAFETY_ACCEL_DURING_DELAY

        # longitudinal RSS formula considers distance reduction during the reaction time and difference between
        # objects' braking distances
        ego_acceleration_dist = 0.5 * LON_SAFETY_ACCEL_DURING_DELAY * SAFETY_MARGIN_TIME_DELAY * SAFETY_MARGIN_TIME_DELAY
        obj_acceleration_dist = 0.5 * LON_SAFETY_ACCEL_DURING_DELAY * SPECIFICATION_MARGIN_TIME_DELAY * SPECIFICATION_MARGIN_TIME_DELAY
        safe_dist = np.maximum(np.divide(sign_of_lon_relative_to_obj * (obj_vel_after_reaction_time ** 2 -
                                                                        ego_vel_after_reaction_time ** 2),
                                         2 * max_brake), 0) + \
                    (1 - ego_ahead) * (ego_vel * ego_response_time + ego_acceleration_dist) + \
                    ego_ahead * (obj_vel * obj_response_time + obj_acceleration_dist) + margin

        return sign_of_lon_relative_to_obj * lon_relative_to_obj > safe_dist

    @staticmethod
    def _get_lat_safety(ego_trajectories: FrenetTrajectories2D, ego_response_time: float,
                        obj_trajectories: FrenetTrajectories2D, obj_response_time: float,
                        margin: float, max_brake: float=-LAT_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate lateral safety between ego and another object for all timestamps.
        Lateral safety between two objects considers only their lateral data: latitude and lateral velocity.
        Lateral RSS formula considers lateral distance reduction during the reaction time and difference between
        objects' lateral braking distances.
        An object is defined safe if it's safe either longitudinally OR laterally.
        :param ego_trajectories: ego frenet trajectories: 3D tensor of shape: traj_num x timestamps_num x 6
        :param ego_response_time: [sec] object1 response time
        :param obj_trajectories: object's frenet trajectories: 2D matrix: timestamps_num x 6
        :param obj_response_time: [sec] object2 response time
        :param margin: [m] half sum of objects' widths + mu (mu is from Mobileye's paper)
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] 1. lateral safety per timestamp. 2D matrix shape: traj_num x timestamps_num
                        2. lateral velocity blame: True if ego moves laterally towards object. The same shape.
        """
        # extract the relevant lateral data from the trajectories
        ego_lat, ego_lat_vel = ego_trajectories[..., FS_DX], ego_trajectories[..., FS_DV]
        obj_lat, obj_lat_vel = obj_trajectories[..., FS_DX], obj_trajectories[..., FS_DV]

        # determine which object is on the left side (per trajectory and timestamp)
        lat_relative_to_obj = ego_lat - obj_lat
        sign_of_lat_relative_to_obj = np.sign(lat_relative_to_obj)

        # The worst-case lateral velocity of any object (either ego or another object) may increase during its
        # reaction time, since it may accelerate laterally toward the second object before it starts to brake laterally.
        ego_vel_after_reaction_time = ego_lat_vel - sign_of_lat_relative_to_obj * ego_response_time * LAT_SAFETY_ACCEL_DURING_DELAY
        obj_vel_after_reaction_time = obj_lat_vel + sign_of_lat_relative_to_obj * obj_response_time * LAT_SAFETY_ACCEL_DURING_DELAY

        # the distance objects move one towards another during their reaction time
        avg_ego_vel = 0.5 * (ego_lat_vel + ego_vel_after_reaction_time)
        avg_obj_vel = 0.5 * (obj_lat_vel + obj_vel_after_reaction_time)
        reaction_dist = sign_of_lat_relative_to_obj * (avg_obj_vel * obj_response_time - avg_ego_vel * ego_response_time)

        # lateral RSS formula considers lateral distance reduction during the reaction time and difference between
        # objects' lateral braking distances
        safe_dist = np.maximum(np.divide(sign_of_lat_relative_to_obj *
                                         (obj_vel_after_reaction_time * np.abs(obj_vel_after_reaction_time) -
                                          ego_vel_after_reaction_time * np.abs(ego_vel_after_reaction_time)),
                                         2 * max_brake) + reaction_dist, 0) + margin

        # lateral velocity blame is true if one of the 3 following conditions holds:
        #   ego moves laterally towards object faster than small thresh or
        #   ego moves laterally towards object faster than the object moves laterally towards ego or
        #   the action is towards the object
        action_lat_dir = (ego_lat[:, -1] - ego_lat[:, 0])[:, np.newaxis]  # last lat - initial lat
        lat_vel_blame = np.logical_or(-sign_of_lat_relative_to_obj * ego_lat_vel >
                                      np.clip(sign_of_lat_relative_to_obj * obj_lat_vel, 0, LAT_VEL_BLAME_THRESH),
                                      -sign_of_lat_relative_to_obj * action_lat_dir > 0)

        return sign_of_lat_relative_to_obj * lat_relative_to_obj > safe_dist, lat_vel_blame

    @staticmethod
    def _calc_fully_safe_times(ego_longitudes: np.array, obj_longitudes: np.array,
                               lon_safe_times: np.array, lat_safe_times: np.array,
                               lat_vel_blame: np.array, lon_margin: float) -> np.array:
        """
        For all trajectories and times check if ego is safe wrt non-rear objects and not unsafe wrt rear on ego blame.
        A timestamp is defined as safe if
            1. ego is safe wrt non-rear object AND
            2. ego is not blamed for unsafe situation (wrt rear object)
        :param ego_longitudes: ego longitudes: 2D matrix of shape: trajectories_num x timestamps_num
        :param obj_longitudes: object longitudes: 1D array of size timestamps_num
        :param lon_safe_times: longitudinally safe times; boolean 2D matrix of shape: trajectories_num x timestamps_num
        :param lat_safe_times: laterally safe times; boolean 2D matrix of shape: trajectories_num x timestamps_num
        :param lat_vel_blame: times for which ego lat_vel towards object is larger than object's lat_vel;
                boolean 2D matrix of shape: trajectories_num x timestamps_num
        :param lon_margin: [m] cars' lengths half sum
        :return: 2D boolean matrix (2D matrix of shape: trajectories_num x timestamps_num) of times, when ego is safe.
        """
        # find points, for which longitudinal safety changes from true to false, while unsafe laterally,
        # and the object is not behind ego
        becomes_unsafe_longitudinally_excluding_time_0 = np.logical_and(lon_safe_times[:, :-1],
                                                                        np.logical_not(lon_safe_times[:, 1:]))  # become unsafe longitudinally
        becomes_unsafe_longitudinally = np.insert(becomes_unsafe_longitudinally_excluding_time_0, 0,
                                                  np.logical_not(lon_safe_times[:, 0]), axis=-1)  # add first column
        unsafe_times_with_lon_blame = np.logical_and(becomes_unsafe_longitudinally,
                                                     np.logical_not(lat_safe_times))  # becomes fully unsafe
        lon_ego_blame_times = np.logical_and(unsafe_times_with_lon_blame,
                                             (ego_longitudes < obj_longitudes))  # don't blame for unsafe rear object

        # find points, for which lateral safety changes from true to false, while the unsafe longitudinally,
        # and lateral velocity of ego towards the object is larger than of the object towards ego
        becomes_unsafe_laterally_excluding_time_0 = np.logical_and(lat_safe_times[:, :-1],
                                                                   np.logical_not(lat_safe_times[:, 1:]))  # become unsafe laterally
        # blame in time=0: laterally and longitudinally unsafe at time=0 AND ego is behind object
        ego_unsafe_at_time_0 = np.logical_and(np.logical_not(lat_safe_times[:, 0]), lon_ego_blame_times[:, 0])
        becomes_unsafe_laterally = np.insert(becomes_unsafe_laterally_excluding_time_0, 0,
                                             ego_unsafe_at_time_0, axis=-1)  # add first column
        unsafe_times_with_lat_blame = np.logical_and(becomes_unsafe_laterally,
                                                     np.logical_not(lon_safe_times))  # becomes fully unsafe
        lat_ego_blame_times = np.logical_and(unsafe_times_with_lat_blame,
                                             lat_vel_blame)  # blame according to lateral velocity

        # calculate final safety: a timestamp is defined as safe if
        #   1. ego is safe wrt non-rear object AND
        #   2. ego is not blamed for unsafe situation (wrt rear object), unless the rear object's trajectory
        #       intersects ego trajectory (considering lon_margin), i.e. an accident is expected.
        #       In case of accident ego is defined unsafe, despite the blame on the rear object.
        not_blame_on_ego = np.logical_not(np.logical_or(lon_ego_blame_times, lat_ego_blame_times))
        # safe_or_rear_obj: either ego is safe wrt the object or the object is rear and we don't care whether it's safe
        safe_obj_or_rear_obj = np.logical_or(ego_longitudes > obj_longitudes + lon_margin,
                                             np.logical_or(lon_safe_times, lat_safe_times))
        safe_times = np.logical_and(safe_obj_or_rear_obj, not_blame_on_ego)
        return safe_times
