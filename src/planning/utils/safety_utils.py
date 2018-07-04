import numpy as np
import time

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS, LAT_ACC_LIMITS, LATERAL_SAFETY_MU
from decision_making.src.planning.types import LIMIT_MIN, FrenetTrajectories2D, FrenetState2D, FS_DX, FS_DV, FS_DA
from decision_making.src.planning.utils.optimal_control.generate_traj import TrajectoriesGenerator


class SafetyUtils:

    @staticmethod
    def calc_safe_T_d(ego_fstate: FrenetState2D, ego_size: np.array, specs: np.array, specs_mask: np.array,
                      time_samples: np.array, predictions: np.array, obj_sizes: np.array) -> np.array:
        """
        Calculate maximal safe T_d for each given action spec.
        If an action is not safe, the appropriate T_d is 0.
        :param ego_fstate: initial ego frenet state
        :param ego_size: array of size 2
        :param specs: array of ActionSpec
        :param specs_mask: boolean array of size specs.shape[0]
        :param time_samples: array of time samples for the given predictions; samples_num = predictions.shape[1]
        :param predictions: 2D array of objects predictions; shape: objects_num x samples_num
        :param obj_sizes: 2D array of objects' sizes; shape: objects_num x 2
        :return: 1D array of T_d per spec.
        """
        samples_num = time_samples.shape[0]
        # collect non-filtered specs details: t,v,s,d, and mapping between valid specs and all specs.
        specs_arr = np.array([np.array([i, spec.t, spec.v, spec.s, spec.d])
                              for i, spec in enumerate(specs) if specs_mask[i]])
        (spec_orig_idxs, specs_t, specs_v, specs_s, specs_d) = specs_arr.transpose()
        spec_orig_idxs = spec_orig_idxs.astype(int)

        st_tot = time.time()
        st = time.time()
        valid_specs_num = specs_t.shape[0]  # after filtering

        # choose a grid of T_d between reasonable bounds
        grid_resolution = 2  # seconds between two adjacent T_d samples
        T_d = np.arange(7, 3 - np.finfo(np.float16).eps, -grid_resolution)
        T_d_num = T_d.shape[0]

        # calculate ego lateral trajectories (only location and velocity) for all specs and all T_d values
        ego_dx, ego_dv = TrajectoriesGenerator.calc_lateral_trajectories(ego_fstate, specs_d, T_d, time_samples)
        lat_time = time.time()-st
        st = time.time()

        # calculate ego longitudinal trajectories (only location and velocity) for all specs
        ego_sx, ego_sv = TrajectoriesGenerator.calc_longitudinal_trajectories(ego_fstate, specs_t, specs_v, specs_s,
                                                                              time_samples)
        lon_time = time.time()-st
        st = time.time()
        # duplicate longitudinal trajectories to be aligned with lateral trajectories
        dup_ego_sx = np.tile(ego_sx, T_d_num).reshape(specs_t.shape[0] * T_d_num, samples_num)
        dup_ego_sv = np.tile(ego_sv, T_d_num).reshape(specs_t.shape[0] * T_d_num, samples_num)
        zeros = np.zeros(dup_ego_sx.shape)
        # ego Frenet trajectories
        ego_ftraj = np.dstack((dup_ego_sx, dup_ego_sv, zeros, ego_dx, ego_dv, zeros))

        dup_time = time.time()-st
        st = time.time()

        # calculate RSS for all trajectories and all time samples
        # returns 3D array; safe times for all valid specs, T_d's, objects, time samples;
        # shape: valid_specs_num*T_d_num x objects_num x samples_num
        safe_times = SafetyUtils.calc_safety_for_trajectories(ego_ftraj, ego_size, predictions, obj_sizes)

        RSS_time = time.time()-st

        # 2D array; safe times for all actions and T_d's; shape: valid_specs x T_d_num
        safe_specs_T_d = safe_times.all(axis=(1, 2)).reshape(valid_specs_num, T_d_num)

        # 1D array; unsafe actions; shape: actions_num
        unsafe_specs = np.logical_not(safe_specs_T_d).all(axis=1)

        # 1D array; maximal safe T_d for each valid spec; shape: valid_specs
        max_safe_T_d = T_d[np.argmax(safe_specs_T_d, axis=1)]  # for each spec find max T_d
        max_safe_T_d[unsafe_specs] = 0
        max_safe_T_d = np.minimum(max_safe_T_d, specs_t)

        # 1D array; maximal safe T_d for each spec; shape: specs.shape[0]
        safe_T_d = np.zeros(len(specs))
        safe_T_d[spec_orig_idxs[np.arange(0, valid_specs_num)]] = max_safe_T_d

        time_tot = time.time() - st_tot

        print('calc_safe_T_d time=%f: lat=%f lon=%f dup=%f RSS=%f' % (time_tot, lat_time, lon_time, dup_time, RSS_time))
        return safe_T_d

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
        lon_safe_times = SafetyUtils.get_lon_safety(ego, SAFETY_MARGIN_TIME_DELAY,
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
