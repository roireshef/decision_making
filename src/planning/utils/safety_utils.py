from logging import Logger
from typing import List

import numpy as np
import time

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS, SAFETY_SAMPLING_RESOLUTION, LAT_ACC_LIMITS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.types import LIMIT_MIN, FS_SV, FS_SX, FrenetState2D, FS_SA, FS_DX, FS_DV, FS_DA
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from mapping.src.service.map_service import MapService


class SafetyUtils:

    @staticmethod
    def calc_safety_and_jerk(behavioral_state: BehavioralGridState, ego_init_fstate: FrenetState2D,
                             specs: List[ActionSpec], action_specs_mask: List[bool]) -> \
            [np.array, np.array, np.array]:
        """
        Calculate longitudinally safe intervals w.r.t. F, LF/RF, LB/RB, that are permitted for for lateral motion
        for all action specifications.
        If an action does not perform lateral motion, it should be safe in the whole interval [0, spec.t].
        :param behavioral_state: the current behavioral state
        :param ego_init_fstate: the current ego Frenet state
        :param specs: list of action specifications
        :param action_specs_mask: array of boolean values: mask[i]=True if specs[i] was not filtered
        :return: for each action, array of time intervals inside [0, spec.t], where ego is safe
        """
        ego = behavioral_state.ego_state
        (ego_length, ego_width) = (ego.size.length, ego.size.width)
        lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width

        # create a matrix of all non-filtered specs with their original index
        specs_arr = np.array([np.array([i, spec.t, spec.v, spec.s, spec.d])
                              for i, spec in enumerate(specs) if action_specs_mask[i]])
        spec_orig_idxs = specs_arr[:, 0].astype(int)
        (specs_t, specs_v, specs_s, specs_d) = (specs_arr[:, 1], specs_arr[:, 2], specs_arr[:, 3], specs_arr[:, 4])

        # create time samples array for the safety sampling
        # since we vectorize the safety calculations, we create a common matrix of time samples for all specs with
        # number of columns according to the longest spec: np.max(specs_t)
        max_t = np.max(specs_t)
        sampling_step = max_t / (SAFETY_SAMPLING_RESOLUTION * np.round(max_t))  # the last sample is max_t
        time_samples = np.arange(0, max_t + np.finfo(np.float16).eps, sampling_step)
        # for each spec calculate the samples number until the real spec length: spec.t
        samples_num_by_spec_t = (specs_t / sampling_step).astype(int) + 1
        samples_num = time_samples.shape[0]
        actions_num = specs_t.shape[0]
        (lon_front, lon_same, lon_rear) = (RelativeLongitudinalPosition.FRONT, RelativeLongitudinalPosition.PARALLEL, RelativeLongitudinalPosition.REAR)
        grid = behavioral_state.road_occupancy_grid

        # get origin & target relative lane for all actions, based on the current and target latitudes
        # origin_lane may be different from the current ego lane, if ego elapsed > half lateral distance to target
        origin_lanes, target_lanes = SafetyUtils._get_rel_lane_from_specs(lane_width, ego_init_fstate, specs_d)

        # duplicate time_samples array actions_num times
        dup_time_samples = np.tile(time_samples, actions_num).reshape((actions_num, samples_num))

        # calculate ego trajectories for all actions and time samples
        ego_x, ego_vx = SafetyUtils._calc_longitudinal_ego_trajectories(ego_init_fstate, specs_t, specs_v, specs_s,
                                                                        dup_time_samples)

        # pick current states of the followed objects (e.g. F, LF, RF) for all actions
        # for each action/row hold 3 elements: FS_SX, FS_SV, object's length
        # TODO: this code does not support recipes follow parallel or follow back car
        follow_states = np.array([np.append(grid[(tar_lane, lon_front)][0].fstate[:FS_DV],
                                            grid[(tar_lane, lon_front)][0].dynamic_object.size.length,
                                            grid[(tar_lane, lon_front)][0].dynamic_object.size.width)
                                  if (tar_lane, lon_front) in grid  # if followed object exists
                                  else np.array([np.inf, 0, 0, 0, 0, 0])  # there is no followed object: always safe
                                  for tar_lane in target_lanes])

        # duplicate follow_states samples_num times
        dup_follow = np.tile(follow_states, samples_num).reshape((actions_num, samples_num, 6))
        follow_v = dup_follow[:, :, 1]  # matrix of velocities of the followed objects in all time_samples
        follow_x = dup_follow[:, :, 0] + dup_time_samples * follow_v  # matrix of longitudes of the followed objects
        follow_y = dup_follow[:, :, 3]  # matrix of latitudes of the followed objects
        follow_margins_x = (ego_length + dup_follow[:, :, 4]) / 2  # matrix of car size margins
        follow_margins_y = (ego_width + dup_follow[:, :, 5]) / 2  # matrix of car size margins
        # calculate boolean matrix: the safety w.r.t. the followed objects in all time_samples
        follow_safe_times = SafetyUtils._get_lon_safe_times(follow_x, follow_v, ego_x, ego_vx, follow_margins_x,
                                                            SAFETY_MARGIN_TIME_DELAY)
        # set follow_safe_times=True for times beyond spec.t
        follow_safe_times = np.array([np.concatenate((row[:samples_num_by_spec_t[i]],
                                                      np.repeat(True, samples_num - samples_num_by_spec_t[i])))
                                      for i, row in enumerate(follow_safe_times)])

        # pick current states of the front objects (F) on the original lane for all actions
        # for each action/row hold 3 elements: FS_SX, FS_SV, object's length
        front_states = np.array([np.array([-np.inf, 0, 0, 0, 0, 0])  # always unsafe
                                 if (orig_lane, lon_same) in grid   # if F is very close
                                 else np.array([np.inf, 0, 0, 0, 0, 0])  # always safe
                                 if orig_lane == target_lanes[i] or (orig_lane, lon_front) not in grid # no change lane or F does not exist
                                 else np.append(grid[(orig_lane, lon_front)][0].fstate[:FS_DV],  # F
                                           grid[(orig_lane, lon_front)][0].dynamic_object.size.length,
                                           grid[(orig_lane, lon_front)][0].dynamic_object.size.width)
                                 for i, orig_lane in enumerate(origin_lanes)])

        # duplicate front_states samples_num times
        dup_front = np.tile(front_states, samples_num).reshape((actions_num, samples_num, 6))
        front_v = dup_front[:, :, 1]  # matrix of velocities of the front objects in all time_samples
        front_x = dup_front[:, :, 0] + dup_time_samples * front_v  # matrix of longitudes of the front objects
        front_y = dup_front[:, :, 3]  # matrix of latitudes of the front objects in all time samples
        front_margins_x = (ego_length + dup_front[:, :, 4]) / 2  # matrix of car size margins
        front_margins_y = (ego_width + dup_front[:, :, 5]) / 2  # matrix of car size margins
        # TODO: increase time_delay w.r.t. front object F
        # calculate boolean matrix: the safety w.r.t. the front objects in all time_samples
        front_safe_times = SafetyUtils._get_lon_safe_times(front_x, front_v, ego_x, ego_vx, front_margins_x,
                                                           SAFETY_MARGIN_TIME_DELAY)

        # pick current states of the back or parallel objects (LB/RB/L/R) on the target lane for all actions
        # for each action/row hold 3 elements: FS_SX, FS_SV, object's length
        # TODO: this code does not support recipes follow parallel or follow back car
        back_states = np.array([np.array([-np.inf, 0, 0, 0, 0, 0])  # always safe
                                if tar_lane == origin_lanes[i]  # if staying on the same lane, then always safe
                                else np.append(grid[(tar_lane, lon_same)][0].fstate[:FS_DV],  # overtake lon_parallel
                                          grid[(tar_lane, lon_same)][0].dynamic_object.size.length,
                                          grid[(tar_lane, lon_same)][0].dynamic_object.size.width)
                                if (tar_lane, lon_same) in grid  # if L/R (parallel) exists, overtake it instead of LB
                                else np.append(grid[(tar_lane, lon_rear)][0].fstate[:FS_DV],  # overtake LB
                                          grid[(tar_lane, lon_rear)][0].dynamic_object.size.length,
                                          grid[(tar_lane, lon_rear)][0].dynamic_object.size.width)
                                if (tar_lane, lon_rear) in grid  # if LB exists but L does not exist, overtake LB
                                else np.array([-np.inf, 0, 0, 0, 0, 0])  # L & LB don't exist, then always safe
                                for i, tar_lane in enumerate(target_lanes)])

        # duplicate back_states samples_num times
        dup_back = np.tile(back_states, samples_num).reshape((actions_num, samples_num, 6))
        back_v = dup_back[:, :, 1]  # matrix of velocities of the back objects in all time_samples
        back_x = dup_back[:, :, 0] + dup_time_samples * back_v  # matrix of longitudes of the back objects
        back_y = dup_back[:, :, 3]  # matrix of latitudes of the back objects
        back_margins_x = (ego_length + dup_back[:, :, 4]) / 2  # matrix of car size margins
        back_margins_y = (ego_width + dup_back[:, :, 5]) / 2  # matrix of car size margins
        # calculate boolean matrix: the safety w.r.t. the back objects in all time_samples
        back_safe_times = SafetyUtils._get_lon_safe_times(ego_x, ego_vx, back_x, back_v, back_margins_x,
                                                          SPECIFICATION_MARGIN_TIME_DELAY)

        # calculate the range of safe indices based on LF and F
        last_unsafe_ind = samples_num - np.fliplr(follow_safe_times).argmin(axis=1)  # last unsafe index w.r.t. LF
        # if a whole row is True, argmin returns 0, i.e. last_unsafe_ind = samples_num. So we set follow_safe_from = 0
        follow_lon_safe_from = last_unsafe_ind * (1 - follow_safe_times.all(axis=1).astype(int))

        # the first unsafe time w.r.t. F per spec
        # if all times are safe, argmin returns 0, so we set front_safe_till=inf instead of 0
        front_lon_safe_till = front_safe_times.argmin(axis=1) + samples_num * front_safe_times.all(axis=1).astype(int)

        # find safe intervals inside the above range and based on LB
        # the first safe time w.r.t. LB per spec
        back_lon_safe_from = back_safe_times.argmax(axis=1) + samples_num * (1 - back_safe_times.any(axis=1).astype(int))
        # the first unsafe time w.r.t. LB per spec, after back_safe_from
        back_safe_till = [back_lon_safe_from[i] + row[back_lon_safe_from[i]:].argmin() +
                          samples_num * row[back_lon_safe_from[i]:].all().astype(int)
                          for i, row in enumerate(back_safe_times)]
        back_safe_till_time = time_samples[np.clip(back_safe_till, 0, samples_num-1)]

        # ext_safe_times = np.c_[np.repeat(False, actions_num), back_safe_times, np.repeat(False, actions_num)]
        # # find all start-points of all safe intervals w.r.t. LB, by horizontal shifting of back_safe_times matrix
        # # start_points[0] is array of action (rows) indices of the start-points
        # # start_points[1] is array of time_samples (columns) indices of the start-points
        # start_points = np.where(np.logical_and(ext_safe_times[:, 1:], np.logical_not(ext_safe_times[:, :-1])))
        # # clip the intervals' start points by the first safe time w.r.t. the followed object
        # clip_start_points = np.maximum(follow_safe_from[start_points[0]], start_points[1]).astype(int)
        # # find all end points of all safe intervals w.r.t. LB, by horizontal shifting of back_safe_times matrix
        # # end_points[0] is array of action (rows) indices of the end-points
        # # end_points[1] is array of time_samples (columns) indices of the end-points
        # end_points = np.where(np.logical_and(ext_safe_times[:, :-1], np.logical_not(ext_safe_times[:, 1:])))
        # # clip the intervals' end points by the last safe time w.r.t. the front object
        # clip_end_points = np.minimum(front_safe_till[end_points[0]], end_points[1]).astype(int)

        # pick safe intervals for all actions, by filtering empty intervals
        # intervals = np.array([(spec_orig_idxs[spec_i],
        #                       time_samples[clip_start_points[i]], min(specs_t[spec_i], time_samples[clip_end_points[i]-1]))
        #                      for i, spec_i in enumerate(start_points[0])
        #                      if clip_start_points[i] < len(time_samples) and clip_end_points[i] > 0 and
        #                       time_samples[clip_start_points[i]] <= min(specs_t[spec_i], time_samples[clip_end_points[i]-1])])

        front_safe_till_time = np.minimum(specs_t, time_samples[np.clip(front_lon_safe_till, 0, samples_num-1)])
        ego_y, ego_vy = SafetyUtils._calc_lateral_ego_trajectories(ego_init_fstate, front_safe_till_time, specs_d,
                                                                   time_samples)

        back_vy = follow_vy = np.zeros(ego_y.shape)
        follow_lat_safe_times = SafetyUtils._get_lat_safe_times(
            follow_y, follow_vy, SAFETY_MARGIN_TIME_DELAY, ego_y, ego_vy, SAFETY_MARGIN_TIME_DELAY, follow_margins_y)
        # the first laterally unsafe time w.r.t. LF per spec
        follow_lat_safe_till = follow_lat_safe_times.argmin(axis=1) + samples_num * follow_lat_safe_times.all(axis=1).astype(int)
        follow_safe = (follow_lon_safe_from <= follow_lat_safe_till)

        back_lat_safe_times = SafetyUtils._get_lat_safe_times(
            back_y, back_vy, SAFETY_MARGIN_TIME_DELAY, ego_y, ego_vy, SAFETY_MARGIN_TIME_DELAY, back_margins_y)

        # the first laterally unsafe time w.r.t. LB per spec
        back_lat_safe_till = back_lat_safe_times.argmin(axis=1) + samples_num * back_lat_safe_times.all(axis=1).astype(int)
        back_safe = (back_lon_safe_from <= back_lat_safe_till)

        # calculate lateral jerk required to be safe w.r.t. LB and F
        (da, dv, dx) = (ego_init_fstate[FS_DA], ego_init_fstate[FS_DV], specs_d - ego_init_fstate[FS_DX])
        back_lat_jerk = QuinticPoly1D.cumulative_jerk_from_constraints(da, dv, 0, dx - back_margins_y, back_safe_till_time)
        front_lat_jerk = QuinticPoly1D.cumulative_jerk_from_constraints(da, dv, 0, dx, front_safe_till_time)

        return np.logical_and(follow_safe, back_safe), np.maximum(back_lat_jerk, front_lat_jerk)

    @staticmethod
    def _get_rel_lane_from_specs(lane_width: float, ego_fstate: FrenetState2D, specs_d: np.array) -> \
            [List[RelativeLane], List[RelativeLane]]:
        """
        Return origin and target relative lanes
        :param lane_width:
        :param ego_fstate:
        :param specs_d:
        :return: origin & target relative lanes
        """
        specs_num = specs_d.shape[0]
        d_dist = specs_d - ego_fstate[FS_DX]

        # same_same_idxs = np.where(-lane_width/4 <= d_dist <= lane_width/4)[0]
        right_same_idxs = np.where(np.logical_and(lane_width/4 < d_dist, d_dist <= lane_width/2))
        left_same_idxs = np.where(np.logical_and(-lane_width/2 <= d_dist, d_dist < -lane_width/4))
        same_left_idxs = np.where(d_dist > lane_width/2)
        same_right_idxs = np.where(d_dist < -lane_width/2)

        origin_lanes = np.array([RelativeLane.SAME_LANE] * specs_num)
        target_lanes = np.array([RelativeLane.SAME_LANE] * specs_num)
        origin_lanes[right_same_idxs] = RelativeLane.RIGHT_LANE
        origin_lanes[left_same_idxs] = RelativeLane.LEFT_LANE
        target_lanes[same_left_idxs] = RelativeLane.LEFT_LANE
        target_lanes[same_right_idxs] = RelativeLane.RIGHT_LANE
        return list(origin_lanes), list(target_lanes)

    @staticmethod
    def _calc_longitudinal_ego_trajectories(ego_init_fstate: FrenetState2D,
                                            specs_t: np.array, specs_v: np.array, specs_s: np.array,
                                            time_samples: np.array) -> [np.array, np.array]:
        """
        Calculate longitudinal ego trajectory for the given time samples.
        :param ego_init_fstate:
        :param specs:
        :param time_samples:
        :return: x_profile, v_profile
        """
        # TODO: Acceleration is not calculated.

        trans_times = np.transpose(time_samples)
        dx = specs_s - ego_init_fstate[FS_SX]
        # profiles for the cases, when dynamic object is in front of ego
        x_t = QuinticPoly1D.distance_by_constraints(a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV],
                                                    v_T=specs_v, ds=dx, T=specs_t)
        x = np.transpose(x_t(trans_times))
        v_t = QuinticPoly1D.velocity_by_constraints(a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV],
                                                    v_T=specs_v, ds=dx, T=specs_t)
        v = np.transpose(v_t(trans_times))
        return ego_init_fstate[FS_SX] + x, v

    @staticmethod
    def _calc_lateral_ego_trajectories(ego_init_fstate: FrenetState2D, specs_t: np.array, specs_d: np.array,
                                       time_samples: np.array) -> [np.array, np.array]:
        """
        Calculate longitudinal ego trajectory for the given time samples.
        :param ego_init_fstate:
        :param specs:
        :param time_samples:
        :return: x_profile, v_profile
        """
        # TODO: Acceleration is not calculated.

        trans_times = np.transpose(time_samples)
        dd = specs_d - ego_init_fstate[FS_DX]
        # profiles for the cases, when dynamic object is in front of ego
        d_t = QuinticPoly1D.distance_by_constraints(a_0=ego_init_fstate[FS_DA], v_0=ego_init_fstate[FS_DV],
                                                    v_T=0, ds=dd, T=specs_t)
        y = np.transpose(d_t(trans_times))
        v_t = QuinticPoly1D.velocity_by_constraints(a_0=ego_init_fstate[FS_DA], v_0=ego_init_fstate[FS_DV],
                                                    v_T=0, ds=dd, T=specs_t)
        vy = np.transpose(v_t(trans_times))
        return ego_init_fstate[FS_DX] + y, vy

    @staticmethod
    def _get_lon_safe_times(front_x: np.array, front_v: np.array, back_x: np.array, back_v: np.array,
                            cars_length_margin: np.array, time_delay: float) -> np.array:
        """
        Calculate bool matrix of time samples complying longitudinal safety for given front & back Frenet trajectories.
        :param front_x: NxM matrix of longitudes of the front vehicle, where N is actions number, M time samples number
        :param front_v: NxM matrix of velocities of the front vehicle
        :param back_x: NxM matrix of longitudes of the back vehicle
        :param back_v: NxM matrix of velocities of the back vehicle
        :param cars_length_margin: half sum of the vehicles lengths
        :param time_delay: reaction delay of the back vehicle
        :return: array of booleans: which time samples are safe
        """
        safety_dist = SafetyUtils._get_lon_safety_dist(front_v, back_v, front_x - back_x, time_delay, cars_length_margin)
        return safety_dist > 0

    @staticmethod
    def _get_lat_safe_times(y1: np.array, vy1: np.array, delay1: float, y2: np.array, vy2: np.array, delay2: float,
                            margins: np.array) -> np.array:
        if np.sum(y1) > np.sum(y2):  # first vehicle is on the left side
            safety_dist = SafetyUtils._get_lat_safety_dist(vy1, vy2, y1 - y2, delay1, delay2, margins)
        else:
            safety_dist = SafetyUtils._get_lat_safety_dist(vy2, vy1, y2 - y1, delay2, delay1, margins)
        return safety_dist > 0

    @staticmethod
    def _get_lon_safety_dist(v_front: np.array, v_back: np.array, dist: np.array, time_delay: float, margin: float,
                             max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate differences between the actual distances and minimal safe distances (longitudinal RSS formula)
        :param v_front: [m/s] matrix of front vehicle velocities
        :param v_back: [m/s] matrix of back vehicle velocities
        :param dist: [m] matrix of distances between the vehicles
        :param time_delay: [sec] time delays of the back vehicle
        :param margin: [m] cars size margin
        :param max_brake: [m/s^2] maximal deceleration of the vehicles
        :return: Array of safety distances: positive if the back vehicle is safe, negative otherwise
        """
        safe_dist = np.clip(v_back**2 - v_front**2, 0, None) / (2*max_brake) + v_back*time_delay + margin
        return dist - safe_dist

    @staticmethod
    def _get_lat_safety_dist(v_left: np.array, v_right: np.array, dist: np.array, left_delay: float, right_delay: float,
                             margin: float, max_brake: float=-LAT_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate differences between the actual distances and minimal safe distances (longitudinal RSS formula)
        :param v_left: [m/s] matrix of front vehicle velocities
        :param v_right: [m/s] matrix of back vehicle velocities
        :param dist: [m] matrix of distances between the vehicles
        :param left_delay: [sec] time delay of the left vehicle
        :param right_delay: [sec] time delays of the right vehicle
        :param margin: [m] cars size margin
        :param max_brake: [m/s^2] maximal lateral deceleration of the vehicles
        :return: Array of safety distances: positive if both vehicle is safe, negative otherwise
        """
        safe_dist = np.clip(v_right * np.abs(v_right) - v_left * np.abs(v_left), 0, None) / (2*max_brake) + \
                            np.clip(v_right, 0, None) * right_delay + np.clip(-v_left, 0, None) * left_delay + \
                    margin
        return dist - safe_dist
