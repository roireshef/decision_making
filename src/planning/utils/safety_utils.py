from logging import Logger
from typing import List

import numpy as np
import time

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS, SAFETY_SAMPLING_RESOLUTION
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.types import LIMIT_MIN, FS_SV, FS_SX, FrenetState2D, FS_SA, FS_DX, FS_DV
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from mapping.src.service.map_service import MapService


class SafetyUtils:

    @staticmethod
    def calc_safe_intervals_for_specs(behavioral_state: BehavioralGridState, ego_init_fstate: FrenetState2D,
                                      specs: List[ActionSpec]) -> np.array:

        st = time.time()

        ego = behavioral_state.ego_state
        ego_length = ego.size.length
        lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width

        specs_arr = np.array([np.array([i, spec.t, spec.v, spec.s, spec.d])
                              for i, spec in enumerate(specs) if spec is not None])
        spec_orig_idxs = specs_arr[:, 0].astype(int)
        specs_t = specs_arr[:, 1]
        specs_v = specs_arr[:, 2]
        specs_s = specs_arr[:, 3]
        specs_d = specs_arr[:, 4]

        max_t = np.max(specs_t)
        sampling_step = max_t / (SAFETY_SAMPLING_RESOLUTION * np.round(max_t))  # the last sample is max_t
        time_samples = np.arange(0, max_t + np.finfo(np.float16).eps, sampling_step)
        samples_num_by_spec_t = (specs_t / sampling_step).astype(int) + 1
        samples_num = time_samples.shape[0]
        actions_num = specs_t.shape[0]

        origin_lanes, target_lanes = SafetyUtils._get_rel_lane_from_specs(lane_width, ego_init_fstate, specs_d)

        (lon_front, lon_same, lon_rear) = (RelativeLongitudinalPosition.FRONT, RelativeLongitudinalPosition.PARALLEL, RelativeLongitudinalPosition.REAR)
        grid = behavioral_state.road_occupancy_grid

        dup_time_samples = np.tile(time_samples, actions_num).reshape((actions_num, samples_num))
        ego_x, ego_v = SafetyUtils._calc_longitudinal_ego_trajectories(ego_init_fstate, specs_t, specs_v, specs_s,
                                                                       dup_time_samples)

        follow_states = np.array([np.append(grid[(tar_lane, lon_front)][0].fstate[:FS_SA],
                                            grid[(tar_lane, lon_front)][0].dynamic_object.size.length)
                                  if (tar_lane, lon_front) in grid
                                  else np.array([-np.inf, 1, 0]) if (tar_lane, lon_same) in grid
                                  else np.array([np.inf, 1, 0])
                                  for tar_lane in target_lanes])
        dup_follow = np.tile(follow_states, samples_num).reshape((actions_num, samples_num, 3))
        follow_v = dup_follow[:, :, 1]
        follow_x = dup_follow[:, :, 0] + dup_time_samples * follow_v
        follow_margins = (ego_length + dup_follow[:, :, 2]) / 2
        follow_safe_times = SafetyUtils._get_safe_times(follow_x, follow_v, ego_x, ego_v, follow_margins,
                                                        SAFETY_MARGIN_TIME_DELAY)
        # set follow_safe_times=True for times beyond spec.t
        follow_safe_times = np.array([np.concatenate((row[:samples_num_by_spec_t[i]],
                                                      np.repeat(True, samples_num - samples_num_by_spec_t[i])))
                                      for i, row in enumerate(follow_safe_times)])

        front_states = np.array([np.append(grid[(orig_lane, lon_front)][0].fstate[:FS_SA],
                                           grid[(orig_lane, lon_front)][0].dynamic_object.size.length)
                                 if orig_lane != target_lanes[i] and (orig_lane, lon_front) in grid
                                 else np.array([-np.inf, 1, 0]) if (orig_lane, lon_same) in grid
                                 else np.array([np.inf, 1, 0])
                                 for i, orig_lane in enumerate(origin_lanes)])
        dup_front = np.tile(front_states, samples_num).reshape((actions_num, samples_num, 3))
        front_v = dup_front[:, :, 1]
        front_x = dup_front[:, :, 0] + dup_time_samples * front_v
        front_margins = (ego_length + dup_front[:, :, 2]) / 2
        # TODO: increase time_delay w.r.t. front object F
        front_safe_times = SafetyUtils._get_safe_times(front_x, front_v, ego_x, ego_v, front_margins, time_delay=0)

        back_states = np.array([np.append(grid[(tar_lane, lon_rear)][0].fstate[:FS_SA],
                                          grid[(tar_lane, lon_rear)][0].dynamic_object.size.length)
                                if tar_lane != origin_lanes[i] and (tar_lane, lon_rear) in grid else np.array([-np.inf, 1, 0])
                                for i, tar_lane in enumerate(target_lanes)])
        dup_back = np.tile(back_states, samples_num).reshape((actions_num, samples_num, 3))
        back_v = dup_back[:, :, 1]
        back_x = dup_back[:, :, 0] + dup_time_samples * back_v
        back_margins = (ego_length + dup_back[:, :, 2]) / 2
        back_safe_times = SafetyUtils._get_safe_times(ego_x, ego_v, back_x, back_v, back_margins,
                                                      SPECIFICATION_MARGIN_TIME_DELAY)

        # calculate the range of safe indices based on LF and F
        last_unsafe_ind = samples_num - np.fliplr(follow_safe_times).argmin(axis=1)  # last unsafe index w.r.t. LF
        # if all times are safe, argmin returns 0, i.e. last_unsafe_ind=samples_num, so we set follow_safe_from=0
        follow_safe_from = last_unsafe_ind * (np.logical_not(follow_safe_times.min(axis=1))).astype(int)
        front_safe_till = front_safe_times.argmin(axis=1)  # the first unsafe time w.r.t. F per spec
        # if all times are safe, argmin returns 0, so set front_safe_till=inf instead of 0
        front_safe_till += samples_num * front_safe_times.min(axis=1).astype(int)

        # find safe intervals inside the above range and based on LB
        ext_safe_times = np.c_[np.repeat(False, actions_num), back_safe_times, np.repeat(False, actions_num)]

        start_points = np.where(np.logical_and(ext_safe_times[:, 1:], np.logical_not(ext_safe_times[:, :-1])))
        clip_start_points = np.maximum(follow_safe_from[start_points[0]], start_points[1]).astype(int)

        end_points = np.where(np.logical_and(ext_safe_times[:, :-1], np.logical_not(ext_safe_times[:, 1:])))
        clip_end_points = np.minimum(front_safe_till[end_points[0]], end_points[1]).astype(int)

        intervals = np.array([(spec_orig_idxs[spec_i],
                              time_samples[clip_start_points[i]], min(specs_t[spec_i], time_samples[clip_end_points[i]-1]))
                             for i, spec_i in enumerate(start_points[0])
                             if clip_start_points[i] < len(time_samples) and clip_end_points[i] > 0 and
                              time_samples[clip_start_points[i]] <= min(specs_t[spec_i], time_samples[clip_end_points[i]-1])])

        print('safety time=%f' % (time.time()-st))
        return intervals

    # @staticmethod
    # def calc_safe_intervals_for_lane_change(behavioral_state: BehavioralGridState, ego_init_fstate: FrenetState2D,
    #                                         spec: ActionSpec, test_followed_obj: bool) -> np.array:
    #     """
    #     Calculate time intervals, where ego is safe w.r.t. F, LF/RF, LB/RB.
    #     :param behavioral_state: current behavioral state
    #     :param ego_init_fstate: initial ego Frenet state
    #     :param spec: action specification
    #     :param test_followed_obj: whether to test safety w.r.t. the followed object (e.g. LF), used by static actions
    #     :return: Nx2 matrix of safe intervals (each row is start & end time of interval)
    #     """
    #     ego = behavioral_state.ego_state
    #     ego_length = ego.size.length
    #     rel_origin_lane, rel_target_lane = \
    #         SafetyUtils._get_rel_lane_from_spec(ego.road_localization.road_id, ego_init_fstate, spec)
    #     # lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width
    #
    #     forward_cell = (rel_target_lane, RelativeLongitudinalPosition.FRONT)
    #     front_cell = (rel_origin_lane, RelativeLongitudinalPosition.FRONT)
    #     side_rear_cell = (rel_target_lane, RelativeLongitudinalPosition.REAR)
    #
    #     # create time samples for safety checking
    #     sampling_step = spec.t / (SAFETY_SAMPLING_RESOLUTION * np.round(spec.t))  # the last sample is spec.t
    #     time_samples = np.arange(0, spec.t + np.finfo(np.float16).eps, sampling_step)
    #     samples_num = len(time_samples)
    #     zeros = np.zeros(samples_num)
    #     ego_ftrajectory = None
    #
    #     forward_safe_times = np.array([True]*samples_num)
    #     front_safe_times = np.array([True]*samples_num)
    #     back_safe_times = np.array([True]*samples_num)
    #
    #     t1 = t2 = t3 = t4 = t5 = t6 = t7 = 0
    #     st = time.time()
    #
    #     # filter action if there is a side car
    #     if (rel_origin_lane, RelativeLongitudinalPosition.PARALLEL) in behavioral_state.road_occupancy_grid:
    #         return np.array([])
    #
    #     # check safety w.r.t. the followed object on the target lane (if exists)
    #     if test_followed_obj and forward_cell in behavioral_state.road_occupancy_grid:
    #
    #         st7 = time.time()
    #         cell = behavioral_state.road_occupancy_grid[forward_cell][0]
    #         obj_ftrajectory = np.c_[cell.fstate[FS_SX] + cell.fstate[FS_SV] * time_samples,
    #                                 np.repeat(cell.fstate[FS_SV], samples_num)]
    #         t7 = time.time() - st7
    #
    #         # create longitudinal ego frenet trajectory for the time samples based on the spec
    #         if ego_ftrajectory is None:
    #             st5 = time.time()
    #             ego_ftrajectory = SafetyUtils._calc_longitudinal_ego_trajectory(ego_init_fstate, spec, time_samples)
    #             t5 = time.time() - st5
    #
    #         st6 = time.time()
    #         margin = (ego.size.length + cell.dynamic_object.size.length)/2
    #         forward_safe_times = SafetyUtils._get_safe_time_samples(obj_ftrajectory, ego_ftrajectory, margin, SAFETY_MARGIN_TIME_DELAY)
    #         t6 = time.time() - st6
    #
    #         # don't enable the action to enter to the two-seconds distance from the followed object
    #         # start_spec_dist = SafetyUtils._get_safety_dist_for_states(
    #         #     obj_ftrajectory[0], ego_init_fstate, SPECIFICATION_MARGIN_TIME_DELAY, margin)
    #         # end_spec_dist = SafetyUtils._get_safety_dist_for_states(
    #         #     obj_ftrajectory[-1], ego_ftrajectory[-1], SPECIFICATION_MARGIN_TIME_DELAY, margin)
    #         # if end_spec_dist <= min(0., start_spec_dist):
    #         #     forward_safe_times[-1] = False
    #
    #     t1 = time.time() - st
    #
    #     # change lane
    #     if rel_origin_lane != rel_target_lane:
    #
    #         st = time.time()
    #
    #         # filter action if there is a side car
    #         if (rel_target_lane, RelativeLongitudinalPosition.PARALLEL) in behavioral_state.road_occupancy_grid:
    #             return np.array([])
    #
    #         # check safety w.r.t. the front object F on the original lane (if exists)
    #         if front_cell in behavioral_state.road_occupancy_grid:
    #             # calculate last safe time w.r.t. F
    #             cell = behavioral_state.road_occupancy_grid[front_cell][0]
    #             obj_ftrajectory = np.c_[cell.fstate[FS_SX] + cell.fstate[FS_SV] * time_samples,
    #                                     np.repeat(cell.fstate[FS_SV], samples_num)]
    #
    #             # time delay decreases as function of lateral distance to the target,
    #             # since as latitude advances the lateral escape is easier
    #             # td_0 = SAFETY_MARGIN_TIME_DELAY * abs(spec.d - ego_init_fstate[FS_DX]) / lane_width
    #             # td_T = 0  # dist to F after completing lane change. TODO: increase it when the planning will be deep
    #             # time_delay = np.arange(td_0 + np.finfo(np.float16).eps, td_T, (td_T - td_0) / (len(time_samples) - 1))
    #
    #             # create longitudinal ego ftrajectory for the time samples based on the spec
    #             if ego_ftrajectory is None:
    #                 ego_ftrajectory = SafetyUtils._calc_longitudinal_ego_trajectory(ego_init_fstate, spec, time_samples)
    #
    #             front_safe_times = SafetyUtils._get_safe_time_samples(
    #                 obj_ftrajectory, ego_ftrajectory, (ego_length + cell.dynamic_object.size.length) / 2, time_delay=0)
    #
    #         t2 = time.time() - st
    #         st = time.time()
    #
    #         # check safety w.r.t. the back object LB / RB on the target lane (if exists)
    #         if side_rear_cell in behavioral_state.road_occupancy_grid:
    #             cell = behavioral_state.road_occupancy_grid[side_rear_cell][0]
    #             obj_ftrajectory = np.c_[cell.fstate[FS_SX] + cell.fstate[FS_SV] * time_samples,
    #                                     np.repeat(cell.fstate[FS_SV], samples_num)]
    #             # time_delay = np.repeat(SPECIFICATION_MARGIN_TIME_DELAY, len(time_samples))
    #
    #             # create longitudinal ego frenet trajectory for the time samples based on the spec
    #             if ego_ftrajectory is None:
    #                 ego_ftrajectory = SafetyUtils._calc_longitudinal_ego_trajectory(ego_init_fstate, spec, time_samples)
    #
    #             back_safe_times = SafetyUtils._get_safe_time_samples(
    #                 ego_ftrajectory, obj_ftrajectory, (ego_length + cell.dynamic_object.size.length) / 2, SPECIFICATION_MARGIN_TIME_DELAY)
    #
    #         t3 = time.time() - st
    #
    #     st = time.time()
    #
    #     # calculate the range of safe indices based on LF and F
    #     unsafe_idxs = np.where(np.logical_not(forward_safe_times))[0]
    #     forward_safe_from = 0
    #     if len(unsafe_idxs) > 0:
    #         forward_safe_from = unsafe_idxs[-1] + 1
    #     unsafe_idxs = np.where(np.logical_not(front_safe_times))[0]
    #     front_safe_till = samples_num
    #     if len(unsafe_idxs) > 0:
    #         front_safe_till = unsafe_idxs[0]
    #
    #     # find safe intervals inside the above range and based on LB
    #     ext_safe_times = np.concatenate(([False], back_safe_times[forward_safe_from:front_safe_till], [False]))
    #     start_points = np.where(np.logical_and(ext_safe_times[1:], np.logical_not(ext_safe_times[:-1])))[0]
    #     end_points = np.where(np.logical_and(ext_safe_times[:-1], np.logical_not(ext_safe_times[1:])))[0]
    #     intervals = np.c_[time_samples[forward_safe_from + start_points], time_samples[forward_safe_from + end_points-1]]
    #
    #     t4 = time.time() - st
    #     print('times: t1=%f (%f %f %f) t2=%f t3=%f t4=%f tot=%f' % (t1, t7, t5, t6, t2, t3, t4, t1+t2+t3+t4))
    #
    #     return intervals

    @staticmethod
    def _get_rel_lane_from_specs(lane_width: float, ego_fstate: FrenetState2D, specs_d: np.array) -> \
            [List[RelativeLane], List[RelativeLane]]:
        """
        Return origin and target relative lanes
        :param lane_width:
        :param ego_fstate:
        :param spec:
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

    # @staticmethod
    # def _calc_longitudinal_ego_trajectory(ego_init_fstate: FrenetState2D, spec: ActionSpec, time_samples: np.array) -> \
    #         FrenetTrajectory2D:
    #     """
    #     Calculate longitudinal ego trajectory for the given time samples.
    #     :param ego_init_fstate:
    #     :param spec:
    #     :param time_samples:
    #     :return:
    #     """
    #     dx = spec.s - ego_init_fstate[FS_SX]
    #     # profiles for the cases, when dynamic object is in front of ego
    #     dist_profile = QuinticPoly1D.distance_by_constraints(a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV],
    #                                                          v_T=spec.v, ds=dx, T=spec.t)(time_samples)
    #     vel_profile = QuinticPoly1D.velocity_by_constraints(a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV],
    #                                                         v_T=spec.v, ds=dx, T=spec.t)(time_samples)
    #     zeros = np.zeros(len(time_samples))
    #     ego_fstates = np.c_[ego_init_fstate[FS_SX] + dist_profile, vel_profile]
    #     return ego_fstates

    @staticmethod
    def _get_safe_times(front_x: np.array, front_v: np.array, back_x: np.array, back_v: np.array,
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
        safety_dist = SafetyUtils._get_safety_dist(front_v, back_v, front_x - back_x, time_delay, cars_length_margin)
        return safety_dist > 0

    # @staticmethod
    # def _get_safe_time_samples(front_traj: FrenetTrajectory2D, back_traj: FrenetTrajectory2D,
    #                            cars_length_margin: float, time_delay: float) -> np.array:
    #     """
    #     Calculate bool array of time samples complying longitudinal safety for given front & back Frenet trajectories.
    #     :param front_traj: Frenet trajectory of the front vehicle
    #     :param back_traj: Frenet trajectory of the back vehicle
    #     :param cars_length_margin: half sum of the vehicles lengths
    #     :param time_delay: reaction delay of the back vehicle
    #     :return: array of booleans: which time samples are safe
    #     """
    #     safety_dist = SafetyUtils._get_safety_dist(front_traj[:, FS_SV], back_traj[:, FS_SV],
    #                                                front_traj[:, FS_SX] - back_traj[:, FS_SX], time_delay,
    #                                                cars_length_margin)
    #     return safety_dist > 0

    # @staticmethod
    # def _get_safety_dist_for_states(front: FrenetState2D, back: FrenetState2D, margin: float, td: float,
    #                                 max_brake: float = -LON_ACC_LIMITS[LIMIT_MIN]) -> float:
    #     """
    #     Calculate last time sample complying longitudinal safety for given front & back Frenet trajectories.
    #     :param front: Frenet state of the front vehicle
    #     :param back: Frenet state of the back vehicle
    #     :param margin: half sum of the vehicles lengths
    #     :param td: reaction delay of the back vehicle
    #     :return: safety distance
    #     """
    #     dist = front[FS_SX] - back[FS_SX]
    #     safe_dist = max(back[FS_SV]**2 - front[FS_SV]**2, 0.) / (2*max_brake) + back[FS_SV]*td + margin
    #     return dist - safe_dist

    @staticmethod
    def _get_safety_dist(v_front: np.array, v_back: np.array, dist: np.array, time_delay: float, margin: float,
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
