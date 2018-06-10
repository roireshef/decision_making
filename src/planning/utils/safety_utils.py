from logging import Logger
import numpy as np
import math

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS, BP_ACTION_T_LIMITS, SAFETY_SAMPLING_RESOLUTION
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.types import LIMIT_MIN, FrenetTrajectory2D, FS_SV, FS_SX, FrenetState2D, FS_SA, \
    FS_DX, FS_DV, LIMIT_MAX, FS_DA
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from mapping.src.service.map_service import MapService


class SafetyUtils:

    @staticmethod
    def calc_safe_intervals_for_lane_change(behavioral_state: BehavioralGridState, ego_init_fstate: FrenetState2D,
                                            spec: ActionSpec, test_followed_obj: bool) -> np.array:
        """
        Calculate time intervals, where ego is safe w.r.t. F, LF/RF, LB/RB.
        :param behavioral_state: current behavioral state
        :param ego_init_fstate: initial ego Frenet state
        :param spec: action specification
        :param test_followed_obj: whether to test safety w.r.t. the followed object (e.g. LF), used by static actions
        :return: Nx2 matrix of safe intervals (each row is start & end time of interval)
        """
        ego = behavioral_state.ego_state
        ego_length = ego.size.length
        rel_lane = SafetyUtils._get_rel_lane_from_spec(ego.road_localization.road_id, ego_init_fstate, spec)
        lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width

        forward_cell = (rel_lane, RelativeLongitudinalPosition.FRONT)
        front_cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)
        side_rear_cell = (rel_lane, RelativeLongitudinalPosition.REAR)

        # create time samples for safety checking
        sampling_step = spec.t / (SAFETY_SAMPLING_RESOLUTION * np.round(spec.t))  # the last sample is spec.t
        time_samples = np.arange(0, spec.t + np.finfo(np.float16).eps, sampling_step)
        samples_num = len(time_samples)
        zeros = np.zeros(samples_num)
        ego_ftrajectory = None

        forward_safe_times = np.array([True]*samples_num)
        front_safe_times = np.array([True]*samples_num)
        back_safe_times = np.array([True]*samples_num)

        # check safety w.r.t. the followed object on the target lane (if exists)
        if test_followed_obj and forward_cell in behavioral_state.road_occupancy_grid:
            cell = behavioral_state.road_occupancy_grid[forward_cell][0]
            obj_ftrajectory = np.c_[cell.fstate[FS_SX] + cell.fstate[FS_SV] * time_samples,
                                    cell.fstate[FS_SV] + zeros, zeros, zeros, zeros, zeros]
            time_delay = np.array([SAFETY_MARGIN_TIME_DELAY] * samples_num)

            # create longitudinal ego frenet trajectory for the time samples based on the spec
            if ego_ftrajectory is None:
                ego_ftrajectory = SafetyUtils._calc_longitudinal_ego_trajectory(ego_init_fstate, spec, time_samples)

            margin = (ego.size.length + cell.dynamic_object.size.length)/2
            forward_safe_times = SafetyUtils._get_safe_time_samples(obj_ftrajectory, ego_ftrajectory, margin, time_delay)

            # don't enable the action to enter to the two-seconds distance from the followed object
            start_spec_dist = SafetyUtils._get_safety_dist_for_states(
                obj_ftrajectory[0], ego_init_fstate, SPECIFICATION_MARGIN_TIME_DELAY, margin)
            end_spec_dist = SafetyUtils._get_safety_dist_for_states(
                obj_ftrajectory[-1], ego_ftrajectory[-1], SPECIFICATION_MARGIN_TIME_DELAY, margin)
            if end_spec_dist <= min(0., start_spec_dist):
                forward_safe_times[-1] = False

        if rel_lane != RelativeLane.SAME_LANE:  # change lane

            # filter action if there is a side car
            if (rel_lane, RelativeLongitudinalPosition.PARALLEL) in behavioral_state.road_occupancy_grid:
                return np.array([])

            # check safety w.r.t. the front object F on the original lane (if exists)
            if front_cell in behavioral_state.road_occupancy_grid:
                # calculate last safe time w.r.t. F
                cell = behavioral_state.road_occupancy_grid[front_cell][0]
                obj_ftrajectory = np.c_[cell.fstate[FS_SX] + cell.fstate[FS_SV] * time_samples,
                                        cell.fstate[FS_SV] + zeros, zeros, zeros, zeros, zeros]

                # time delay decreases as function of lateral distance to the target,
                # since as latitude advances the lateral escape is easier
                td_0 = SAFETY_MARGIN_TIME_DELAY * abs(spec.d - ego_init_fstate[FS_DX]) / lane_width
                td_T = 0  # dist to F after completing lane change. TODO: increase it when the planning will be deep
                time_delay = np.arange(td_0 + np.finfo(np.float16).eps, td_T, (td_T - td_0) / (len(time_samples) - 1))

                # create longitudinal ego ftrajectory for the time samples based on the spec
                if ego_ftrajectory is None:
                    ego_ftrajectory = SafetyUtils._calc_longitudinal_ego_trajectory(ego_init_fstate, spec, time_samples)

                front_safe_times = SafetyUtils._get_safe_time_samples(
                    obj_ftrajectory, ego_ftrajectory, (ego_length + cell.dynamic_object.size.length) / 2, time_delay)

            # check safety w.r.t. the back object LB / RB on the target lane (if exists)
            if side_rear_cell in behavioral_state.road_occupancy_grid:
                cell = behavioral_state.road_occupancy_grid[side_rear_cell][0]
                obj_ftrajectory = np.c_[cell.fstate[FS_SX] + cell.fstate[FS_SV] * time_samples,
                                        cell.fstate[FS_SV] + zeros, zeros, zeros, zeros, zeros]
                time_delay = np.repeat(SPECIFICATION_MARGIN_TIME_DELAY, len(time_samples))

                # create longitudinal ego frenet trajectory for the time samples based on the spec
                if ego_ftrajectory is None:
                    ego_ftrajectory = SafetyUtils._calc_longitudinal_ego_trajectory(ego_init_fstate, spec, time_samples)

                back_safe_times = SafetyUtils._get_safe_time_samples(
                    ego_ftrajectory, obj_ftrajectory, (ego_length + cell.dynamic_object.size.length) / 2, time_delay)

        # calculate the range of safe indices based on LF and F
        unsafe_idxs = np.where(np.logical_not(forward_safe_times))[0]
        forward_safe_from = 0
        if len(unsafe_idxs) > 0:
            forward_safe_from = unsafe_idxs[-1] + 1
        unsafe_idxs = np.where(np.logical_not(front_safe_times))[0]
        front_safe_till = samples_num
        if len(unsafe_idxs) > 0:
            front_safe_till = unsafe_idxs[0]

        # find safe intervals based on the range and LB
        ext_safe_times = np.concatenate(([False], back_safe_times[forward_safe_from:front_safe_till], [False]))
        start_points = np.where(np.logical_and(ext_safe_times[1:], np.logical_not(ext_safe_times[:-1])))[0]
        end_points = np.where(np.logical_and(ext_safe_times[:-1], np.logical_not(ext_safe_times[1:])))[0]
        intervals = np.c_[time_samples[start_points], time_samples[end_points-1]]
        return intervals

    @staticmethod
    def _get_rel_lane_from_spec(road_id: int, ego_init_fstate: FrenetState2D, spec: ActionSpec) -> RelativeLane:
        lane_width = MapService.get_instance().get_road(road_id).lane_width
        if abs(spec.d - ego_init_fstate[FS_DX]) <= lane_width/2:
            return RelativeLane.SAME_LANE
        elif spec.d - ego_init_fstate[FS_DX] > lane_width/2:
            return RelativeLane.LEFT_LANE
        else:
            return RelativeLane.RIGHT_LANE

    @staticmethod
    def _calc_longitudinal_ego_trajectory(ego_init_fstate: FrenetState2D, spec: ActionSpec, time_samples: np.array) -> \
            FrenetTrajectory2D:
        """
        Calculate longitudinal ego trajectory for the given time samples.
        :param ego_init_fstate:
        :param spec:
        :param time_samples:
        :return:
        """
        # TODO: Acceleration is not calculated.

        dx = spec.s - ego_init_fstate[FS_SX]
        # profiles for the cases, when dynamic object is in front of ego
        dist_profile = QuinticPoly1D.distance_by_constraints(a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV],
                                                             v_T=spec.v, ds=dx, T=spec.t)(time_samples)
        vel_profile = QuinticPoly1D.velocity_by_constraints(a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV],
                                                            v_T=spec.v, ds=dx, T=spec.t)(time_samples)
        zeros = np.zeros(len(time_samples))
        ego_fstates = np.c_[ego_init_fstate[FS_SX] + dist_profile, vel_profile, zeros, zeros, zeros, zeros]
        return ego_fstates

    @staticmethod
    def calc_T_d(weights: np.array, ego_init_fstate: FrenetState2D, spec: ActionSpec) -> float:
        """
        Calculate lateral movement time for the given Jerk/T weights.
        :param weights: array of size 2: weights[0] is jerk weight, weights[1] is T weight
        :param ego_init_fstate: ego initial frenet state
        :param spec: action specification
        :return: lateral movement time
        """
        cost_coeffs_d = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=np.array([weights[1]]), w_J=np.array([weights[0]]), dx=np.array([spec.d - ego_init_fstate[FS_DX]]),
            a_0=np.array([ego_init_fstate[FS_DA]]), v_0=np.array([ego_init_fstate[FS_DV]]),
            v_T=np.array([0]), T_m=np.array([0]))
        roots_d = Math.find_real_roots_in_limits(cost_coeffs_d, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_d = np.fmin.reduce(roots_d, axis=-1)[0]
        return min(T_d, spec.t)

    @staticmethod
    def _get_safe_time_samples(front_traj: FrenetTrajectory2D, back_traj: FrenetTrajectory2D,
                               cars_length_margin: float, time_delay: np.array) -> np.array:
        """
        Calculate last time sample complying longitudinal safety for given front & back Frenet trajectories.
        :param front_traj: Frenet trajectory of the front vehicle
        :param back_traj: Frenet trajectory of the back vehicle
        :param cars_length_margin: half sum of the vehicles lengths
        :param time_delay: array of reaction delays of the back vehicle
        :return: array of booleans: which time samples are safe
        """
        safety_dist = SafetyUtils._get_safety_dist(front_traj[:, FS_SV], back_traj[:, FS_SV],
                                                   front_traj[:, FS_SX] - back_traj[:, FS_SX], time_delay,
                                                   cars_length_margin)
        return safety_dist > 0

    @staticmethod
    def _get_safety_dist_for_states(front: FrenetState2D, back: FrenetState2D, margin: float, td: float,
                                    max_brake: float = -LON_ACC_LIMITS[LIMIT_MIN]) -> float:
        """
        Calculate last time sample complying longitudinal safety for given front & back Frenet trajectories.
        :param front: Frenet state of the front vehicle
        :param back: Frenet state of the back vehicle
        :param margin: half sum of the vehicles lengths
        :param td: reaction delay of the back vehicle
        :return: safety distance
        """
        dist = front[FS_SX] - back[FS_SX]
        safe_dist = max(back[FS_SV]**2 - front[FS_SV]**2, 0.) / (2*max_brake) + back[FS_SV]*td + margin
        return dist - safe_dist

    @staticmethod
    def _get_safety_dist(v_front: np.array, v_back: np.array, dist: np.array, time_delay: np.array, margin: float,
                         max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate differences between the actual distances and minimal safe distances (longitudinal RSS formula)
        :param v_front: [m/s] array of front vehicle velocities
        :param v_back: [m/s] array of back vehicle velocities
        :param dist: [m] array of distances between the vehicles
        :param time_delay: [sec] array of time delays of the back vehicle
        :param margin: [m] cars size margin
        :param max_brake: [m/s^2] maximal deceleration of the vehicles
        :return: Array of safety distances: positive if the back vehicle is safe, negative otherwise
        """
        safe_dist = np.clip(v_back**2 - v_front**2, 0, None) / (2*max_brake) + v_back*time_delay + margin
        return dist - safe_dist
