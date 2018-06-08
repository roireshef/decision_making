from logging import Logger
import numpy as np
import math

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS, BP_ACTION_T_LIMITS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.types import LIMIT_MIN, FrenetTrajectory2D, FS_SV, FS_SX, FrenetState2D, FS_SA, \
    FS_DX, FS_DV, LIMIT_MAX, FS_DA
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D
from mapping.src.service.map_service import MapService


class SafetyUtils:

    @staticmethod
    def is_safe_spec(behavioral_state: BehavioralGridState, ego_init_fstate: FrenetState2D, spec: ActionSpec):
        """
        Check safety for the spec w.r.t. F, and LF/RF, LB/RB in case of lane change.
        The safety w.r.t. the followed object is calculated for the whole ego trajectory (used for static actions).
        In case of lane change the safety w.r.t. F, LB/RB is calculated only for the most aggressive lateral movement.
        :param behavioral_state: current behavioral state
        :param ego_init_fstate: initial ego frenet state
        :param spec: action specification
        :return: True for safe action, False for unsafe action (see the description above)
        """
        ego = behavioral_state.ego_state
        rel_lane = SafetyUtils._get_rel_lane_from_spec(ego.road_localization.road_id, ego_init_fstate, spec)
        forward_cell = (rel_lane, RelativeLongitudinalPosition.FRONT)

        # create time samples for safety checking
        sampling_step = spec.t / np.round(spec.t)  # approximately every second
        time_samples = np.arange(0, spec.t + np.finfo(np.float16).eps, sampling_step)
        zeros = np.zeros(len(time_samples))
        ego_ftrajectory = None

        # check safety w.r.t. the followed object on the target lane (if exists)
        if forward_cell in behavioral_state.road_occupancy_grid:
            cell = behavioral_state.road_occupancy_grid[forward_cell][0]
            obj_ftrajectory = np.c_[cell.fstate[FS_SX] + cell.fstate[FS_SV] * time_samples, cell.fstate[FS_SV] + zeros,
                                    zeros, zeros, zeros, zeros]
            time_delay = np.repeat(SAFETY_MARGIN_TIME_DELAY, len(time_samples))

            # create longitudinal ego frenet trajectory for the time samples based on the spec
            if ego_ftrajectory is None:
                ego_ftrajectory = SafetyUtils._calc_longitudinal_ego_trajectory(ego_init_fstate, spec, time_samples)

            last_safe_time = SafetyUtils._get_last_safe_time_for_trajectories(
                obj_ftrajectory, ego_ftrajectory, (ego.size.length + cell.dynamic_object.size.length)/2, time_delay, time_samples)
            if last_safe_time < spec.t:
                return False

        # for lane change actions check safety w.r.t. F, LB, RB
        if rel_lane != RelativeLane.SAME_LANE:
            # filter action if there is a side car
            if (rel_lane, RelativeLongitudinalPosition.PARALLEL) in behavioral_state.road_occupancy_grid:
                return False
            aggressive_weights = np.array([0.016, 1])  # very aggressive lateral movement
            T_d_min = SafetyUtils._calc_T_d(aggressive_weights, ego_init_fstate, spec)

            # Pick lateral_time_samples, such that lateral_time_samples is a subset of time_samples,
            # since if ego_ftrajectory was calculated based on original time_samples, then lateral_time_samples
            # should be a subset of time_samples, so there is no need to recalculate ego_ftrajectory.
            lat_samples_num = len(np.where(time_samples < T_d_min + 0.5)[0])
            lateral_time_samples = time_samples[:lat_samples_num]
            if ego_ftrajectory is not None:
                ego_ftrajectory = ego_ftrajectory[:lat_samples_num]

            last_safe_time = SafetyUtils._calc_last_safe_time_for_time_samples(
                behavioral_state, ego_init_fstate, ego_ftrajectory, spec, lateral_time_samples, rel_lane)
            if last_safe_time < T_d_min:
                return False

        return True

    @staticmethod
    def calc_lane_change_last_safe_time(behavioral_state: BehavioralGridState, ego_init_fstate: FrenetState2D,
                                        spec: ActionSpec) -> float:
        """
        Calculate the last safe time for a lane change action
        :param behavioral_state:
        :param ego_init_fstate:
        :param spec:
        :return: the last safe time
        """
        rel_lane = SafetyUtils._get_rel_lane_from_spec(behavioral_state.ego_state.road_localization.road_id,
                                                       ego_init_fstate, spec)
        if rel_lane == RelativeLane.SAME_LANE:  # for lane change actions check safety w.r.t. F, LB, RB
            return np.inf

        front_cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)
        side_rear_cell = (rel_lane, RelativeLongitudinalPosition.REAR)
        if front_cell in behavioral_state.road_occupancy_grid or side_rear_cell in behavioral_state.road_occupancy_grid:
            # calculate calm T_d
            calm_weights = np.array([1.5, 1])  # calm lateral movement
            T_d = SafetyUtils._calc_T_d(calm_weights, ego_init_fstate, spec)
            sampling_step = 0.1
            time_samples = np.arange(0, T_d, sampling_step)

            return SafetyUtils._calc_last_safe_time_for_time_samples(
                behavioral_state, ego_init_fstate, None, spec, time_samples, rel_lane)
        return np.inf

    @staticmethod
    def _calc_last_safe_time_for_time_samples(behavioral_state: BehavioralGridState, ego_init_fstate: FrenetState2D,
                                              ego_ftrajectory: FrenetTrajectory2D, spec: ActionSpec,
                                              time_samples: np.array, rel_lane: RelativeLane) -> float:
        """
        Calculate last safe time w.r.t. F, LB, RB, for the given time samples.
        :param behavioral_state:
        :param ego_init_fstate:
        :param ego_ftrajectory: may be None, is calculated on demand
        :param spec:
        :param time_samples: starting from 0
        :param rel_lane:
        :return: last safe time w.r.t. F, LB, RB; np.inf if always safe; -1 if unsafe from the beginning
        """
        if rel_lane == RelativeLane.SAME_LANE:
            return np.inf
        ego = behavioral_state.ego_state
        ego_length = ego.size.length
        lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width
        front_cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)
        side_rear_cell = (rel_lane, RelativeLongitudinalPosition.REAR)
        last_safe_time = np.inf
        zeros = np.zeros(len(time_samples))

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

            # create longitudinal ego frenet trajectory for the time samples based on the spec
            if ego_ftrajectory is None:
                ego_ftrajectory = SafetyUtils._calc_longitudinal_ego_trajectory(ego_init_fstate, spec, time_samples)

            safe_time = SafetyUtils._get_last_safe_time_for_trajectories(
                obj_ftrajectory, ego_ftrajectory, (ego_length + cell.dynamic_object.size.length)/2, time_delay, time_samples)
            last_safe_time = min(safe_time, last_safe_time)
            if last_safe_time < 0:
                return -1

        # check safety w.r.t. the back object LB / RB on the target lane (if exists)
        if side_rear_cell in behavioral_state.road_occupancy_grid:
            cell = behavioral_state.road_occupancy_grid[side_rear_cell][0]
            obj_ftrajectory = np.c_[cell.fstate[FS_SX] + cell.fstate[FS_SV] * time_samples,
                                    cell.fstate[FS_SV] + zeros, zeros, zeros, zeros, zeros]
            time_delay = np.repeat(SPECIFICATION_MARGIN_TIME_DELAY, len(time_samples))

            # create longitudinal ego frenet trajectory for the time samples based on the spec
            if ego_ftrajectory is None:
                ego_ftrajectory = SafetyUtils._calc_longitudinal_ego_trajectory(ego_init_fstate, spec, time_samples)

            safe_time = SafetyUtils._get_last_safe_time_for_trajectories(
                ego_ftrajectory, obj_ftrajectory, (ego_length + cell.dynamic_object.size.length)/2, time_delay, time_samples)
            last_safe_time = min(safe_time, last_safe_time)

        return last_safe_time

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
    def _calc_T_d(weights: np.array, ego_init_fstate: FrenetState2D, spec: ActionSpec) -> float:
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
    def _get_last_safe_time_for_trajectories(front_traj: FrenetTrajectory2D, back_traj: FrenetTrajectory2D,
                                             cars_length_margin: float, time_delay: np.array,
                                             time_samples: np.array) -> float:
        """
        Calculate last time sample complying longitudinal safety for given front & back Frenet trajectories.
        :param front_traj: Frenet trajectory of the front vehicle
        :param back_traj: Frenet trajectory of the back vehicle
        :param cars_length_margin: half sum of the vehicles lengths
        :param time_delay: array of reaction delays of the back vehicle
        :param time_samples: time samples, for which the safety is checked
        :return: last safe time sample from the given time_samples array
        """
        safety_dist = SafetyUtils._get_safety_dist(front_traj[:, FS_SV], back_traj[:, FS_SV],
                                                   front_traj[:, FS_SX] - back_traj[:, FS_SX], time_delay,
                                                   cars_length_margin)
        last_safe_time = np.inf
        unsafe_idxs = np.where(safety_dist <= 0)[0]
        if len(unsafe_idxs) > 0:
            last_safe_time = -1
            if unsafe_idxs[0] > 0:
                last_safe_time = time_samples[unsafe_idxs[0] - 1]
        return last_safe_time

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
