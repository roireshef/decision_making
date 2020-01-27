from typing import List, Optional
import numpy as np
from decision_making.src.global_constants import LAT_ACC_LIMITS, LAT_ACC_LIMITS_BY_K, LON_ACC_LIMITS, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS, BP_ACTION_T_LIMITS, TRAJECTORY_TIME_RESOLUTION, EPS
from decision_making.src.planning.types import FrenetState2D, FS_DX, FS_SX, FrenetState1D, FS_SV, FS_SA
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.math_utils import Math
from scipy.signal import argrelextrema


class CurveRoutePlanner:

    def __init__(self, points: np.array, velocity_limit: float):
        self.route = FrenetSerret2DFrame.fit(points)
        self.pointwise_vel_limits = np.minimum(np.sqrt(LAT_ACC_LIMITS_BY_K[1, 2] / self.route.k[:, 0]), velocity_limit)
        # lat_jerk = da/dt = (v^2*k)/dt = 2v*k + v^2*(dk/dt) = 2v*k + v^2*(dk/ds * ds/dt) = 2v*k + v^2*(k'*v) = 2v*k + v^3*k'
        self.vel_grid = np.flip(np.arange(0, np.max(self.pointwise_vel_limits)))

    def choose_actions_sequence(self, ego_state: FrenetState2D, target_s: float) -> List[np.array]:
        intervals = list(CurveRoutePlanner.calc_velocity_intervals(self.pointwise_vel_limits, self.route))
        w_J_arr = BP_JERK_S_JERK_D_TIME_WEIGHTS[:, 0]
        ds = self.route.ds

        sequences = []

        for w_J in w_J_arr:

            full_vts = np.empty((0, 3))
            source = ego_state[:FS_DX]
            for interval in intervals:
                if interval[1] < len(self.pointwise_vel_limits) - 1:
                    interval_target_s = interval[1] * ds
                    interval_target_v = self.vel_grid[np.argmax(self.vel_grid <= self.pointwise_vel_limits[interval[1]])]
                else:  # last interval
                    interval_target_s = target_s
                    interval_target_v = 0
                target = np.array([interval_target_s, interval_target_v, 0])

                v_T = self.vel_grid[self.vel_grid >= source[FS_SV]] if source[FS_SX] > ego_state[FS_SX] else self.vel_grid
                forward_vts = CurveRoutePlanner.create_accelerations_sequence(
                    ds, source, target, self.pointwise_vel_limits[interval[0], interval[1]],
                    v_T, LON_ACC_LIMITS, w_J)

                inv_v_T = self.vel_grid[self.vel_grid >= target[FS_SV]]
                inv_origin = np.array([self.route.s_max - target[FS_SX], target[FS_SV], -target[FS_SA]])
                inv_target = np.array([self.route.s_max - source[FS_SX], source[FS_SV], -source[FS_SA]])
                backward_vts = CurveRoutePlanner.create_accelerations_sequence(
                    ds, inv_origin, inv_target, self.pointwise_vel_limits[interval[0], interval[1]],
                    inv_v_T, -LON_ACC_LIMITS, w_J)

                interval_vts = CurveRoutePlanner.sew_forward_backward_vts(
                    source, target, forward_vts, backward_vts, target[FS_SX] - source[FS_SX], w_J)
                full_vts = np.concatenate((full_vts, interval_vts), axis=0)

                source = target

            if full_vts.shape[0] > 0:
                sequences.append(full_vts)

        return sequences

    @staticmethod
    def calc_velocity_intervals(vel_limits: np.array, route: FrenetSerret2DFrame) -> np.array:
        """
        points_of interest: first & last Frenet points + local minima of pointwise_vel_limits
        return intervals between the points of interest
        :param vel_limits:
        :param route:
        :return: pairs of indices of Frenet points
        """
        radius = 2 * np.max(vel_limits) / route.ds  # acceleration during 3 seconds
        poi = list(argrelextrema(vel_limits, np.less, order=radius)[0])
        if 0 not in poi:
            poi = [0] + poi
        if len(route.points) - 1 not in poi:
            poi = poi + [len(route.points) - 1]
        return np.c_[np.array(poi[:-1]), np.array(poi[1:]) + 1]

    @staticmethod
    def create_accelerations_sequence(ds: float, source: FrenetState1D, target: FrenetState1D,
                                      vel_limits: np.array, v_T: np.array,
                                      acc_limits: np.array, w_J: float) -> np.array:

        w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[0, 2]

        vts = []
        current_state = source
        while current_state[FS_SX] < target[FS_SX]:
            v_0, a_0 = current_state[[FS_SV, FS_SA]]

            s, T, poly_s = KinematicUtils.specify_quartic_actions(w_T, w_J, v_0, v_T, a_0, acc_limits=acc_limits)
            poly_vel = Math.polyder2d(poly_s, 1)
            times = np.arange(0, BP_ACTION_T_LIMITS[1], TRAJECTORY_TIME_RESOLUTION)
            TT = np.full((times.shape[0], T.shape[0]), T).T
            sampled_vel = Math.polyval2d(poly_vel, times)
            sampled_vel[times > TT] = -np.inf
            sampled_vel[~np.isfinite(T), :] = np.inf
            sampled_s = current_state[FS_SX] + Math.polyval2d(poly_s, times)
            indices_s = np.round(sampled_s / ds).astype(int)
            sampled_vel_limits = vel_limits[indices_s]
            valid = (sampled_vel <= sampled_vel_limits).all(axis=-1)
            fastest_vel_idx = np.argmax(valid)

            current_state[FS_SX] += s[fastest_vel_idx]
            current_state[FS_SV] = v_T[fastest_vel_idx]
            current_state[FS_SA] = 0

            vts.append(np.array([v_T[fastest_vel_idx], T[fastest_vel_idx]], s[fastest_vel_idx]))

        return np.array(vts)

    @staticmethod
    def sew_forward_backward_vts(init_state: FrenetState1D, end_state: FrenetState1D,
                                 forward_vts: np.array, backward_vts: np.array, total_s: float, w_J: float) -> np.array:
        forward_cum_s = np.concatenate(([0], np.cumsum(forward_vts[:, 2])))
        backward_cum_s = np.concatenate(([0], np.cumsum(backward_vts[:, 2])))
        forward_v = np.concatenate((init_state[FS_SV], forward_vts[:, 0]))
        backward_v = np.concatenate((end_state[FS_SV], backward_vts[:, 0]))
        fi = bi = 0
        dv = None
        while forward_cum_s[fi] + backward_cum_s[bi] < total_s:
            dv = int(forward_v[fi] < backward_v[bi])
            fi += dv
            bi += 1 - dv
        if dv is not None:
            fi -= dv
            bi -= 1 - dv

        sew_T, sew_s, fi, bi = CurveRoutePlanner.calc_sew_action_time(
            forward_v, backward_v, forward_cum_s, backward_cum_s, init_state[FS_SA], total_s, fi, bi, w_J)

        if sew_T is None:
            return np.empty((0, 3))

        reversed_backward = np.c_[backward_v[bi-1:-1:-1], np.flip(backward_vts[:bi, 1]), np.flip(backward_vts[:bi, 2])]
        sew_action = np.array([backward_v[bi], sew_T, sew_s])

        return np.concatenate((forward_vts[:fi], [sew_action], reversed_backward), axis=0)

    @staticmethod
    def calc_sew_action_time(forward_v, backward_v, forward_cum_s, backward_cum_s, init_a, total_s, fi, bi, w_J):
        if fi < 0 or bi < 0:
            return None, None, None, None

        v_0, v_T = forward_v[fi], backward_v[bi]
        a_0 = init_a * (fi == 0)
        a_T = 0
        sew_s = total_s - (forward_cum_s[fi] + backward_cum_s[bi])

        # calculate time sew_T for quartic action given all constraints including s
        if abs(a_0 - a_T) > EPS:
            sew_T = (-3*(v_0 + v_T) + np.sqrt(12*sew_s*(a_0 - a_T) + 9*(v_0 + v_T)**2)) / (a_0 - a_T)
        else:
            sew_T = 2*sew_s/(v_0 + v_T)

        # if the sew action is calm enough, then return it as is
        w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[0, 2]
        _, desired_T = KinematicUtils.specify_quartic_actions(w_T, w_J, v_0, v_T, a_0, acc_limits=LON_ACC_LIMITS)
        if desired_T <= sew_T:
            return sew_T, sew_s, fi, bi

        # otherwise remove an existing action (the faster between forward & backward) and call recursively the function
        return CurveRoutePlanner.calc_sew_action_time(forward_v, backward_v, forward_cum_s, backward_cum_s,
                                                      init_a, total_s, fi - (v_0 >= v_T), bi - (v_0 < v_T), w_J)
