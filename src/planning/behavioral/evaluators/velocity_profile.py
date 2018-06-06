from logging import Logger

import numpy as np
import copy


class VelocityProfile:
    def __init__(self, v_init: float, t_first: float, v_mid: float, t_flat: float, t_last: float, v_tar: float):
        self.v_init = v_init    # initial ego velocity
        self.t_first = t_first  # acceleration time period
        self.v_mid = v_mid      # constant velocity after acceleration
        self.t_flat = t_flat    # time period for going with maximal velocity
        self.t_last = t_last    # deceleration time
        self.v_tar = v_tar      # end velocity

    def calc_profile_details(self, T_max: float=np.inf) -> [np.array, np.array, np.array, np.array, np.array]:
        """
        Return times, longitudes, velocities, accelerations for the current velocity profile.
        A profile has at most 3 segments and at most 4 nodes between segments.
        :param T_max: if the profile is longer than T_max, then truncate it.
        :return: numpy arrays per segment or per node:
            times: time period of each segment
            cumulative times: cumulative times per node of the profile, with leading 0
            longitudes: cumulative distances per segment (except the last one), with leading 0
            velocities: velocities per node of the profile
            accelerations: accelerations per segment
        """
        T_max = max(0., T_max)
        truncated = self.cut_by_time(T_max)

        t = np.array([truncated.t_first, truncated.t_flat, truncated.t_last])
        t_cum = np.concatenate(([0], np.cumsum(t)))

        acc1 = acc3 = 0
        if truncated.t_first > 0:
            acc1 = (truncated.v_mid - truncated.v_init) / truncated.t_first
        if truncated.t_last > 0:
            acc3 = (truncated.v_tar - truncated.v_mid) / truncated.t_last
        a = np.array([acc1, 0, acc3])
        v = np.array([truncated.v_init, truncated.v_mid, truncated.v_mid, truncated.v_tar])
        lengths = np.array([0.5 * (truncated.v_init + truncated.v_mid) * truncated.t_first,
                            truncated.v_mid * truncated.t_flat,
                            0.5 * (truncated.v_mid + truncated.v_tar) * truncated.t_last])
        s_cum = np.concatenate(([0], np.cumsum(lengths)))
        return t, t_cum, s_cum, v, a

    def total_time(self) -> float:
        """
        total profile time
        :return: [s] total time
        """
        return self.t_first + self.t_flat + self.t_last

    def total_dist(self) -> float:
        """
        total profile distance
        :return: [m] total distance
        """
        return 0.5 * ((self.v_init + self.v_mid) * self.t_first + (self.v_mid + self.v_tar) * self.t_last) + \
               self.v_mid * self.t_flat

    def sample_at(self, t: float) -> [float, float]:
        """
        sample profile at time t and return elapsed distance and velocity at time t
        :param t: [s] time since profile beginning
        :return: elapsed distance and velocity at t
        """
        if t < self.t_first:
            acc = (self.v_mid - self.v_init) / self.t_first
            return t * (self.v_init + 0.5 * acc * t), self.v_init + acc * t
        s_acc = 0.5 * (self.v_init + self.v_mid) * self.t_first
        if t < self.t_first + self.t_flat:
            return s_acc + self.v_mid * (t - self.t_first), self.v_mid
        s_flat = self.v_mid * self.t_flat
        if t < self.total_time():
            dec = (self.v_tar - self.v_mid) / self.t_last
            t_dec = t - self.t_first - self.t_flat
            s_dec = t_dec * (self.v_mid + 0.5 * dec * t_dec)
            return s_acc + s_flat + s_dec, self.v_mid + dec * t_dec
        s_dec = 0.5 * (self.v_mid + self.v_tar) * self.t_last
        return s_acc + s_flat + s_dec, self.v_tar

    def cut_by_time(self, max_time: float):
        """
        cut profile in a given time
        :param max_time: [s] cutting time
        :return: a new cut profile
        """
        tot_time = self.total_time()
        if tot_time <= max_time:
            return copy.copy(self)
        if self.t_first + self.t_flat <= max_time:
            acc = (self.v_tar - self.v_mid) / self.t_last
            t_dec = max_time - self.t_first - self.t_flat
            v_tar = self.v_mid + acc * t_dec
            return VelocityProfile(self.v_init, self.t_first, self.v_mid, self.t_flat, t_dec, v_tar)
        if self.t_first <= max_time:
            t_flat = max_time - self.t_first
            return VelocityProfile(self.v_init, self.t_first, self.v_mid, t_flat, 0, self.v_mid)
        acc = (self.v_mid - self.v_init) / self.t_first
        t_acc = max_time
        v_tar = self.v_init + acc * t_acc
        return VelocityProfile(self.v_init, t_acc, v_tar, 0, 0, v_tar)

    @classmethod
    def calc_profile_given_acc(cls, v_init: float, a: float, dist: float, v_tar: float):
        """
        Given start & end velocities, distance to the followed car and acceleration, calculate velocity profile:
            1. acceleration to a velocity v_mid <= v_max for t1 time,
            2. moving by v_max for t2 time (t2 = 0 if v_mid < v_max),
            3. deceleration to end_vel for t3 time.
        If this profile is infeasible, then try an opposite order of accelerations: 1. deceleration, 3. acceleration.
        In the case of opposite order, the constant velocity segment is missing.
        In each velocity segment the acceleration is constant.
        :param v_init: start ego velocity
        :param a: absolute ego acceleration in the first and the last profile segments
        :param dist: initial distance to the safe location from the target
        :param v_tar: target object velocity
        return: VelocityProfile class or None in case of infeasible semantic action
        """
        # print('CALC PROFILE: v_init=%f dist=%f' % (v_init, dist))

        v_init_rel = v_init - v_tar  # relative velocity; may be negative

        if abs(v_init_rel) < 0.1 and abs(dist) < 0.1:
            return cls(v_init, 0, v_tar, 0, 0, v_tar)  # just follow the target car for min_time

        # first acceleration, then deceleration
        # here we use formula (vm^2 - v^2)/2(a-a_tar) + vm^2/2(a+a_tar) = dist
        v_mid_rel_sqr = v_init_rel * v_init_rel / 2 + dist * a
        if v_mid_rel_sqr >= 0:
            v_mid_rel = np.sqrt(v_mid_rel_sqr)  # should be positive
            t_acc = (v_mid_rel - v_init_rel) / a  # acceleration time
            t_dec = v_mid_rel / a                 # deceleration time
            if t_acc >= 0 and t_dec >= 0:  # negative time, try single segment with another acceleration
                return cls(v_init, t_acc, v_mid_rel + v_tar, 0, t_dec, v_tar)

        # try opposite order: first deceleration, then acceleration
        # here the formula (v^2 - vm^2)/2(a+a_tar) - vm^2/2(a-a_tar) = dist
        v_mid_rel_sqr = v_init_rel * v_init_rel / 2 - dist * a
        if v_mid_rel_sqr >= 0:
            v_mid_rel = -np.sqrt(v_mid_rel_sqr)  # should be negative
            t_acc = (v_init_rel - v_mid_rel) / a  # deceleration time
            t_dec = -v_mid_rel / a  # acceleration time
            if t_acc >= 0 and t_dec >= 0:  # negative time, try single segment with another acceleration
                return cls(v_init, t_acc, v_mid_rel + v_tar, 0, t_dec, v_tar)

        # if two segments failed, try a single segment with lower acceleration
        if v_init_rel * dist > 0:
            t_acc = 2 * dist / v_init_rel
            return cls(v_init, t_acc, v_tar, 0, 0, v_tar)

        # take single segment with another acceleration
        logger = Logger("VelocityProfile._calc_profile_given_acc")
        logger.warning("NO PROFILE: v_mid_rel_sqr=%f; v_init_rel=%f a=%f dist=%f" %
                            (v_mid_rel_sqr, v_init_rel, a, dist))
        return None  # illegal action

    @classmethod
    def calc_profile_given_T(cls, v_init: float, T: float, dist: float, v_tar: float):
        """
        Given start & end velocities, distance to the followed car and acceleration, calculate velocity profile:
            1. acceleration to a velocity v_mid for t1 time,
            2. deceleration to v_tar for t3 time or moving with constant velocity for t2.
        In each velocity segment the acceleration is constant.
        :param v_init: start ego velocity
        :param T: total time for the profile
        :param dist: initial distance to the safe location from the target
        :param v_tar: target object velocity
        return: VelocityProfile class or None in case of infeasible semantic action
        """
        logger = Logger("VelocityProfile.calc_profile_given_T")
        if T <= 0:
            logger.warning("NO PROFILE: T=%.2f" % T)
            return None
        # first try acceleration + constant vel
        LON_CALM_ACC = 1
        v_init_rel = v_init - v_tar  # relative velocity; may be negative
        if v_init_rel * dist > 0:
            t_acc = 2 * dist / v_init_rel
            acc = v_init_rel / t_acc
            if t_acc <= T and abs(acc) <= LON_CALM_ACC:
                return cls(v_init, t_acc, v_tar, T - t_acc, 0, v_tar)  # acceleration/deceleration + constant vel
        # let v = v_init_rel, v1 = v_mid_rel, t = t1, d = dist, solve for a (acceleration)
        # for the simple case (acceleration, deceleration) solve the following equations:
        # 2*v1^2 - v^2 = 2ad, v1 = v + at, v1 = a(T-t)
        # it is simplified to quadratic equation for a: T^2*a^2 - 2(2d-Tv)a - v^2 = 0
        # solution: a = ( (2d-Tv) +- sqrt((2d-Tv)^2 + (Tv)^2) ) / T^2
        Tv_2d = (2 * dist - T * v_init_rel) / (T ** 2)
        sqrt_disc = np.sqrt((2 * dist - T * v_init_rel) ** 2 + (T * v_init_rel) ** 2) / (T ** 2)
        # try acceleration + deceleration
        acc = Tv_2d + sqrt_disc  # always positive
        t_acc = 0.5 * (T - v_init_rel / acc)
        if 0. <= t_acc <= T:  # valid t1
            v_mid_rel = v_init_rel + acc * t_acc
            return cls(v_init, t_acc, v_mid_rel + v_tar, 0, T - t_acc, v_tar)  # acceleration, deceleration
        # try deceleration + acceleration
        acc = Tv_2d - sqrt_disc  # always negative
        t_acc = 0.5 * (T - v_init_rel / acc)
        if 0. <= t_acc <= T:  # valid t1
            v_mid_rel = v_init_rel + acc * t_acc
            return cls(v_init, t_acc, v_mid_rel + v_tar, 0, T - t_acc, v_tar)  # acceleration, deceleration

        logger.warning("NO PROFILE v_init_rel <= v_max_rel: t1=%.2f a=%.2f v_init=%.2f v_tar=%.2f T=%.2f" %
                       (t_acc, acc, v_init, v_tar, T))
        return None
