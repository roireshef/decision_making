import numpy as np
import copy

from decision_making.src.global_constants import AGGRESSIVENESS_TO_LON_ACC, LON_ACC_LIMITS, AGGRESSIVENESS_TO_LAT_ACC
from decision_making.src.planning.behavioral.data_objects import ActionType, AggressivenessLevel
from decision_making.src.planning.types import LIMIT_MIN
from decision_making.src.state.state import DynamicObject


class VelocityProfile:
    def __init__(self, v_init: float, t1: float, v_mid: float, t2: float, t3: float, v_tar: float):
        self.v_init = v_init    # initial ego velocity
        self.t1 = t1            # acceleration/deceleration time period
        self.v_mid = v_mid      # maximal velocity after acceleration
        self.t2 = t2            # time period for going with maximal velocity
        self.t3 = t3            # deceleration/acceleration time
        self.v_tar = v_tar      # end velocity

    def get_details(self, T_max: float=np.inf) -> [np.array, np.array, np.array, np.array, np.array]:
        """
        Return times, longitudes, velocities, accelerations for the current velocity profile.
        :param T_max: if the profile is longer than max_time, then truncate it
        :return: numpy arrays per segment:
            times: time period of each segment
            cumulative times: cumulative times of the segments, with leading 0
            longitudes: cumulated distances per segment (except the last one), with leading 0
            velocities: velocities per segment
            accelerations: accelerations per segment
        All arrays' size is equal to the (truncated) segments number, except t_cum having extra 0 at the beginning.
        """
        t = np.array([self.t1, self.t2, self.t3])
        t_cum = np.concatenate(([0], np.cumsum(t)))
        T_max = max(0., T_max)

        acc1 = acc3 = 0
        if self.t1 > 0:
            acc1 = (self.v_mid - self.v_init) / self.t1
        if self.t3 > 0:
            acc3 = (self.v_tar - self.v_mid) / self.t3
        a = np.array([acc1, 0, acc3])
        v = np.array([self.v_init, self.v_mid, self.v_mid])
        lengths = np.array([0.5 * (self.v_init + self.v_mid) * self.t1, self.v_mid * self.t2])  # without the last segment

        if t_cum[-1] > T_max:  # then truncate all arrays by max_time
            truncated_size = np.where(t_cum[:-1] < T_max)[0][-1] + 1
            t = t[:truncated_size]  # truncate times array
            t[-1] -= t_cum[truncated_size] - T_max  # decrease the last segment time
            t_cum = np.concatenate(([0], np.cumsum(t)))
            a = a[:truncated_size]  # truncate accelerations array
            v = v[:truncated_size]  # truncate velocities array
            lengths = lengths[:(truncated_size-1)]  # truncate distances array

        s_cum = np.concatenate(([0], np.cumsum(lengths)))

        return t, t_cum, s_cum, v, a

    def total_time(self) -> float:
        return self.t1 + self.t2 + self.t3

    def total_dist(self) -> float:
        return 0.5 * ((self.v_init + self.v_mid) * self.t1 + (self.v_mid + self.v_tar) * self.t3) + self.v_mid * self.t2

    def sample_at(self, t: float) -> [float, float]:
        if t < self.t1:
            a1 = (self.v_mid - self.v_init) / self.t1
            return t * (self.v_init + 0.5 * a1 * t), self.v_init + a1 * t
        d1 = 0.5 * (self.v_init + self.v_mid) * self.t1
        if t < self.t1 + self.t2:
            return d1 + self.v_mid * (t - self.t1), self.v_mid
        d2 = self.v_mid * self.t2
        if t < self.total_time():
            a3 = (self.v_tar - self.v_mid) / self.t3
            t3 = t - self.t1 - self.t2
            return d1 + d2 + t3 * (self.v_mid + 0.5 * a3 * t3), self.v_mid + a3 * t3
        return d1 + d2 + 0.5 * (self.v_mid + self.v_tar) * self.t3, self.v_tar

    def when_decel_velocity_is_equal_to(self, v: float) -> float:
        """
        Search only during deceleration
        :param v: velocity to search
        :return: time during deceleration when the velocity is equal to v
        """
        if v > self.v_mid or v < self.v_tar:
            return None
        if v == self.v_mid:
            return self.t1 + self.t2
        t = None
        if v >= self.v_tar:  # v_tar <= v < v_mid, then self.t3 > 0
            a3 = (self.v_tar - self.v_mid) / self.t3
            t = (v - self.v_mid) / a3
        return self.t1 + self.t2 + t

    def when_accel_velocity_is_equal_to(self, v: float) -> float:
        """
        Search only during acceleration
        :param v: velocity to search
        :return: time during acceleration when the velocity is equal to v
        """
        if v > self.v_mid or v < self.v_init:
            return None
        if v == self.v_mid:
            return self.t1
        t = None
        if v >= self.v_init:  # v_init <= v < v_mid, then self.t1 > 0
            a1 = (self.v_mid - self.v_init) / self.t1
            t = (v - self.v_init) / a1
        return t

    def cut_by_time(self, max_time: float):
        tot_time = self.total_time()
        if tot_time <= max_time:
            return copy.copy(self)
        if self.t1 + self.t2 <= max_time:
            a = (self.v_tar - self.v_mid) / self.t3
            t3 = max_time - self.t1 - self.t2
            v_tar = self.v_mid + a * t3
            return VelocityProfile(self.v_init, self.t1, self.v_mid, self.t2, t3, v_tar)
        if self.t1 <= max_time:
            t2 = max_time - self.t1
            return VelocityProfile(self.v_init, self.t1, self.v_mid, t2, 0, self.v_mid)
        a = (self.v_mid - self.v_init) / self.t1
        t1 = max_time
        v_tar = self.v_init + a * t1
        return VelocityProfile(self.v_init, t1, v_tar, 0, 0, v_tar)

    @classmethod
    def _calc_profile_given_acc(cls, v_init: float, a: float, dist: float, v_tar: float):
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
            t1 = (v_mid_rel - v_init_rel) / a  # acceleration time
            t3 = v_mid_rel / a                 # deceleration time
            if t1 >= 0 and t3 >= 0:  # negative time, try single segment with another acceleration
                return cls(v_init, t1, v_mid_rel + v_tar, 0, t3, v_tar)

        # try opposite order: first deceleration, then acceleration
        # here the formula (v^2 - vm^2)/2(a+a_tar) - vm^2/2(a-a_tar) = dist
        v_mid_rel_sqr = v_init_rel * v_init_rel / 2 - dist * a
        if v_mid_rel_sqr >= 0:
            v_mid_rel = -np.sqrt(v_mid_rel_sqr)  # should be negative
            t1 = (v_init_rel - v_mid_rel) / a  # deceleration time
            t3 = -v_mid_rel / a  # acceleration time
            if t1 >= 0 and t3 >= 0:  # negative time, try single segment with another acceleration
                return cls(v_init, t1, v_mid_rel + v_tar, 0, t3, v_tar)

        # if two segments failed, try a single segment with lower acceleration
        if v_init_rel * dist > 0:
            t1 = 2 * dist / v_init_rel
            return cls(v_init, t1, v_tar, 0, 0, v_tar)

        # take single segment with another acceleration
        print('NO PROFILE: v_mid_rel_sqr=%f; v_init_rel=%f a=%f dist=%f' % (v_mid_rel_sqr, v_init_rel, a, dist))
        return None  # illegal action

    @staticmethod
    def calc_lateral_time(init_lat_vel: float, signed_lat_dist: float, lane_width: float,
                          aggressiveness_level: AggressivenessLevel) -> [float, float, float]:
        """
        Given initial lateral velocity and signed lateral distance, estimate a time it takes to perform the movement.
        The time estimation assumes movement by velocity profile like in the longitudinal case.
        :param init_lat_vel: [m/s] initial lateral velocity
        :param signed_lat_dist: [m] signed distance to the target
        :param lane_width: [m] lane width
        :param aggressiveness_level: aggressiveness_level
        :return: [s] the lateral movement time to the target, [m] maximal lateral deviation from lane center,
        [m/s] initial lateral velocity toward target (negative if opposite to the target direction)
        """
        if signed_lat_dist > 0:
            lat_v_init_toward_target = init_lat_vel
        else:
            lat_v_init_toward_target = -init_lat_vel
        # normalize lat_acc by lane_width, such that T_d will NOT depend on lane_width
        acc = AGGRESSIVENESS_TO_LAT_ACC[aggressiveness_level.value]
        lat_acc = acc * lane_width / 3.6
        lateral_profile = VelocityProfile._calc_profile_given_acc(lat_v_init_toward_target, lat_acc, abs(signed_lat_dist), 0)

        return lateral_profile.t1 + lateral_profile.t3

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
        if T <= 0:
            print('NO PROFILE: T=%.2f' % T)
            return None
        # first try acceleration + constant vel
        v_init_rel = v_init - v_tar  # relative velocity; may be negative
        if v_init_rel * dist > 0:
            t1 = 2 * dist / v_init_rel
            a = v_init_rel / t1
            if t1 <= T and abs(a) <= AGGRESSIVENESS_TO_LON_ACC[AggressivenessLevel.CALM.value]:
                return cls(v_init, t1, v_tar, T - t1, 0, v_tar)  # acceleration/deceleration + constant vel
        # let v = v_init_rel, v1 = v_mid_rel, t = t1, d = dist, solve for a (acceleration)
        # for the simple case (acceleration, deceleration) solve the following equations:
        # v1^2 - v^2 = 2ad, v1 = v + at, v1 = a(T-t)
        # it is simplified to quadratic equation for a: T^2*a^2 - 2(2d-Tv)a - v^2 = 0
        # solution: a = ( (2d-Tv) +- sqrt((2d-Tv)^2 + (Tv)^2) ) / T^2
        Tv_2d = (2 * dist - T * v_init_rel) / (T ** 2)
        sqrt_disc = np.sqrt(Tv_2d ** 2 + (T * v_init_rel) ** 2) / (T ** 2)
        # try acceleration + deceleration
        a = Tv_2d + sqrt_disc  # always positive
        t1 = 0.5 * (T - v_init_rel / a)
        if 0. <= t1 <= T:  # valid t1
            v_mid_rel = v_init_rel + a * t1
            return cls(v_init, t1, v_mid_rel + v_tar, 0, T - t1, v_tar)  # acceleration, deceleration
        # try deceleration + acceleration
        a = Tv_2d - sqrt_disc  # always negative
        t1 = 0.5 * (T - v_init_rel / a)
        if 0. <= t1 <= T:  # valid t1
            v_mid_rel = v_init_rel + a * t1
            return cls(v_init, t1, v_mid_rel + v_tar, 0, T - t1, v_tar)  # acceleration, deceleration
        print('NO PROFILE v_init_rel <= v_max_rel: t1=%.2f a=%.2f v_init=%.2f v_tar=%.2f T=%.2f' % (t1, a, v_init, v_tar, T))
        return None

    def calc_last_safe_time(self, ego_lon: float, ego_half_size: float, dyn_obj: DynamicObject, T: float,
                            td_0: float, td_T: float) -> float:
        """
        Given ego velocity profile and dynamic object, calculate the last time, when the safety complies.
        :param ego_lon: ego initial longitude
        :param ego_half_size: [m] half length of ego
        :param dyn_obj: the dynamic object, for which the safety is tested
        :param T: maximal time to check the safety (usually T_d for safety w.r.t. F and LB)
        :param td_0: reaction time of the back car at time 0
        :param td_T: reaction time of the back car at time T_max
        :return: the latest safe time
        """
        if T <= 0:
            return np.inf

        margin = ego_half_size + dyn_obj.size.length / 2
        # initialization of motion parameters
        (init_s_ego, init_v_obj, a_obj) = (ego_lon, dyn_obj.v_x, dyn_obj.acceleration_lon)
        init_s_obj = dyn_obj.road_localization.road_lon

        if abs(a_obj) < 0.01:  # the object has constant velocity, then use fast safety test
            dist = abs(init_s_obj - init_s_ego)
            if init_s_ego < init_s_obj:
                v_front = init_v_obj
                v_back = self.v_init
            else:
                v_back = init_v_obj
                v_front = self.v_init
            if not VelocityProfile.is_safe_state(v_front, v_back, dist, td_0, margin):
                return -1
            if self._is_safe_profile(ego_lon, ego_half_size, dyn_obj, T, td_0, td_T):
                return np.inf

        t, t_cum, s_ego, v_ego, a_ego = self.get_details(T)
        s_ego += init_s_ego
        s_obj = init_s_obj + init_v_obj * t_cum[:-1] + 0.5 * a_obj * t_cum[:-1] * t_cum[:-1]
        v_obj = init_v_obj + a_obj * t_cum[:-1]

        # create (ego, obj) pairs of longitudes, velocities and accelerations for all segments
        (s, v, a) = (np.c_[s_ego, s_obj], np.c_[v_ego, v_obj], np.c_[a_ego, np.repeat(a_obj, t.shape[0])])

        front = int(init_s_ego < init_s_obj)  # 0 if the object is behind ego; 1 otherwise
        back = 1 - front

        # calculate last_safe_time
        last_safe_time = 0
        a_max = -LON_ACC_LIMITS[LIMIT_MIN]

        for seg in range(t.shape[0]):
            if t[seg] == 0:
                continue
            td = td_0 + (td_T - td_0) * t_cum[seg] / T
            safe_time = VelocityProfile._calc_largest_time_for_segment(
                s[seg, front], v[seg, front], a[seg, front], s[seg, back], v[seg, back], a[seg, back], t[seg],
                margin, td)
            if safe_time < 0:
                return last_safe_time
            last_safe_time += safe_time
            if last_safe_time < t_cum[seg + 1]:  # becomes unsafe inside this segment
                # check if delayed last_safe_time (t+td) overflowed to the next segment
                T = last_safe_time + td
                if T > t_cum[seg + 1]:  # then check safety on delayed point of vel_segment
                    # suppose the object moves last_safe_time, then fully brakes during time_delay
                    # ego moves according to vel_profile during T (then fully brakes)
                    braking_time = min(td, init_v_obj / a_max)
                    if init_s_ego < init_s_obj:
                        s_back, v_back = self.sample_at(T)
                        s_back += init_s_ego
                        s_front = init_s_obj + init_v_obj * T + 0.5 * a_obj * last_safe_time * last_safe_time - \
                                  0.5 * a_max * braking_time * braking_time
                        v_front = max(0., init_v_obj + a_obj * last_safe_time - a_max * td)
                    else:
                        s_front, v_front = self.sample_at(T)
                        s_front += init_s_ego
                        s_back = init_s_obj + init_v_obj * T + 0.5 * a_obj * last_safe_time * last_safe_time - \
                                 0.5 * a_max * braking_time * braking_time
                        v_back = max(0., init_v_obj + a_obj * last_safe_time - a_max * td)
                    if max(0., v_back**2 - v_front**2)/(2*a_max) + margin <= s_front - s_back:
                        last_safe_time = t_cum[seg + 1]
                        continue  # this segment is safe
                return last_safe_time
        return np.inf  # always safe

    @staticmethod
    def is_safe_state(v_front: float, v_back: float, dist: float, time_delay: float, margin: float,
                      max_brake: float=-LON_ACC_LIMITS[0]):
        return max(0., v_back**2 - v_front**2) / (2*max_brake) + v_back*time_delay + margin < dist

    def _is_safe_profile(self, ego_lon: float, ego_half_size: float, dyn_obj: DynamicObject, T: float,
                         td_0: float, td_T: float) -> bool:
        """
        Given ego velocity profile and dynamic object, check if the profile complies safety.
        Here we assume the object has a constant velocity.
        :param ego_lon: ego initial longitude
        :param ego_half_size: [m] half length of ego
        :param dyn_obj: the dynamic object, for which the safety is tested
        :param T: maximal time to check the safety (usually T_d for safety w.r.t. F and LB)
        :param td_0: reaction time of the back car at time 0
        :param td_T: reaction time of the back car at time T_max
        :return: the latest safe time
        """
        if T <= 0:
            return True
        (init_s_ego, init_s_obj, v_obj) = (ego_lon, dyn_obj.road_localization.road_lon, dyn_obj.v_x)
        margin = ego_half_size + dyn_obj.size.length / 2
        cut_profile = self.cut_by_time(T)

        if init_s_ego < init_s_obj:  # the object is in front of ego
            t = cut_profile.when_decel_velocity_is_equal_to(v_obj)
            if t is None:  # cut_profile does not contain velocity v_obj
                v_max = max(max(cut_profile.v_init, cut_profile.v_mid), cut_profile.v_tar)
                if v_max < v_obj:  # ego is always slower
                    return VelocityProfile.is_safe_state(v_obj, cut_profile.v_init, init_s_obj - init_s_ego, td_0, margin)
                else:  # ego is always faster, check only at the end
                    tot_time = cut_profile.total_time()
                    tot_dist = cut_profile.total_dist()
                    return VelocityProfile.is_safe_state(v_obj, cut_profile.v_tar,
                                                         (init_s_obj + v_obj*tot_time) - (init_s_ego + tot_dist),
                                                         td_T, margin)
            else:  # t was found
                s, _ = cut_profile.sample_at(t)  # v_ego(t2) = v_obj
                td = td_0 + (td_T - td_0) * t / T
                return init_s_obj + v_obj*t - (init_s_ego + s) > v_obj * td + margin
        else:  # the object is behind ego
            t = cut_profile.when_accel_velocity_is_equal_to(v_obj)
            if t is None:  # cut_profile does not contain velocity v_obj
                v_min = min(min(cut_profile.v_init, cut_profile.v_mid), cut_profile.v_tar)
                if v_min > v_obj:  # ego is always faster
                    return VelocityProfile.is_safe_state(cut_profile.v_init, v_obj, init_s_ego - init_s_obj, td_0, margin)
                else:  # ego is always slower, check only at the end
                    tot_time = cut_profile.total_time()
                    tot_dist = cut_profile.total_dist()
                    return VelocityProfile.is_safe_state(cut_profile.v_tar, v_obj,
                                                         (init_s_ego + tot_dist) - (init_s_obj + v_obj*tot_time),
                                                         td_T, margin)
            else:  # t was found
                s, _ = cut_profile.sample_at(t)  # v_ego(t2) = v_obj
                td = td_0 + (td_T - td_0) * t / T
                return (init_s_ego + s) - (init_s_obj + v_obj*t) > v_obj * td + margin

    @staticmethod
    def _calc_largest_time_for_segment(s1: float, v1: float, a1: float, s2: float, v2: float, a2: float,
                                       T: float, margin: float, td: float) -> float:
        """
        Given two vehicles with constant acceleration in time period [0, T], calculate the largest 0 <= t <= T,
        for which the second car (rear) is safe w.r.t. the first car in [0, t].
        :param s1: first car longitude
        :param v1: first car initial velocity
        :param a1: first car acceleration
        :param s2: second car longitude
        :param v2: second car initial velocity
        :param a2: second car acceleration
        :param T: time period
        :param margin: size margin of the cars
        :param td: back car's time delay i.e. reaction time of the back car (for AV td is smaller)
        :return: the largest safe time; if unsafe for t=0, return -1
        """
        if T < 0:
            return -1
        # the first vehicle is in front of the second one
        # s1(t) = s1 + v1*t + a1*t^2/2, s2(t) = s2 + v2*t + a2*t^2/2
        # v1(t) = v1 + a1*t, v2(t) = v2 + a2*t
        # td = time_delay
        # safe_dist(t) = (v2(t+td)^2 - v1(t)^2) / (2*a_max) + margin
        # solve quadratic inequality: s1(t) - s2(t+td) - safe_dist(t) = A*t^2 + B*t + C >= 0

        a_max = -LON_ACC_LIMITS[LIMIT_MIN]

        C = s1 - s2 + (v1*v1 - v2*v2)/(2*a_max) - (0.5 * a2 * td + v2) * td * (1 + a2 / a_max) - margin

        if C < 0:
            return -1  # the current state (for t=0) is not safe
        if T == 0:
            return 0  # the current state (for t=0) is safe

        A = (a1-a2) * (a_max+a1+a2) / (2*a_max)
        B = (a1*v1 - a2*v2)/a_max + v1 - v2 - a2 * td * (1 + a2 / a_max)
        if A == 0 and B == 0:  # constant function C > 0
            return T  # for all t it's safe
        if A == 0:  # B != 0; linear inequality
            t = -C/B
            if t >= 0:
                return min(T, t)
            return T  # for all t it's safe

        # solve quadratic inequality
        discriminant = B*B - 4*A*C
        if discriminant < 0:
            return T  # for all t it's safe
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-B - sqrt_disc)/(2*A)
        t2 = (-B + sqrt_disc)/(2*A)
        if t1 >= 0:
            return min(t1, T)
        if t2 >= 0:
            return min(t2, T)
        return T

    @staticmethod
    def calc_collision_time(v_init: float, v_max: float, acc: float, v_tar: float, dist: float) -> float:
        v_init_rel = v_init - v_tar
        v_max_rel = v_max - v_tar
        if v_max_rel <= 0 and v_init_rel <= 0:
            return np.inf
        if v_init_rel < v_max_rel:
            acceleration_dist = (v_max_rel**2 - v_init_rel**2) / (2*acc)
            if acceleration_dist < dist:
                acceleration_time = (v_max_rel - v_init_rel) / acc
                const_vel_time = (dist - acceleration_dist) / v_max_rel
                return acceleration_time + const_vel_time
            else:  # acceleration_dist >= dist; solve for t: v*t + at^2/2 = dist
                acceleration_time = (np.sqrt(v_init_rel**2 + 2*acc*dist) - v_init_rel) / acc
                return acceleration_time
        else:  # v_init_rel > v_max_rel
            return dist / v_init_rel
