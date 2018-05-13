import numpy as np
import copy

from decision_making.src.global_constants import AGGRESSIVENESS_TO_LON_ACC, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, \
    SAFE_DIST_TIME_DELAY, LON_ACC_LIMITS, AGGRESSIVENESS_TO_LAT_ACC, BP_MAX_VELOCITY_TOLERANCE
from decision_making.src.planning.behavioral.data_objects import ActionType, AggressivenessLevel
from decision_making.src.planning.types import FP_SX, LIMIT_MIN, FS_SV, FS_SA
from decision_making.src.state.state import DynamicObject


class VelocityProfile:
    def __init__(self, v_init: float, t1: float, v_mid: float, t2: float, t3: float, v_tar: float):
        self.v_init = v_init    # initial ego velocity
        self.t1 = t1            # acceleration/deceleration time period
        self.v_mid = v_mid      # maximal velocity after acceleration
        self.t2 = t2            # time period for going with maximal velocity
        self.t3 = t3            # deceleration/acceleration time
        self.v_tar = v_tar      # end velocity

    def get_details(self, max_time: float=np.inf) -> [np.array, np.array, np.array, np.array, np.array]:
        """
        Return times, longitudes, velocities, accelerations for the current velocity profile.
        :param max_time: if the profile is longer than max_time, then truncate it
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
        max_time = max(0., max_time)

        acc1 = acc3 = 0
        if self.t1 > 0:
            acc1 = (self.v_mid - self.v_init) / self.t1
        if self.t3 > 0:
            acc3 = (self.v_tar - self.v_mid) / self.t3
        a = np.array([acc1, 0, acc3])
        v = np.array([self.v_init, self.v_mid, self.v_mid])
        lengths = np.array([0.5 * (self.v_init + self.v_mid) * self.t1, self.v_mid * self.t2])  # without the last segment

        if t_cum[-1] > max_time:  # then truncate all arrays by max_time
            truncated_size = np.where(t_cum[:-1] < max_time)[0][-1] + 1
            t = t[:truncated_size]  # truncate times array
            t[-1] -= t_cum[truncated_size] - max_time  # decrease the last segment time
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
    def calc_velocity_profile_given_acc(cls, action_type: ActionType, lon_init: float, v_init: float, lon_target: float,
                                        v_target: float, a_target: float, aggressiveness_level: AggressivenessLevel,
                                        cars_size_margin: float, min_time: float):
        """
        calculate velocities profile for semantic action: either following car or following lane
        :param action_type: [ActionType] type of action
        :param lon_init: [m] initial longitude of ego
        :param v_init: [m/s] initial velocity of ego
        :param lon_target: [m] initial longitude of followed object (None if follow lane)
        :param v_target: [m/s] followed object's velocity or target velocity for follow lane
        :param a_target: [m/s^2] followed object's acceleration
        :param aggressiveness_level: attribute of the semantic action
        :param cars_size_margin: [m] sum of half lengths of ego and target
        :param min_time: [sec] minimal time for the profile
        :return: VelocityProfile class or None in case of infeasible semantic action
        """
        acc = AGGRESSIVENESS_TO_LON_ACC[aggressiveness_level.value]  # profile acceleration
        v_max = max(BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, v_target)
        if action_type == ActionType.FOLLOW_VEHICLE:
            dist = lon_target - lon_init - SAFE_DIST_TIME_DELAY * v_target - cars_size_margin
            return VelocityProfile._calc_profile_given_acc(v_init, acc, v_max, dist, v_target, a_target, min_time)
        elif action_type == ActionType.OVER_TAKE_VEHICLE:
            dist = lon_target - lon_init + SAFE_DIST_TIME_DELAY * v_target + cars_size_margin
            return VelocityProfile._calc_profile_given_acc(v_init, acc, v_max, dist, v_target, a_target, min_time)
        elif action_type == ActionType.FOLLOW_LANE:
            t1 = abs(v_target - v_init) / acc
            if 0 <= t1 < min_time:  # two segments
                vel_profile = cls(v_init=v_init, t1=t1, v_mid=v_target, t2=min_time - t1, t3=0, v_tar=v_target)
            else:  # single segment (if too short profile, then choose lower acceleration according to min_time)
                vel_profile = cls(v_init=v_init, t1=max(t1, min_time), v_mid=v_target, t2=0, t3=0, v_tar=v_target)
            return vel_profile

    @classmethod
    def _calc_profile_given_acc(cls, v_init: float, a: float, v_max: float, dist: float,
                                v_tar: float, a_tar: float, min_time: float=0):
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
        :param v_max: maximal desired velocity of ego
        :param dist: initial distance to the safe location from the target
        :param v_tar: target object velocity
        :param a_tar: target object acceleration
        :param min_time: minimal time for the profile
        return: VelocityProfile class or None in case of infeasible semantic action
        """
        # print('CALC PROFILE: v_init=%f dist=%f' % (v_init, dist))

        v_init_rel = v_init - v_tar  # relative velocity; may be negative
        v_max_rel = max(v_max - v_tar, BP_MAX_VELOCITY_TOLERANCE)  # v_max > v_tar to enable reaching the target car

        if abs(v_init_rel) < 0.1 and abs(dist) < 0.1:
            return cls(v_init, min_time, v_tar, 0, 0, v_tar)  # just follow the target car for min_time

        if v_init_rel * dist > 0 and v_init_rel * v_init_rel > 2 * (a + a_tar) * abs(dist):  # too big acceleration needed
            # print('NO PROFILE: too big acceleration needed: v_init_rel=%f dist=%f acc=%f' % (v_init_rel, dist, a + a_tar))
            # return None
            t1 = max(min_time, abs(v_init_rel) / a)  # increase dist (if it's unsafe, the action will be filtered later)
            return cls(v_init, t1, v_tar, 0, 0, v_tar)  # one segment profile

        if a > 0 and (np.math.isclose(a, a_tar, rel_tol=0.05) or np.math.isclose(a, -a_tar, rel_tol=0.05)):
            a -= 0.1  # slightly change the acceleration to prevent division by zero

        # first try profile with two segments
        # first acceleration, then deceleration
        # here we use formula (vm^2 - v^2)/2(a-a_tar) + vm^2/2(a+a_tar) = dist
        v_mid_rel_sqr = (v_init_rel * v_init_rel / 2 + dist * (a - a_tar)) * (a + a_tar) / a
        two_segments_failed = False
        regular_order = True
        t1 = t3 = 0
        v_mid_rel = v_init_rel
        if v_mid_rel_sqr >= 0:
            v_mid_rel = np.sqrt(v_mid_rel_sqr)  # should be positive
            t1 = (v_mid_rel - v_init_rel) / (a - a_tar)  # acceleration time
            t3 = v_mid_rel / (a + a_tar)                 # deceleration time
            if t1 < 0 or t3 < 0:  # negative time, try single segment with another acceleration
                two_segments_failed = True
        else:  # the target is unreachable, try single segment with another acceleration
            two_segments_failed = True

        if two_segments_failed:  # try opposite order: first deceleration, then acceleration
            # here the formula (v^2 - vm^2)/2(a+a_tar) - vm^2/2(a-a_tar) = dist
            v_mid_rel_sqr = (v_init_rel * v_init_rel / 2 - dist * (a + a_tar)) * (a - a_tar) / a
            two_segments_failed = False
            regular_order = False
            t1 = t3 = 0
            v_mid_rel = v_init_rel
            if v_mid_rel_sqr >= 0:
                v_mid_rel = -np.sqrt(v_mid_rel_sqr)  # should be negative
                t1 = (v_init_rel - v_mid_rel) / (a + a_tar)  # deceleration time
                t3 = -v_mid_rel / (a - a_tar)  # acceleration time
                if t1 < 0 or t3 < 0:  # negative time, try single segment with another acceleration
                    two_segments_failed = True
            else:  # the target is unreachable, try single segment with another acceleration
                two_segments_failed = True

        # if two segments failed, try a single segment with lower acceleration
        if two_segments_failed:
            if v_init_rel * dist <= 0:
                print('NO PROFILE: v_mid_rel_sqr=%f or time t1=%f t3=%f; v_init_rel=%f v_mid_rel=%f a=%f dist=%f' %
                      (v_mid_rel_sqr, t1, t3, v_init_rel, v_mid_rel, a, dist))
                return None  # illegal action
            else:  # then take single segment with another acceleration
                t1 = 2*dist/v_init_rel
                return cls(v_init, t1, v_tar, 0, 0, v_tar)

        # if the profile is shorter than min_time, then decrease the deceleration (the acceleration does not change):
        # decrease t1 and v_mid_rel and increase t3 by decreasing deceleration.
        if t1 + t3 < min_time:
            if regular_order:  # acceleration, deceleration
                # solve the equation for t1: (v0+vm)*t1/2 + vm*(min_time-t1)/2 = dist; where vm = v0 + a*t1
                t1 = (2 * dist - v_init_rel * min_time) / (v_init_rel + (a - a_tar) * min_time)
                if t1 > 0:  # two segments
                    t3 = min_time - t1
                    v_mid_rel = v_init_rel + (a - a_tar) * t1
                    return cls(v_init, t1, v_mid_rel + v_tar + a_tar*t1, 0, t3, v_tar)
                else:  # single decelerating segment
                    t1 = 2*dist / v_init_rel
                    return cls(v_init, t1, v_tar, 0, 0, v_tar)
            else:  # opposite order: deceleration, acceleration
                # solve the same equation for t1: (v0+vm)*t1/2 + vm*(min_time-t1)/2 = dist; but here vm = v0 - a*t1
                t1 = (v_init_rel * min_time - 2 * dist) / ((a + a_tar) * min_time - v_init_rel)
                if 0. <= t1 <= min_time:
                    t3 = min_time - t1
                    v_mid_rel = v_init_rel - (a + a_tar) * t1  # negative
                    return cls(v_init, t1, v_mid_rel + v_tar + a_tar*t1, 0, t3, v_tar)
                else:  # no profile
                    print('NO PROFILE (time < min_time): t1=%f min_time=%f; v_init_rel=%f v_mid_rel=%f a=%f dist=%f' %
                        (t1, min_time, v_init_rel, v_mid_rel, a, dist))
                    return None  # illegal action

        if v_mid_rel <= v_max_rel or dist < 0:  # ego does not reach max_vel, then t2 = 0
            return cls(v_init, t1, v_mid_rel + v_tar + a_tar*t1, 0, t3, v_tar)

        # from now: ego reaches max_vel, such that t2 > 0

        t1 = abs(v_max_rel - v_init_rel) / a  # acceleration time (maybe deceleration)
        if a_tar == 0:  # a simple case: the followed car has constant velocity
            t3 = v_max_rel / a  # deceleration time
            dist_mid = dist - (abs(v_max_rel*v_max_rel - v_init_rel*v_init_rel) + v_max_rel*v_max_rel) / (2*a)
            t2 = max(0., dist_mid / v_max_rel)  # constant velocity (max_vel) time
            return cls(v_init, t1, v_max_rel + v_tar, t2, t3, v_tar)

        # from now the most general case: t2 > 0 and the followed car has non-zero acceleration

        # Notations:
        #   v is initial relative velocity
        #   a > 0 is ego acceleration, a_tar target object acceleration
        #   vm1 is relative velocity of ego immediately after acceleration
        #   vm2 is relative velocity of ego immediately before the deceleration, i.e. vm2 = vm1 - a_tar*t2
        # Quadratic equation: tot_dist is the sum of 3 distances:
        #   acceleration distance     max_vel distance     deceleration distance
        #   (vm1^2 - v^2)/2(a-a_tar)  +  (vm1+vm2)/2 * t2  +  vm2^2/2(a+a_tar)   =   dist
        v = v_init_rel
        vm1 = v + (a - a_tar) * t1
        # after substitution of vm2 = vm1 - a1*t and simplification, solve quadratic equation on t2:
        # a*a_tar * t^2 - 2*vm1*a * t2 - (vm1^2 + 2(a+a_tar) * ((vm1^2 - v^2)/2(a-a_tar) - dist)) = 0
        c = vm1*vm1 + 2*(a + a_tar) * ((vm1 * vm1 - v * v) / (2 * (a - a_tar)) - dist)  # free coefficient
        discriminant = vm1*vm1 * a*a + a*a_tar * c  # discriminant of the quadratic equation
        if discriminant < 0:
            print('NO PROFILE: general case: discriminant < 0')
            return None  # illegal action
        t2 = (vm1 * a + np.sqrt(discriminant)) / (a * a_tar)
        vm2 = vm1 - a_tar*t2
        t3 = vm2 / (a + a_tar)
        if t3 < 0:
            print('NO PROFILE: general case: t3 < 0')
            return None  # illegal action
        return cls(v_init, t1, v_max, t2, t3, v_tar)

    @classmethod
    def calc_profile_given_T(cls, v_init: float, T: float, dist: float, v_tar: float):
        """
        Given start & end velocities, distance to the followed car and acceleration, calculate velocity profile:
            1. acceleration to a velocity v_mid <= v_max for t1 time,
            2. moving by v_max for t2 time (t2 = 0 if v_mid < v_max),
            3. deceleration to end_vel for t3 time.
        If this profile is infeasible, then try an opposite order of accelerations: 1. deceleration, 3. acceleration.
        In the case of opposite order, the constant velocity segment is missing.
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
        v_init_rel = v_init - v_tar  # relative velocity; may be negative
        # let v = v_init_rel, v1 = v_mid_rel, t = t1, d = dist, solve for a (acceleration)
        # for the simple case (acceleration, deceleration) solve the following equations:
        # v1^2 - v^2 = 2ad, v1 = v + at, v1 = a(T-t)
        # it is simplified to quadratic equation for a: T^2*a^2 - 2(2d-Tv)a - v^2 = 0
        # solution: a = ( (2d-Tv) +- sqrt((2d-Tv)^2 + (Tv)^2) ) / T^2
        Tv_2d = 2 * dist - T * v_init_rel
        discriminant = Tv_2d ** 2 + (T * v_init_rel) ** 2  # always positive
        if dist > 0:
            a = (Tv_2d + np.sqrt(discriminant)) / T ** 2  # always positive
        else:  # dist < 0
            a = (Tv_2d - np.sqrt(discriminant)) / T ** 2  # always negative
        t1 = 0.5 * (T - v_init_rel / a)
        if 0. <= t1 <= T:
            v_mid_rel = v_init_rel + a * t1
            return cls(v_init, t1, v_mid_rel + v_tar, 0, T - t1, v_tar)  # acceleration, deceleration
        else:  # invalid t1, try acceleration/deceleration + constant vel
            t1 = 2 * dist / v_init_rel
            if 0. <= t1 <= T:
                return cls(v_init, t1, v_tar, T - t1, 0, v_tar)  # deceleration + constant vel
            else:
                print('NO PROFILE v_init_rel <= v_max_rel: t1=%.2f v_init=%.2f v_tar=%.2f T=%.2f' % (t1, v_init, v_tar, T))
                return None

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
        lat_acc = AGGRESSIVENESS_TO_LAT_ACC[aggressiveness_level.value] * lane_width / 3.6
        lateral_profile = VelocityProfile._calc_profile_given_acc(
            lat_v_init_toward_target, lat_acc, np.inf, abs(signed_lat_dist), 0, 0)

        # calculate total deviation from lane center
        acc = AGGRESSIVENESS_TO_LAT_ACC[aggressiveness_level.value]
        rel_lat = abs(signed_lat_dist)/lane_width
        rel_vel = init_lat_vel/lane_width
        if signed_lat_dist * init_lat_vel < 0:  # changes lateral direction
            rel_lat += rel_vel*rel_vel/(2*acc)
        max_dev = min(2*rel_lat, 1)  # for half-lane deviation, max_dev = 1

        return lateral_profile.t1 + lateral_profile.t3, max_dev, lat_v_init_toward_target


class ProfileSafety:

    @staticmethod
    def is_safe(v_front: float, v_back: float, dist: float, time_delay: float, margin: float):
        max_brake = -LON_ACC_LIMITS[0]
        return max(0., v_back**2 - v_front**2) / (2*max_brake) + v_back*time_delay + margin < dist

    @staticmethod
    def check_safety(ego_lon: float, ego_half_size: float, vel_profile: VelocityProfile,
                            dyn_obj: DynamicObject, T_max: float, time_delay: float) -> bool:
        """
        Given ego velocity profile and dynamic object, calculate the last time, when the safety complies.
        Here we assume the object has a constant velocity.
        :param ego_lon: ego initial longitude
        :param ego_half_size: [m] half length of ego
        :param vel_profile: ego velocity profile
        :param dyn_obj: the dynamic object, for which the safety is tested
        :param T_max: maximal time to check the safety (usually T_d for safety w.r.t. F and LB)
        :param time_delay: reaction time of the back car
        :return: the latest safe time
        """
        if T_max <= 0:
            return True
        (init_s_ego, init_s_obj, v_obj) = (ego_lon, dyn_obj.road_localization.road_lon, dyn_obj.v_x)
        margin = ego_half_size + dyn_obj.size.length / 2
        cut_profile = vel_profile.cut_by_time(T_max)

        if init_s_ego < init_s_obj:  # the object is in front of ego
            t = cut_profile.when_decel_velocity_is_equal_to(v_obj)
            if t is None:  # cut_profile does not contain velocity v_obj
                if cut_profile.v_init < v_obj:  # ego is always slower
                    return ProfileSafety.is_safe(v_obj, cut_profile.v_init, init_s_obj - init_s_ego, time_delay, margin)
                else:  # ego is always faster, check only at the end
                    tot_time = cut_profile.total_time()
                    tot_dist = cut_profile.total_dist()
                    return ProfileSafety.is_safe(v_obj, cut_profile.v_tar,
                                                 (init_s_obj + v_obj*tot_time) - (init_s_ego + tot_dist),
                                                 time_delay, margin)
            else:  # t was found
                s, _ = cut_profile.sample_at(t)  # v_ego(t2) = v_obj
                return init_s_obj + v_obj*t - (init_s_ego + s) > v_obj * time_delay + margin
        else:  # the object is behind ego
            t = cut_profile.when_accel_velocity_is_equal_to(v_obj)
            if t is None:  # cut_profile does not contain velocity v_obj
                if cut_profile.v_init > v_obj:  # ego is always faster
                    return ProfileSafety.is_safe(cut_profile.v_init, v_obj, init_s_ego - init_s_obj, time_delay, margin)
                else:  # ego is always slower, check only at the end
                    tot_time = cut_profile.total_time()
                    tot_dist = cut_profile.total_dist()
                    return ProfileSafety.is_safe(cut_profile.v_tar, v_obj,
                                                 (init_s_ego + tot_dist) - (init_s_obj + v_obj*tot_time),
                                                 time_delay, margin)
            else:  # t was found
                s, _ = cut_profile.sample_at(t)  # v_ego(t2) = v_obj
                return (init_s_ego + s) - (init_s_obj + v_obj*t) > v_obj * time_delay + margin

    @staticmethod
    def calc_last_safe_time(ego_lon: float, ego_half_size: float, vel_profile: VelocityProfile,
                            dyn_obj: DynamicObject, T_max: float, time_delay: float) -> float:
        """
        Given ego velocity profile and dynamic object, calculate the last time, when the safety complies.
        :param ego_lon: ego initial longitude
        :param ego_half_size: [m] half length of ego
        :param vel_profile: ego velocity profile
        :param dyn_obj: the dynamic object, for which the safety is tested
        :param T_max: maximal time to check the safety (usually T_d for safety w.r.t. F and LB)
        :param time_delay: reaction time of the back car
        :return: the latest safe time
        """
        if T_max <= 0:
            return np.inf

        margin = ego_half_size + dyn_obj.size.length / 2
        # initialization of motion parameters
        (init_s_ego, init_v_obj, a_obj) = (ego_lon, dyn_obj.v_x, dyn_obj.acceleration_lon)
        init_s_obj = dyn_obj.road_localization.road_lon

        if a_obj == 0:  # the object has constant velocity, then use fast safety test
            dist = abs(init_s_obj - init_s_ego)
            if init_s_ego < init_s_obj:
                v_front = init_v_obj
                v_back = vel_profile.v_init
            else:
                v_back = init_v_obj
                v_front = vel_profile.v_init
            if not ProfileSafety.is_safe(v_front, v_back, dist, time_delay, margin):
                return -1
            if ProfileSafety.check_safety(ego_lon, ego_half_size, vel_profile, dyn_obj, T_max, time_delay):
                return np.inf

        t, t_cum, s_ego, v_ego, a_ego = vel_profile.get_details(T_max)
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
            safe_time = ProfileSafety._calc_largest_time_for_segment(
                s[seg, front], v[seg, front], a[seg, front], s[seg, back], v[seg, back], a[seg, back], t[seg],
                margin, time_delay)
            if safe_time < 0:
                return last_safe_time
            last_safe_time += safe_time
            if last_safe_time < t_cum[seg + 1]:  # becomes unsafe inside this segment
                # check if delayed last_safe_time (t+td) overflowed to the next segment
                T = last_safe_time + time_delay
                if T > t_cum[seg + 1]:  # then check safety on delayed point of vel_segment
                    # suppose the object moves last_safe_time, then fully brakes during time_delay
                    # ego moves according to vel_profile during T (then fully brakes)
                    braking_time = min(time_delay, init_v_obj / a_max)
                    if init_s_ego < init_s_obj:
                        s_back, v_back = vel_profile.sample_at(T)
                        s_back += init_s_ego
                        s_front = init_s_obj + init_v_obj * T + 0.5 * a_obj * last_safe_time * last_safe_time - \
                                  0.5 * a_max * braking_time * braking_time
                        v_front = max(0., init_v_obj + a_obj * last_safe_time - a_max * time_delay)
                    else:
                        s_front, v_front = vel_profile.sample_at(T)
                        s_front += init_s_ego
                        s_back = init_s_obj + init_v_obj * T + 0.5 * a_obj * last_safe_time * last_safe_time - \
                                 0.5 * a_max * braking_time * braking_time
                        v_back = max(0., init_v_obj + a_obj * last_safe_time - a_max * time_delay)
                    if max(0., v_back**2 - v_front**2)/(2*a_max) + margin <= s_front - s_back:
                        last_safe_time = t_cum[seg + 1]
                        continue  # this segment is safe
                return last_safe_time
        return np.inf  # always safe

    @staticmethod
    def _calc_largest_time_for_segment(s1: float, v1: float, a1: float, s2: float, v2: float, a2: float,
                                       T: float, margin: float, time_delay: float) -> float:
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
        :param time_delay: reaction time of the back car (AV is faster)
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

        C = s1 - s2 + (v1*v1 - v2*v2)/(2*a_max) - (0.5*a2*time_delay + v2) * time_delay*(1 + a2/a_max) - margin

        #print('safety: s1=%.2f v1=%.2f a1=%.2f  s2=%.2f v2=%.2f a2=%.2f  T=%.2f margin=%.2 C=%.2f A=%.2f B=%.2f' % \
        #      (s1, v1, a1, s2, v2, a2, T, margin, C, (a1-a2) * (a_max+a1+a2) / (2*a_max), (a1*v1 - a2*v2)/a_max + v1 - v2 - a2*time_delay*(1 + a2/a_max)))


        if C < 0:
            return -1  # the current state (for t=0) is not safe
        if T == 0:
            return 0  # the current state (for t=0) is safe

        A = (a1-a2) * (a_max+a1+a2) / (2*a_max)
        B = (a1*v1 - a2*v2)/a_max + v1 - v2 - a2*time_delay*(1 + a2/a_max)
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
