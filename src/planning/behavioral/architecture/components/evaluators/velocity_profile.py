import numpy as np

from decision_making.src.global_constants import AGGRESSIVENESS_TO_LON_ACC, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, \
    SAFE_DIST_TIME_DELAY, LON_ACC_LIMITS, AGGRESSIVENESS_TO_LAT_ACC
from decision_making.src.planning.behavioral.architecture.data_objects import ActionType, AggressivenessLevel
from decision_making.src.planning.types import FP_SX, LIMIT_MIN
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
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

    def total_time(self):
        return self.t1 + self.t2 + self.t3

    def total_dist(self):
        return 0.5 * ((self.v_init + self.v_mid) * self.t1 + (self.v_mid + self.v_tar) * self.t3) + self.v_mid * self.t2

    @classmethod
    def calc_velocity_profile(cls, action_type: ActionType, lon_init: float, v_init: float, lon_target: float,
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
            return VelocityProfile._calc_velocity_profile_follow_car(v_init, acc, v_max, dist, v_target, a_target, min_time)
        elif action_type == ActionType.TAKE_OVER_VEHICLE:
            dist = lon_target - lon_init + SAFE_DIST_TIME_DELAY * v_target + cars_size_margin
            return VelocityProfile._calc_velocity_profile_follow_car(v_init, acc, v_max, dist, v_target, a_target, min_time)
        elif action_type == ActionType.FOLLOW_LANE:
            t1 = abs(v_target - v_init) / acc
            if 0 <= t1 < min_time:  # two segments
                vel_profile = cls(v_init=v_init, t1=t1, v_mid=v_target, t2=min_time - t1, t3=0, v_tar=v_target)
            else:  # single segment (if too short profile, then choose lower acceleration according to min_time)
                vel_profile = cls(v_init=v_init, t1=max(t1, min_time), v_mid=v_target, t2=0, t3=0, v_tar=v_target)
            return vel_profile

    @classmethod
    def _calc_velocity_profile_follow_car(cls, v_init: float, a: float, v_max: float, dist: float,
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

        MAX_VELOCITY_TOLERANCE = 2.
        v_init_rel = v_init - v_tar  # relative velocity; may be negative
        v_max_rel = max(v_max - v_tar, max(v_init_rel, MAX_VELOCITY_TOLERANCE))  # let max_vel > end_vel to enable reaching the target car

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

        t1 = abs(v_max - v_init) / a  # acceleration time
        if a_tar == 0:  # a simple case: the followed car has constant velocity
            t3 = v_max_rel / a  # deceleration time
            dist_mid = dist - (2*v_max_rel*v_max_rel - v_init_rel*v_init_rel) / (2*a)
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

    @staticmethod
    def calc_lateral_time(init_lat_vel: float, signed_lat_dist: float, lane_width: float,
                          aggressiveness_level: AggressivenessLevel):
        """
        Given initial lateral velocity and signed lateral distance, estimate a time it takes to perform the movement.
        The time estimation assumes movement by velocity profile like in the longitudinal case.
        :param init_lat_vel: [m/s] initial lateral velocity
        :param signed_lat_dist: [m] signed distance to the target
        :param lane_width: [m] lane width
        :param aggressiveness_level: aggressiveness_level
        :return: [s] the lateral movement time to the target
        """
        if signed_lat_dist > 0:
            lat_v_init_toward_target = init_lat_vel
        else:
            lat_v_init_toward_target = -init_lat_vel
        # normalize lat_acc by lane_width, such that T_d will NOT be depend on lane_width
        lat_acc = AGGRESSIVENESS_TO_LAT_ACC[aggressiveness_level.value] * lane_width / 3.6
        lateral_profile = VelocityProfile._calc_velocity_profile_follow_car(
            lat_v_init_toward_target, lat_acc, np.inf, abs(signed_lat_dist), 0, 0)
        return lateral_profile.t1 + lateral_profile.t3


class ProfileSafety:

    @staticmethod
    def calc_last_safe_time(ego_lon: float, ego_half_size: float, vel_profile: VelocityProfile,
                            dyn_obj: DynamicObject, T_max: float) -> float:
        """
        Given ego velocity profile and dynamic object, calculate the last time, when the safety complies.
        :param ego_lon: ego initial longitude
        :param ego_half_size: [m] half length of ego
        :param vel_profile: ego velocity profile
        :param dyn_obj: the dynamic object, for which the safety is tested
        :param T_max: maximal time to check the safety (usually T_d for safety w.r.t. F and LB)
        :return: the latest safe time
        """
        # check safety until completing the lane change
        if T_max <= 0:
            return np.inf
        # initialization of motion parameters
        (init_s_ego, init_v_obj, a_obj) = (ego_lon, dyn_obj.v_x, dyn_obj.acceleration_lon)
        init_s_obj = dyn_obj.road_localization.road_lon

        t, t_cum, s_ego, v_ego, a_ego = vel_profile.get_details(T_max)
        s_ego += init_s_ego
        s_obj = init_s_obj + init_v_obj * t_cum[:-1] + 0.5 * a_obj * t_cum[:-1] * t_cum[:-1]
        v_obj = init_v_obj + a_obj * t_cum[:-1]

        # create (ego, obj) pairs of longitudes, velocities and accelerations for all segments
        (s, v, a) = (np.c_[s_ego, s_obj], np.c_[v_ego, v_obj], np.c_[a_ego, np.repeat(a_obj, t.shape[0])])

        front = int(init_s_ego < init_s_obj)  # 0 if the object is behind ego; 1 otherwise
        back = 1 - front
        if init_s_ego < init_s_obj:
            time_delay = 0.8  # AV has faster reaction; it's necessary for overtaking of a close car
        else:
            time_delay = SAFE_DIST_TIME_DELAY

        # calculate last_safe_time
        last_safe_time = 0
        for seg in range(t.shape[0]):
            if t[seg] == 0:
                continue
            last_safe_time += ProfileSafety._calc_largest_time_for_segment(
                s[seg, front], v[seg, front], a[seg, front], s[seg, back], v[seg, back], a[seg, back], t[seg],
                ego_half_size + dyn_obj.size.length/2, time_delay)

            # print('target_lat=%f: last_safe_time=%f s1=%f, v1=%f, s2=%f, v2=%f, t=%f t_cum[seg+1]=%f max_time=%f cur_lat=%f' %
            #       (target_lat, last_safe_time, s[seg, front], v[seg, front], s[seg, back], v[seg, back], t[seg],
            #        t_cum[seg+1], max_time, ego_fpoint[FP_DX]))

            if last_safe_time < t_cum[seg+1]:  # becomes unsafe inside this segment
                return last_safe_time
        return t_cum[-1]  # always safe

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
