from typing import Optional

import numpy as np

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, SAFE_DIST_TIME_DELAY, \
    WERLING_TIME_RESOLUTION, AGGRESSIVENESS_TO_LON_ACC, LON_ACC_TO_COST_FACTOR, LAT_ACC_TO_COST_FACTOR, \
    RIGHT_LANE_COST_WEIGHT, EFFICIENCY_COST_WEIGHT, LAT_JERK_COST_WEIGHT, LON_JERK_COST_WEIGHT, LON_ACC_LIMITS, \
    MIN_ACTION_PERIOD, PLAN_LATERAL_VELOCITY
from decision_making.src.planning.performance_metrics.cost_functions import EfficiencyMetric
from decision_making.src.planning.types import LIMIT_MAX, LIMIT_MIN, FP_SX, FP_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.state import DynamicObject, EgoState, ObjectSize


class VelocityProfile:
    def __init__(self, v_init: float, t1: float, v_mid: float, t2: float, t3: float, v_tar: float):
        self.v_init = v_init    # initial ego velocity
        self.t1 = t1            # acceleration/deceleration time period
        self.v_mid = v_mid      # maximal velocity after acceleration
        self.t2 = t2            # time period for going with maximal velocity
        self.t3 = t3            # deceleration/acceleration time
        self.v_tar = v_tar      # end velocity
        self.t_follow = max(0., MIN_ACTION_PERIOD - self.t1 - self.t2 - self.t3)  # time for following the target car

    def get_details(self, max_time: float=np.inf) -> [np.array, np.array, np.array, np.array, np.array]:
        """
        Return times, longitudes, velocities, accelerations for the current profile.
        :param max_time: if the profile is longer than max_time, then truncate it
        :return: numpy arrays per segment:
            times: time period of each segment
            cumulative times: cumulative times of the segments, with leading 0
            longitudes: cumulated distances per segment (except the last one), with leading 0
            velocities: velocities per segment
            accelerations: accelerations per segment
        All arrays' size is equal to the (truncated) segments number, except t_cum having extra 0 at the beginning.
        """
        t = np.array([self.t1, self.t2, self.t3, self.t_follow])
        t_cum = np.concatenate(([0], np.cumsum(t)))
        max_time = max(0., max_time)

        acc1 = acc3 = 0
        if self.t1 > 0:
            acc1 = (self.v_mid - self.v_init) / self.t1
        if self.t3 > 0:
            acc3 = (self.v_tar - self.v_mid) / self.t3
        a = np.array([acc1, 0, acc3, 0])
        v = np.array([self.v_init, self.v_mid, self.v_mid, self.v_tar])
        s = np.array([0, 0.5 * (self.v_init + self.v_mid) * self.t1, self.v_mid * self.t2,
                      0.5 * (self.v_mid + self.v_tar) * self.t3])

        if t_cum[-1] > max_time:  # then truncate all arrays by max_time
            truncated_size = np.where(t_cum[:-1] < max_time)[0][-1] + 1
            t = t[:truncated_size]  # truncate times array
            t[-1] -= t_cum[-1] - max_time  # decrease the last segment time
            t_cum = np.concatenate(([0], np.cumsum(t)))
            a = a[:truncated_size]  # truncate accelerations array
            v = v[:truncated_size]  # truncate velocities array
            s = s[:truncated_size]  # truncate distances array

        s_cum = np.concatenate(([0], np.cumsum(s[:-1])))
        return t, t_cum, s_cum, v, a

    @classmethod
    def calc_velocity_profile(cls, lon_init: float, v_init: float, lon_target: float, v_target: float, a_target: float,
                              aggressiveness_level: int):
        """
        calculate velocities profile for semantic action: either following car or following lane
        :param lon_init: [m] initial longitude of ego
        :param v_init: [m/s] initial velocity of ego
        :param lon_target: [m] initial longitude of followed object (None if follow lane)
        :param v_target: [m/s] followed object's velocity or target velocity for follow lane
        :param a_target: [m/s^2] followed object's acceleration
        :param aggressiveness_level: [int] attribute of the semantic action
        :return: VelocityProfile class or None in case of infeasible semantic action
        """
        acc = AGGRESSIVENESS_TO_LON_ACC[aggressiveness_level]  # profile acceleration
        v_max = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
        if lon_target is not None:  # follow car
            dist = lon_target - lon_init - SAFE_DIST_TIME_DELAY * v_target - 5  # TODO: replace 5 by cars' half sizes sum
            return VelocityProfile._calc_velocity_profile_follow_car(v_init, acc, v_max, dist, v_target, a_target)
        else:  # follow lane
            t1 = abs(v_target - v_init) / acc
            return cls(v_init=v_init, t1=t1, v_mid=v_target, t2=0, t3=0, v_end=v_target)

    @classmethod
    def _calc_velocity_profile_follow_car(cls, v_init: float, a: float, v_max: float, dist: float,
                                          v_tar: float, a_tar: float):
        """
        Given start & end velocities and distance to the followed car, calculate velocity profile:
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
        return: VelocityProfile class or None in case of infeasible semantic action
        """
        print('CALC PROFILE: v_init=%f dist=%f' % (v_init, dist))

        v_init_rel = v_init - v_tar  # relative velocity; may be negative
        v_max_rel = max(v_max - v_tar, max(v_init_rel, 1.))  # let max_vel > end_vel to enable reaching the car
        if a > 0 and (np.math.isclose(a, a_tar, rel_tol=0.05) or np.math.isclose(a, -a_tar, rel_tol=0.05)):
            a -= 0.1  # slightly change the acceleration to prevent instability in calculations

        # here we use formula (vm^2 - v^2)/2(a-a_tar) + vm^2/2(a+a_tar) = dist
        v_mid_rel_sqr = (v_init_rel * v_init_rel / 2 + dist * (a - a_tar)) * (a + a_tar) / a
        try_opposite_order = False
        t1 = t3 = 0
        v_mid_rel = v_init_rel
        if v_mid_rel_sqr >= 0:
            v_mid_rel = np.sqrt(v_mid_rel_sqr)
            t1 = (v_mid_rel - v_init_rel) / (a - a_tar)  # acceleration time
            t3 = v_mid_rel / (a + a_tar)                 # deceleration time
            if t1 < 0 or t3 < 0:  # illegal time, try the opposite order of accelerations
                try_opposite_order = True
        else:  # the target is unreachable, try the opposite order of accelerations
            try_opposite_order = True

        # try_opposite_order: first deceleration, then acceleration
        # here we use formula (v^2 - vm^2)/2(a+a_tar) - vm^2/2(a-a_tar) = dist
        if try_opposite_order:  # first deceleration, then acceleration
            v_mid_rel_sqr = (v_init_rel * v_init_rel / 2 - dist * (a + a_tar)) * (a - a_tar) / a
            if v_mid_rel_sqr < 0:  # the target is unreachable
                print('BAD: negative v_mid_rel_sqr=%f tot_dist=%f' % (v_mid_rel_sqr, dist))
                return None  # illegal action
            v_mid_rel = -np.sqrt(v_mid_rel_sqr)
            t1 = (v_init_rel - v_mid_rel) / (a - a_tar)   # deceleration time
            t3 = -v_mid_rel / (a + a_tar)                 # acceleration time
            if t1 < 0 or t3 < 0:
                print('BAD: negative t1 or t3 or v_mid: v_mid_rel=%f t1=%f t3=%f' % (v_mid_rel, t1, t3))
                return None  # illegal action

        if v_init + t1 * a <= v_max or dist < 0:  # ego does not reach max_vel, then t2 = 0
            if t3 < 0 or v_mid_rel + v_tar < 0:  # negative t3 or negative v_mid
                print('BAD: negative t3 or v_mid: t3=%f v_mid_rel=%f' % (t3, v_mid_rel))
                return None  # illegal action
            return VelocityProfile(v_init, t1, v_mid_rel + v_tar, 0, t3, v_tar)

        # from now: ego reaches max_vel, such that t2 > 0

        t1 = (v_max - v_init) / a  # acceleration time
        if a_tar == 0:  # a simple case: the followed car has constant velocity
            t3 = v_max_rel / a  # deceleration time
            dist_mid = dist - (2 * v_max_rel * v_max_rel - v_init_rel * v_init_rel) / (2 * a)
            t2 = max(0., dist_mid / v_max_rel)  # constant velocity (max_vel) time
            return VelocityProfile(v_init, t1, v_max, t2, t3, v_tar)

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
            print('BAD: general case: det < 0')
            return None  # illegal action
        t2 = (vm1 * a + np.sqrt(discriminant)) / (a * a_tar)
        vm2 = vm1 - a_tar*t2
        t3 = vm2 / (a + a_tar)
        return VelocityProfile(v_init, t1, v_max, t2, t3, v_tar)


class ProfileSafety:

    @staticmethod
    def calc_last_safe_time(ego_fpoint: np.array, ego_size: ObjectSize, vel_profile: VelocityProfile,
                            target_lat: float, dyn_obj: DynamicObject, road_frenet: FrenetSerret2DFrame) -> float:
        """
        Given ego velocity profile and dynamic object, calculate the last time, when the safety complies.
        :param ego_fpoint: ego initial Frenet point
        :param ego_size: size of ego
        :param vel_profile: ego velocity profile
        :param target_lat: target latitude in Frenet
        :param dyn_obj: the dynamic object, for which the safety is tested
        :param road_frenet: road Frenet frame
        :return: last safe time
        """
        # check safety until completing the lane change
        max_time = abs(target_lat - ego_fpoint[FP_DX]) / PLAN_LATERAL_VELOCITY
        if max_time < 1.:  # if ego is close to target_lat, then don't check safety
            return np.inf
        margin = 0.5 * (ego_size.length + dyn_obj.size.length)
        # initialization of motion parameters
        (init_s_ego, init_v_obj, a_obj) = (ego_fpoint[FP_SX], dyn_obj.v_x, dyn_obj.acceleration_lon)
        init_s_obj = road_frenet.cpoint_to_fpoint(np.array([dyn_obj.x, dyn_obj.y]))[FP_SX]

        t, t_cum, s_ego, v_ego, a_ego = vel_profile.get_details(max_time)
        s_ego += init_s_ego
        s_obj = init_s_obj + init_v_obj * t_cum[:-1] + 0.5 * a_obj * t_cum[:-1] * t_cum[:-1]
        v_obj = init_v_obj + a_obj * t_cum[:-1]

        # create (ego, obj) pairs of longitudes, velocities and accelerations for all segments
        (s, v, a) = (np.c_[s_ego, s_obj], np.c_[v_ego, v_obj], np.c_[a_ego, np.repeat(a_obj, t.shape[0])])

        front = int(init_s_ego < init_s_obj)  # 0 if the object is behind ego; 1 otherwise
        back = 1 - front

        # calculate last_safe_time
        last_safe_time = 0
        for seg in range(t.shape[0]):
            last_safe_time += ProfileSafety._calc_largest_time_for_segment(
                s[seg, front], v[seg, front], a[seg, front], s[seg, back], v[seg, back], a[seg, back], t[seg], margin)
            if last_safe_time < t_cum[seg+1]:  # becomes unsafe inside this segment
                return last_safe_time
        return t_cum[-1]  # always safe

    @staticmethod
    def _calc_largest_time_for_segment(s1: float, v1: float, a1: float, s2: float, v2: float, a2: float,
                                       T: float, margin: float) -> float:
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
        :return: the largest safe time; if unsafe for t=0, return -1
        """
        if T < 0:
            return -1
        # the first vehicle is in front of the second one
        # dist = (s1 + v1*t + 0.5*a1*t*t) - (s2 + v2*t + 0.5*a2*t*t)
        # let v1_t = v1 + a1*t
        # let v2_t = v2 + a2*t
        # safe_dist = (v2_t*v2_t - v1_t*v1_t) / (2*a_max) + v2_t*time_delay + margin
        # solve quadratic inequality: dist - safe_dist = A*t^2 + B*t + C >= 0

        a_max = -LON_ACC_LIMITS[LIMIT_MIN]
        time_delay = SAFE_DIST_TIME_DELAY

        C = s1 - s2 + (v1*v1 - v2*v2)/(2*a_max) - v2*time_delay - margin
        if C < 0:
            return -1  # the current state (for t=0) is not safe
        if T == 0:
            return 0  # the current state (for t=0) is safe

        A = (a1-a2) * (a_max+a1+a2) / (2*a_max)
        B = (a1*v1 - a2*v2)/a_max + v1 - v2 - a2*time_delay
        if A == 0 and B == 0:  # constant function C > 0
            return T  # for all t it's safe
        if A == 0:  # B != 0; linear inequality
            t = -C/B
            if t >= 0:
                return min(T, t)
            return T  # for all t it's safe

        # solve quadratic inequality
        disc = B*B - 4*A*C
        if disc < 0:
            return T  # for all t it's safe
        sqrt_disc = np.sqrt(disc)
        t1 = (-B - sqrt_disc)/(2*A)
        t2 = (-B + sqrt_disc)/(2*A)
        if t1 >= 0:
            return min(t1, T)
        if t2 >= 0:
            return min(t2, T)
        return T


class PlanEfficiencyMetric:

    @staticmethod
    def calc_cost(ego_vel: float, target_vel: float, vel_profile: VelocityProfile) -> float:
        """
        Calculate efficiency cost for a planned following car or lane.
        Do it by calculation of average velocity during achieving the followed car and then following it with
        constant velocity. Total considered time is given by time_horizon.
        :param ego_vel: [m/s] ego initial velocity
        :param target_vel: [m/s] target car / lane velocity
        :param vel_profile: velocity profile
        :return: the efficiency cost
        """
        profile_time = vel_profile.t1 + vel_profile.t2 + vel_profile.t3

        avg_vel = (vel_profile.t1 * 0.5 * (ego_vel + vel_profile.v_mid) +
                   vel_profile.t2 * vel_profile.v_mid +
                   vel_profile.t3 * 0.5 * (vel_profile.v_mid + target_vel)) / profile_time

        print('profile_time=%f avg_vel=%f t1=%f t2=%f t3=%f v_mid=%f obj_vel=%f' %
              (profile_time, avg_vel, vel_profile.t1, vel_profile.t2, vel_profile.t3,
               vel_profile.v_mid, target_vel))

        efficiency_cost = EfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([avg_vel]))[0]
        return EFFICIENCY_COST_WEIGHT * efficiency_cost * profile_time / WERLING_TIME_RESOLUTION


class PlanComfortMetric:
    @staticmethod
    def calc_cost(vel_profile: VelocityProfile, lat_dist: float, aggressiveness_level: int,
                  max_lateral_time: float):
        # TODO: if T is known, calculate jerks analytically
        lat_time = min(max_lateral_time, lat_dist / PLAN_LATERAL_VELOCITY)
        if lat_time <= 0:
            return np.inf
        lat_acc = 4 * lat_dist / (lat_time * lat_time)  # lateral acceleration and then deceleration
        lat_cost = lat_time * (lat_acc ** 4) * LAT_ACC_TO_COST_FACTOR * LAT_JERK_COST_WEIGHT

        lon_acc = AGGRESSIVENESS_TO_LON_ACC[aggressiveness_level]
        lon_cost = (vel_profile.t1 + vel_profile.t3) * lon_acc * lon_acc * LON_ACC_TO_COST_FACTOR * \
                   LON_JERK_COST_WEIGHT
        return lat_cost + lon_cost


class PlanRightLaneMetric:
    @staticmethod
    def calc_cost(time_period: float, lane_idx: int) -> float:
        return RIGHT_LANE_COST_WEIGHT * lane_idx * time_period / WERLING_TIME_RESOLUTION


class ValueFunction:
    """
    value function approximation for debugging purposes
    """
    @staticmethod
    def calc_cost(time_period: float, vel: float, lane: int) -> float:
        efficiency_cost = EfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([vel]))[0] * \
                          time_period / WERLING_TIME_RESOLUTION
        right_lane_cost = lane * time_period / WERLING_TIME_RESOLUTION
        cost = efficiency_cost * EFFICIENCY_COST_WEIGHT + right_lane_cost * RIGHT_LANE_COST_WEIGHT
        return cost
