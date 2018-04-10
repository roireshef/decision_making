import numpy as np

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, SAFE_DIST_TIME_DELAY, \
    WERLING_TIME_RESOLUTION, AGGRESSIVENESS_TO_LON_ACC, AGGRESSIVENESS_TO_LAT_ACC, ACC_TO_COST_FACTOR, \
    RIGHT_LANE_COST_WEIGHT, EFFICIENCY_COST_WEIGHT, LAT_JERK_COST_WEIGHT, LON_JERK_COST_WEIGHT
from decision_making.src.planning.performance_metrics.cost_functions import EfficiencyMetric


class VelocityProfile:
    def __init__(self, init_vel: float, acc_time: float, mid_vel: float, mid_time: float, dec_time: float,
                 end_vel: float):
        self.init_vel = init_vel  # initial ego velocity
        self.acc_time = acc_time  # acceleration time period
        self.mid_vel = mid_vel    # maximal velocity after acceleration
        self.mid_time = mid_time  # time period for going with maximal velocity
        self.dec_time = dec_time  # deceleration time
        self.end_vel = end_vel    # end velocity


class PlanEfficiencyMetric:

    @staticmethod
    def calc_cost(ego_vel: float, target_vel: float, vel_profile: VelocityProfile, time_horizon: float) -> float:
        """
        Calculate efficiency cost for a planned following car or lane.
        Do it by calculation of average velocity during achieving the followed car and then following it with
        constant velocity. Total considered time is given by time_horizon.
        :param ego_vel: [m/s] ego initial velocity
        :param target_vel: [m/s] target car / lane velocity
        :param vel_profile: velocity profile
        :param time_horizon: [sec] considered time horizon for average velocity and cost calculation
        :return: the efficiency cost
        """
        profile_time = vel_profile.acc_time + vel_profile.mid_time + vel_profile.dec_time

        avg_vel = (vel_profile.acc_time * 0.5 * (ego_vel + vel_profile.mid_vel) +
                   vel_profile.mid_time * vel_profile.mid_vel +
                   vel_profile.dec_time * 0.5 * (vel_profile.mid_vel + target_vel)) / profile_time

        print('profile_time=%f avg_vel=%f t1=%f t2=%f t3=%f v_mid=%f obj_vel=%f' %
              (profile_time, avg_vel, vel_profile.acc_time, vel_profile.mid_time, vel_profile.dec_time,
               vel_profile.mid_vel, target_vel))

        efficiency_cost = EfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([avg_vel]))[0]
        return EFFICIENCY_COST_WEIGHT * efficiency_cost * profile_time / WERLING_TIME_RESOLUTION

    @staticmethod
    def calc_velocity_profile(init_lon: float, init_vel: float, obj_lon: float, target_vel: float, target_acc: float,
                              aggressiveness_level: int) -> VelocityProfile:
        """
        calculate velocities profile for semantic action: either following car or following lane
        :param init_lon: [m] initial longitude of ego
        :param init_vel: [m/s] initial velocity of ego
        :param obj_lon: [m] initial longitude of followed object (None if follow lane)
        :param target_vel: [m/s] followed object's velocity or target velocity for follow lane
        :param target_acc: [m/s^2] followed object's acceleration
        :param aggressiveness_level: [int] attribute of the semantic action
        :return: VelocityProfile class includes times and velocities
        """
        des_acc = AGGRESSIVENESS_TO_LON_ACC[aggressiveness_level]  # desired acceleration
        max_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
        if obj_lon is not None:  # follow car
            s = obj_lon - init_lon - SAFE_DIST_TIME_DELAY * target_vel - 5  # TODO: replace 5 by cars' half sizes sum
            acc_time, mid_vel, mid_time, dec_time = \
                PlanEfficiencyMetric._calc_velocity_profile_follow_car(init_vel, des_acc, max_vel, s,
                                                                       target_vel, target_acc)
        else:  # follow lane
            dec_time = mid_time = 0
            mid_vel = target_vel
            acc_time = abs(target_vel - init_vel) / des_acc
        return VelocityProfile(init_vel, acc_time, mid_vel, mid_time, dec_time, target_vel)

    @staticmethod
    def _calc_velocity_profile_follow_car(v_init: float, a: float, v_max: float, dist: float, v_tar: float,
                                          a_tar: float) -> [float, float, float, float]:
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
        return: t1, v_mid, t2, t3 (time periods of the 3 profile segments and maximal achieved velocity)
        """
        print('CALC PROFILE: v_init=%f dist=%f' % (v_init, dist))

        ILLEGAL_VEL = -1000  # illegal velocity creates "infinite" cost
        v_init_rel = v_init - v_tar  # relative velocity; may be negative
        v_max_rel = max(v_max - v_tar, max(v_init_rel, 1.))  # let max_vel > end_vel to enable reaching the car
        if np.math.isclose(a, a_tar, rel_tol=1e-02) or np.math.isclose(a, -a_tar, rel_tol=1e-02):
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
                return 0, ILLEGAL_VEL, 1, 0  # illegal action
            v_mid_rel = -np.sqrt(v_mid_rel_sqr)
            t1 = (v_init_rel - v_mid_rel) / (a - a_tar)   # deceleration time
            t3 = -v_mid_rel / (a + a_tar)                 # acceleration time
            if t1 < 0 or t3 < 0:
                print('BAD: negative t1 or t3 or v_mid: v_mid_rel=%f t1=%f t3=%f' % (v_mid_rel, t1, t3))
                return 0, ILLEGAL_VEL, 1, 0  # illegal action

        if v_init + t1 * a <= v_max or dist < 0:  # ego does not reach max_vel, then t2 = 0
            if t3 < 0 or v_mid_rel + v_tar < 0:  # negative t3 or negative v_mid
                print('BAD: negative t3 or v_mid: t3=%f v_mid_rel=%f' % (t3, v_mid_rel))
                return 0, ILLEGAL_VEL, 1, 0  # illegal action
            return t1, v_mid_rel + v_tar, 0, t3

        # from now: ego reaches max_vel, such that t2 > 0

        t1 = (v_max - v_init) / a  # acceleration time
        if a_tar == 0:  # a simple case: the followed car has constant velocity
            t3 = v_max_rel / a  # deceleration time
            dist_mid = dist - (2 * v_max_rel * v_max_rel - v_init_rel * v_init_rel) / (2 * a)
            t2 = max(0., dist_mid / v_max_rel)  # constant velocity (max_vel) time
            return t1, v_max, t2, t3

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
        discriminant = vm1*vm1 * a * a + a * a_tar * c  # discriminant of the quadratic equation
        if discriminant < 0:
            print('BAD: general case: det < 0')
            return 0, ILLEGAL_VEL, 1, 0  # illegal action
        t2 = (vm1 * a + np.sqrt(discriminant)) / (a * a_tar)
        vm2 = vm1 - a_tar*t2
        t3 = vm2 / (a + a_tar)
        return t1, v_max, t2, t3


class PlanComfortMetric:

    @staticmethod
    def calc_cost(vel_profile: VelocityProfile, change_lane: bool, aggressiveness_level: int):
        # TODO: if T is known, calculate jerks analytically

        lat_acc = AGGRESSIVENESS_TO_LAT_ACC[aggressiveness_level]
        lane_width = 3.6  # TODO: take it from the map
        if change_lane:
            lane_change_time = 2 * np.sqrt(lane_width / lat_acc)
            lat_cost = lane_change_time * lat_acc * lat_acc * ACC_TO_COST_FACTOR * LAT_JERK_COST_WEIGHT
        else:
            lat_cost = 0

        lon_acc = AGGRESSIVENESS_TO_LON_ACC[aggressiveness_level]
        lon_cost = (vel_profile.acc_time + vel_profile.dec_time) * lon_acc * lon_acc * ACC_TO_COST_FACTOR * \
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
