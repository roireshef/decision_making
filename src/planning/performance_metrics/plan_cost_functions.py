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
    def _calc_velocity_profile_follow_car(init_vel: float, acc: float, max_vel: float, tot_dist: float, end_vel: float,
                                          end_acc: float) -> \
            [float, float, float, float]:
        """
        Given start & end velocities and distance to the followed car, calculate velocity profile:
            1. acceleration/deceleration to mid_vel,
            2. moving by max_vel (empty if mid_vel < max_vel or in case of initial deceleration),
            3. deceleration/acceleration to end_vel.
        In each velocity segment the acceleration is constant.
        Here we assume that the target car moves with constant velocity.
        :param init_vel: start velocity
        :param acc: absolute ego acceleration in the first and the last profile segment
        :param max_vel: maximal desired velocity of ego
        :param tot_dist: initial distance to the safe location from the target
        :param end_vel: target object velocity
        :param end_acc: target object acceleration
        return: t1, mid_vel, t2, t3 (time periods of the 3 profile segments and maximal achieved velocity)
        """
        print('CALC PROFILE: init_vel=%f tot_dist=%f' % (init_vel, tot_dist))

        ILLEGAL_VEL = -1000  # illegal velocity creates "infinite" cost
        init_vel_rel = init_vel - end_vel  # relative velocity; may be negative
        max_vel_rel = max(max_vel - end_vel, max(init_vel_rel, 1.))  # let max_vel > end_vel to enable reaching the car
        if np.math.isclose(acc, end_acc, rel_tol=1e-03) or np.math.isclose(acc, -end_acc, rel_tol=1e-03):
            acc -= 0.1  # slightly change the acceleration to prevent instability in calculations

        # here we use formula (mv^2 - v^2)/2(a-a1) + mv^2/2(a+a1) = tot_dist
        mid_vel_rel_sqr = (init_vel_rel * init_vel_rel / 2 + tot_dist * (acc - end_acc)) * (acc + end_acc) / acc
        try_opposite_order = False
        t1 = t3 = 0
        mid_vel_rel = init_vel_rel
        if mid_vel_rel_sqr >= 0:
            mid_vel_rel = np.sqrt(mid_vel_rel_sqr)
            t1 = (mid_vel_rel - init_vel_rel) / (acc - end_acc)
            t3 = mid_vel_rel / (acc + end_acc)
            if t1 < 0 or t3 < 0:
                try_opposite_order = True
        else:  # the target is unreachable
            try_opposite_order = True

        # try_opposite_order: first deceleration, then acceleration
        # here we use formula (v^2 - mv^2)/2(a+a1) - mv^2/2(a-a1) = tot_dist
        if try_opposite_order:  # first deceleration, then acceleration
            mid_vel_rel_sqr = (init_vel_rel * init_vel_rel / 2 - tot_dist * (acc + end_acc)) * (acc - end_acc) / acc
            if mid_vel_rel_sqr < 0:  # the target is unreachable
                print('BAD: negative mid_vel_rel_sqr=%f tot_dist=%f' % (mid_vel_rel_sqr, tot_dist))
                return 0, ILLEGAL_VEL, 1, 0  # illegal action
            mid_vel_rel = -np.sqrt(mid_vel_rel_sqr)
            t1 = (init_vel_rel - mid_vel_rel) / (acc - end_acc)   # deceleration time
            t3 = -mid_vel_rel / (acc + end_acc)                   # acceleration time
            if t1 < 0 or t3 < 0:
                print('BAD: negative t1 or t3 or mid_vel: mid_vel_rel=%f t1=%f t3=%f' % (mid_vel_rel, t1, t3))
                return 0, ILLEGAL_VEL, 1, 0  # illegal action

        if init_vel + t1 * acc <= max_vel or tot_dist < 0:  # ego does not reach max_vel, then t2 = 0
            if t3 < 0 or mid_vel_rel + end_vel < 0:  # negative t3 or mid_vel
                print('BAD: negative t3 or mid_vel: t3=%f mid_vel_rel=%f' % (t3, mid_vel_rel))
                return 0, ILLEGAL_VEL, 1, 0  # illegal action
            return t1, mid_vel_rel + end_vel, 0, t3

        # ego reaches max_vel and moves with this velocity mid_time
        t1 = (max_vel - init_vel) / acc
        if end_acc == 0:
            t3 = max_vel_rel / acc
            mid_dist = tot_dist - (2 * max_vel_rel * max_vel_rel - init_vel_rel * init_vel_rel) / (2 * acc)
            t2 = max(0., mid_dist / max_vel_rel)
            return t1, max_vel, t2, t3

        # Notations:
        #   v is initial relative velocity
        #   a > 0 is ego acceleration, a1 target object acceleration
        #   vm1 is relative velocity of ego immediately after acceleration
        #   vm2 is relative velocity of ego immediately before the deceleration, vm2 = vm1 - a1*t2
        # Quadratic equation: tot_dist is the sum of 3 distances:
        #   acceleration distance     max_vel distance     deceleration distance
        #   (vm1^2 - v^2)/2(a-a1)  +  (vm1+vm2)/2 * t2  +  vm2^2/2(a+a1) = tot_dist
        (v, a, a1) = (init_vel_rel, acc, end_acc)
        vm1 = v + (a - a1) * t1
        # after substitution of vm2 = vm1 - a1*t and simplification, solve quadratic equation on t2:
        # a*a1*t^2 - 2*vm1*a*t2 - (vm1^2 + 2*(a+a1)*((vm1^2 - v^2) / 2*(a-a1) - tot_dist)) = 0
        c = vm1*vm1 + 2*(a+a1) * ((vm1*vm1 - v*v)/(2*(a-a1)) - tot_dist)  # free coefficient
        det = vm1*vm1*a*a + a*a1*c
        if det < 0:
            print('BAD: general case: det < 0')
            return 0, ILLEGAL_VEL, 1, 0  # illegal action
        t2 = (vm1*a + np.sqrt(det)) / (a*a1)
        vm2 = vm1 - a1*t2
        t3 = vm2 / (a + a1)
        return t1, max_vel, t2, t3


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
