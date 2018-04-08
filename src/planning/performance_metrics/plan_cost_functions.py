import numpy as np

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, SAFE_DIST_TIME_DELAY, \
    WERLING_TIME_RESOLUTION, AGGRESSIVENESS_TO_LON_ACC, AGGRESSIVENESS_TO_LAT_ACC, ACC_TO_COST_FACTOR
from decision_making.src.planning.performance_metrics.cost_functions import EfficiencyMetric


class VelocityProfile:
    def __init__(self, init_vel: float, acc_time: float, mid_vel: float, mid_time: float, dec_time: float,
                 end_vel: float):
        self.init_vel = init_vel
        self.acc_time = acc_time
        self.mid_vel = mid_vel
        self.mid_time = mid_time
        self.dec_time = dec_time
        self.end_vel = end_vel


class PlanEfficiencyMetric:

    @staticmethod
    def calc_cost(ego_vel: float, obj_vel: float, vel_profile: VelocityProfile, time_horizon: float) -> float:
        """
        Calculate efficiency cost for a planned following car or lane.
        Do it by calculation of average velocity during achieving the followed car and then following it with
        constant velocity. Total considered time is given by time_horizon.
        :param ego_vel: [m/s] ego initial velocity
        :param obj_vel: [m/s] target car's constant velocity
        :param vel_profile: velocity profile
        :param time_horizon: [sec] considered time horizon for average velocity and cost calculation
        :return: the efficiency cost
        """
        profile_time = vel_profile.acc_time + vel_profile.mid_time + vel_profile.dec_time
        follow_time = max(0, time_horizon - profile_time)  # the time ego follows the car with it's velocity

        avg_vel = (vel_profile.acc_time * 0.5 * (ego_vel + vel_profile.mid_vel) +
                   vel_profile.mid_time * vel_profile.mid_vel +
                   vel_profile.dec_time * 0.5 * (vel_profile.mid_vel + obj_vel) +
                   follow_time * obj_vel) / \
                  (profile_time + follow_time)

        print('t_follow=%f avg_vel=%f t1=%f t2=%f t3=%f v_mid=%f obj_vel=%f' %
              (follow_time, avg_vel, vel_profile.acc_time, vel_profile.mid_time, vel_profile.dec_time,
               vel_profile.mid_vel, obj_vel))

        efficiency_cost = EfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([avg_vel]))[0]
        return efficiency_cost * time_horizon / WERLING_TIME_RESOLUTION

    @staticmethod
    def calc_velocity_profile(init_lon: float, init_vel: float, obj_lon: float, target_vel: float,
                              aggressiveness_level: int) -> VelocityProfile:
        des_acc = AGGRESSIVENESS_TO_LON_ACC[aggressiveness_level]  # desired acceleration
        max_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
        if obj_lon is not None:  # follow car
            s = obj_lon - init_lon - SAFE_DIST_TIME_DELAY * target_vel - 5  # TODO: replace 5 by cars' half sizes sum
            acc_time, mid_vel, mid_time, dec_time = \
                PlanEfficiencyMetric._calc_velocity_profile_follow_car(init_vel, des_acc, max_vel, s, target_vel)
        else:  # follow lane
            dec_time = mid_time = 0
            mid_vel = target_vel
            acc_time = abs(target_vel - init_vel) / des_acc
        return VelocityProfile(init_vel, acc_time, mid_vel, mid_time, dec_time, target_vel)

    @staticmethod
    def _calc_velocity_profile_follow_car(init_vel: float, acc: float, max_vel: float, tot_dist: float, end_vel: float) -> \
            [float, float, float, float]:
        """
        Given start & end velocities and distance to the followed car, calculate velocity profile:
            1. acceleration to v_mid,
            2. moving by v_max (may be empty if v_mid < v_max),
            3. deceleration to v1.
        In each velocity segment the acceleration is constant.
        Here we assume that the target car moves with constant velocity.
        :param init_vel: start velocity
        :param acc: maximal acceleration
        :param max_vel: maximal desired velocity of ego
        :param tot_dist: initial distance to the safe location from the target
        :param end_vel: target object velocity
        return: acc_time, mid_vel, mid_time, dec_time (time periods of the 3 velocity segments and maximal achieved velocity)
        """
        init_vel_rel = init_vel - end_vel  # relative velocity; may be negative
        max_vel_rel = max(max_vel - end_vel, max(init_vel_rel, 1.))  # let max_vel > end_vel to enable reaching the car
        if init_vel_rel > 0 and init_vel_rel * init_vel_rel > 2 * acc * abs(tot_dist):  # the braking should be stronger than acc
            return 0, -1000, 1, 0  # illegal action creates infinite cost
        else:  # acceleration = acc
            mid_vel_rel_sqr = init_vel_rel * init_vel_rel / 2 + acc * tot_dist  # since mv^2/2a - v^2/2a + mv^2/2a = tot_dist
            if mid_vel_rel_sqr > 0:
                mid_vel_rel = np.sqrt(mid_vel_rel_sqr)
                mid_vel_rel = max(mid_vel_rel, init_vel_rel)
            else:  # the target is behind ego, then just slow down
                mid_vel_rel = init_vel_rel
            if mid_vel_rel <= max_vel_rel:  # ego does not reach max_vel, then mid_time = 0
                acc_time = (mid_vel_rel - init_vel_rel) / acc
                dec_time = mid_vel_rel / acc
                return acc_time, mid_vel_rel + end_vel, 0, dec_time
            else:  # ego reaches max_vel and moves mid_time
                acc_time = (max_vel_rel - init_vel_rel) / acc
                dec_time = max_vel_rel / acc
                mid_dist = tot_dist - (2 * max_vel_rel * max_vel_rel - init_vel_rel * init_vel_rel) / (2 * acc)
                mid_time = max(0., mid_dist / max_vel_rel)
                return acc_time, max_vel, mid_time, dec_time


class PlanComfortMetric:

    @staticmethod
    def calc_cost(vel_profile: VelocityProfile, change_lane: bool, aggressiveness_level: int):
        # TODO: if T is known, calculate jerk analytically
        lon_acc = AGGRESSIVENESS_TO_LON_ACC[aggressiveness_level]
        lat_acc = AGGRESSIVENESS_TO_LAT_ACC[aggressiveness_level]
        lane_width = 3.6
        if change_lane:
            lane_change_time = 2 * np.sqrt(lane_width / lat_acc)
            lat_cost = lane_change_time * lat_acc * ACC_TO_COST_FACTOR
        else:
            lat_cost = 0
        lon_cost = (vel_profile.acc_time + vel_profile.dec_time) * lon_acc * ACC_TO_COST_FACTOR
        return lat_cost + lon_cost
