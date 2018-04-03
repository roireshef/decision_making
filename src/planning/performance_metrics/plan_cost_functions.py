import numpy as np

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, SAFE_DIST_TIME_DELAY, \
    WERLING_TIME_RESOLUTION
from decision_making.src.planning.performance_metrics.cost_functions import EfficiencyMetric


class PlanEfficiencyMetric:

    @staticmethod
    def calc_cost(ego_lon: float, ego_vel: float, obj_lon: float, obj_vel: float, time_horizon: float) -> float:
        """
        Calculate efficiency cost for a planned following car or lane.
        Do it by calculation of average velocity during achieving the followed car and then following it with
        constant velocity. Total considered time is given by time_horizon.
        :param ego_lon: [m] ego longitude
        :param ego_vel: [m/s] ego initial velocity
        :param obj_lon: [m] target car's longitude
        :param obj_vel: [m/s] target car's constant velocity
        :param time_horizon: [sec] considered time horizon for average velocity and cost calculation
        :return: the efficiency cost
        """
        des_acc = 1.5  # desired acceleration
        max_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
        if obj_lon is not None:  # follow car
            s = obj_lon - ego_lon - SAFE_DIST_TIME_DELAY * obj_vel - 5
            follow_vel = obj_vel
            t1, t2, t3, v_mid = \
                PlanEfficiencyMetric._calc_velocity_profile_follow_car(ego_vel, des_acc, max_vel, s, obj_vel)
        else:  # follow lane
            follow_vel = max_vel
            t1, t2, t3, v_mid = PlanEfficiencyMetric._calc_velocity_profile_follow_lane(ego_vel, des_acc, max_vel)
        t_follow = max(0, time_horizon - (t1 + t2 + t3))  # the time ego follows the car with it's velocity

        avg_vel = (t1 * 0.5 * (ego_vel + v_mid) + t2 * v_mid +
                   t3 * 0.5 * (v_mid + follow_vel) + t_follow * follow_vel) / (t1 + t2 + t3 + t_follow)

        # print('t_follow=%f avg_vel=%f' % (t_follow, avg_vel))

        efficiency_cost = EfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([avg_vel]))[0]
        return efficiency_cost * time_horizon / WERLING_TIME_RESOLUTION

    @staticmethod
    def _calc_velocity_profile_follow_car(v0: float, a: float, v_max: float, s: float, v1: float) -> \
            [float, float, float, float]:
        """
        Given start & end velocities and distance to the followed car, calculate velocity profile:
            1. acceleration to v_mid,
            2. moving by v_max (may be empty if v_mid < v_max),
            3. deceleration to v1.
        In each velocity segment the acceleration is constant.
        Here we assume that the target car moves with constant velocity.
        :param v0: start velocity
        :param a: maximal acceleration
        :param v_max: maximal desired velocity of ego
        :param s: initial distance to the safe location from the target
        :param v1: target object velocity
        return: t1, t2, t3, v_mid (time periods of the 3 velocity segments and maximal achieved velocity)
        """
        v = v0 - v1  # relative velocity; may be negative
        v_max_rel = max(v_max - v1, max(v, 1))  # let v_max > v1 to enable reaching the car
        if v > 0 and v * v > 2 * a * s:  # the braking should be stronger than a
            t1 = 2 * s / v
            return t1, 0, 0, v1
        else:  # acceleration = a
            v_mid_rel = np.sqrt(v * v / 2 + a * s)  # since vm^2/2a - v^2/2a + vm^2/2a = s
            if v_mid_rel <= v_max_rel:  # to high vel
                t1 = (v_mid_rel - v) / a
                t3 = v_mid_rel / a
                return t1, 0, t3, v_mid_rel + v1
            else:  # ego reaches v_max and moves t2 sec
                t1 = (v_max_rel - v) / a
                t3 = v_max_rel / a
                s2 = s - (2*v_max_rel*v_max_rel - v*v) / (2*a)
                t2 = s2 / v_max_rel
                return t1, t2, t3, v_max

    @staticmethod
    def _calc_velocity_profile_follow_lane(v0: float, a: float, v_max: float) -> [float, float, float, float]:
        """
        given start and target velocities, calculate velocity profile
        :param v0: start velocity
        :param a: acceleration
        :param v_max: target velocity
        return: t1, t2, t3, v_mid
        """
        return abs(v_max - v0) / a, 0, 0, v_max

