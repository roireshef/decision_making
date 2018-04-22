import numpy as np

from decision_making.src.global_constants import WERLING_TIME_RESOLUTION, LON_ACC_TO_COST_FACTOR, \
    LAT_VEL_TO_COST_FACTOR, RIGHT_LANE_COST_WEIGHT, EFFICIENCY_COST_WEIGHT, LAT_JERK_COST_WEIGHT, LON_JERK_COST_WEIGHT
from decision_making.src.planning.performance_metrics.cost_functions import EfficiencyMetric
from decision_making.src.planning.performance_metrics.velocity_profile import VelocityProfile


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
        profile_time = vel_profile.total_time()

        avg_vel = (vel_profile.t1 * 0.5 * (ego_vel + vel_profile.v_mid) +
                   vel_profile.t2 * vel_profile.v_mid +
                   vel_profile.t3 * 0.5 * (vel_profile.v_mid + target_vel)) / profile_time

        # print('profile_time=%f avg_vel=%f t1=%f t2=%f t3=%f v_mid=%f obj_vel=%f' %
        #       (profile_time, avg_vel, vel_profile.t1, vel_profile.t2, vel_profile.t3,
        #        vel_profile.v_mid, target_vel))

        efficiency_cost = EfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([avg_vel]))[0]
        return EFFICIENCY_COST_WEIGHT * efficiency_cost * profile_time / WERLING_TIME_RESOLUTION


class PlanComfortMetric:
    @staticmethod
    def calc_cost(vel_profile: VelocityProfile, comfortable_lat_time, max_lat_time: float):
        # TODO: if T is known, calculate jerks analytically
        if max_lat_time <= 0:
            return np.inf

        if comfortable_lat_time <= max_lat_time:
            time_factor = 1  # comfortable lane change
        else:
            time_factor = comfortable_lat_time / max(np.finfo(np.float16).eps, max_lat_time)
        lat_time = min(comfortable_lat_time, max_lat_time)

        lat_cost = lat_time * (time_factor ** 6) * LAT_VEL_TO_COST_FACTOR * LAT_JERK_COST_WEIGHT

        acc1 = acc3 = 0
        if vel_profile.t1 > 0:
            acc1 = abs(vel_profile.v_mid - vel_profile.v_init) / vel_profile.t1
        if vel_profile.t3 > 0:
            acc3 = abs(vel_profile.v_tar - vel_profile.v_mid) / vel_profile.t3
        lon_cost = (vel_profile.t1 * (acc1**3) + vel_profile.t3 * (acc3**3)) * LON_ACC_TO_COST_FACTOR * \
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
