import numpy as np

from decision_making.src.global_constants import WERLING_TIME_RESOLUTION, LON_ACC_TO_COST_FACTOR, \
    LAT_ACC_TO_COST_FACTOR, RIGHT_LANE_COST_WEIGHT, EFFICIENCY_COST_WEIGHT, LAT_JERK_COST_WEIGHT, LON_JERK_COST_WEIGHT, \
    BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, PLAN_LANE_DEVIATION_COST_WEIGHT
from decision_making.src.planning.performance_metrics.cost_functions import EfficiencyMetric
from decision_making.src.planning.performance_metrics.behavioral.velocity_profile import VelocityProfile


class PlanEfficiencyMetric:
    @staticmethod
    def calc_cost(vel_profile: VelocityProfile) -> float:
        """
        Calculate efficiency cost for a planned following car or lane.
        Do it by calculation of average velocity during achieving the followed car and then following it with
        constant velocity. Total considered time is given by time_horizon.
        :param vel_profile: velocity profile
        :return: the efficiency cost
        """
        profile_time = vel_profile.total_time()
        des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
        deviation1 = PlanEfficiencyMetric._calc_avg_vel_deviation(vel_profile.v_init, vel_profile.v_mid, des_vel)
        deviation2 = abs(vel_profile.v_mid - des_vel)
        deviation3 = PlanEfficiencyMetric._calc_avg_vel_deviation(vel_profile.v_mid, vel_profile.v_tar, des_vel)

        avg_deviation = (vel_profile.t1 * deviation1 + vel_profile.t2 * deviation2 + vel_profile.t3 * deviation3) / \
                        profile_time
        avg_vel = des_vel - avg_deviation

        # print('profile_time=%f avg_vel=%f t1=%f t2=%f t3=%f v_mid=%f obj_vel=%f' %
        #       (profile_time, avg_vel, vel_profile.t1, vel_profile.t2, vel_profile.t3,
        #        vel_profile.v_mid, target_vel))

        efficiency_cost = EfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([avg_vel]))[0]
        return EFFICIENCY_COST_WEIGHT * efficiency_cost * profile_time / WERLING_TIME_RESOLUTION

    @staticmethod
    def _calc_avg_vel_deviation(v1: float, v2: float, des_vel: float) -> float:
        if (v1 - des_vel) * (v2 - des_vel) >= 0:  # v1 and v2 are on the same side of des_vel
            return abs(0.5 * (v1 + v2) - des_vel)  # simple average deviation
        else:  # v1 and v2 are on different sides of des_vel
            return 0.5 * ((v1 - des_vel) ** 2 + (v2 - des_vel) ** 2) / abs(v2 - v1)


class PlanComfortMetric:
    @staticmethod
    def calc_cost(vel_profile: VelocityProfile, comfortable_lat_time, max_lat_time: float):
        """
        Calculate comfort cost for lateral and longitudinal movement
        :param vel_profile: longitudinal velocity profile
        :param comfortable_lat_time: comfortable lateral movement time
        :param max_lat_time: maximal permitted lateral movement time, bounded according to the safety
        :return: comfort cost in units of the general performance metrics cost
        """
        # TODO: if T is known, calculate jerks analytically
        if max_lat_time <= 0:
            return np.inf

        # calculate factor between the comfortable lateral movement time and the required lateral movement time
        if comfortable_lat_time <= max_lat_time:
            time_factor = 1  # comfortable lane change
        else:
            time_factor = comfortable_lat_time / max(np.finfo(np.float16).eps, max_lat_time)
        lat_time = min(comfortable_lat_time, max_lat_time)

        lat_cost = lat_time * (time_factor ** 6) * LAT_ACC_TO_COST_FACTOR * LAT_JERK_COST_WEIGHT

        # longitudinal cost
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


class PlanLaneDeviationMetric:
    @staticmethod
    def calc_cost(lat_time_period: float) -> float:
        return PLAN_LANE_DEVIATION_COST_WEIGHT * lat_time_period / WERLING_TIME_RESOLUTION
