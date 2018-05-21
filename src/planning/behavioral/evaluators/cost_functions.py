import numpy as np
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, ActionSpec

from decision_making.src.global_constants import BP_RIGHT_LANE_COST_WEIGHT, BP_EFFICIENCY_COST_WEIGHT, \
    LAT_JERK_COST_WEIGHT, LON_JERK_COST_WEIGHT, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, \
    BP_METRICS_LANE_DEVIATION_COST_WEIGHT, EFFICIENCY_COST_DERIV_ZERO_DESIRED_RATIO
from decision_making.src.planning.behavioral.evaluators.velocity_profile import VelocityProfile
from decision_making.src.planning.types import FS_SA, FS_SV, FS_SX, FS_DA, FS_DV, FS_DX
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D


class BP_EfficiencyMetric:
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
        deviation1 = BP_EfficiencyMetric._calc_avg_vel_deviation(vel_profile.v_init, vel_profile.v_mid, des_vel)
        deviation2 = abs(vel_profile.v_mid - des_vel)
        deviation3 = BP_EfficiencyMetric._calc_avg_vel_deviation(vel_profile.v_mid, vel_profile.v_tar, des_vel)

        avg_deviation = (vel_profile.t1 * deviation1 + vel_profile.t2 * deviation2 + vel_profile.t3 * deviation3) / \
                        profile_time
        avg_vel = des_vel - avg_deviation

        # print('profile_time=%f avg_vel=%f t1=%f t2=%f t3=%f v_mid=%f obj_vel=%f' %
        #       (profile_time, avg_vel, vel_profile.t1, vel_profile.t2, vel_profile.t3,
        #        vel_profile.v_mid, target_vel))

        efficiency_cost = BP_EfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([avg_vel]))[0]
        return BP_EFFICIENCY_COST_WEIGHT * efficiency_cost * profile_time

    @staticmethod
    def calc_pointwise_cost_for_velocities(vel: np.array) -> np.array:
        """
        calculate efficiency (velocity) cost by parabola function
        C(vel) = P(v) = a*v*v + b*v, where v = abs(1 - vel/vel_des), C(vel_des) = 0, C(0) = 1, C'(0)/C'(vel_des) = r
        :param vel: input velocities: either 1D or 2D array
        :return: array of size vel.shape of efficiency costs per point
        """
        r = EFFICIENCY_COST_DERIV_ZERO_DESIRED_RATIO  # C'(0)/C'(vel_des) = P'(1)/P'(0)
        # the following two lines are the solution of two equations on a and b: P(1) = 1, P'(1)/P'(0) = r
        a = (r-1)/(r+1)
        b = 2/(r+1)
        normalized_vel = np.absolute(1 - vel / BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED)
        costs = normalized_vel * (a * normalized_vel + b)
        return costs

    @staticmethod
    def _calc_avg_vel_deviation(v1: float, v2: float, des_vel: float) -> float:
        if (v1 - des_vel) * (v2 - des_vel) >= 0:  # v1 and v2 are on the same side of des_vel
            return abs(0.5 * (v1 + v2) - des_vel)  # simple average deviation
        else:  # v1 and v2 are on different sides of des_vel
            return 0.5 * ((v1 - des_vel) ** 2 + (v2 - des_vel) ** 2) / abs(v2 - v1)


class BP_ComfortMetric:
    @staticmethod
    def calc_cost(ego_fstate: np.array, spec: ActionSpec, T_d: float) -> [float, float]:
        """
        Calculate comfort cost for lateral and longitudinal movement
        :param ego_fstate: initial ego Frenet state
        :param spec: action spec
        :param T_d_max: maximal permitted lateral movement time, bounded according to the safety
        :param target_lat: target latitude
        :return: comfort cost in units of the general performance metrics cost
        """
        if T_d <= 0 < spec.td:  # infeasible lateral movement
            return 0, np.inf

        # # calculate factor between the comfortable lateral movement time and the required lateral movement time
        # if T_d <= T_d_max:
        #     time_factor = 1  # comfortable lane change
        # else:
        #     time_factor = T_d / max(np.finfo(np.float16).eps, T_d_max)
        # # acc_factor = np.sqrt((AGGRESSIVENESS_TO_LAT_ACC[aggressiveness.value] / AGGRESSIVENESS_TO_LAT_ACC[0]) ** 5)
        # lat_time = min(T_d, T_d_max)
        # change_lat_dir_time = 0
        # if lat_vel_to_tar < 0:  # only negative lateral velocity increases the cost
        #     change_lat_dir_time = CHANGE_LAT_VEL_WEIGHT * lat_vel_to_tar**2
        # lat_cost = (lat_time + change_lat_dir_time) * (time_factor ** 5) * LAT_ACC_TO_JERK_FACTOR * LAT_JERK_COST_WEIGHT

        lat_cost = lon_cost = 0
        # lateral jerk
        if spec.td > 0:
            lat_jerk = QuinticPoly1D.cumulative_jerk_from_constraints(
                ego_fstate[FS_DA], ego_fstate[FS_DV], 0, spec.d - ego_fstate[FS_DX], T_d)
            lat_cost = lat_jerk * LAT_JERK_COST_WEIGHT

        # longitudinal jerk
        if spec.t > 0:
            lon_jerk = QuinticPoly1D.cumulative_jerk_from_constraints(
                ego_fstate[FS_SA], ego_fstate[FS_SV], spec.v, spec.s - ego_fstate[FS_SX], spec.t)
            lon_cost = lon_jerk * LON_JERK_COST_WEIGHT
            # print('lon: a0=%.2f v0=%.2f vT=%.2f s=%.2f t=%.2f' %
            # (ego_fstate[FS_SA], ego_fstate[FS_SV], spec.v, spec.s - ego_fstate[FS_SX], spec.t))

        # if vel_profile.t1 + vel_profile.t3 > 0:
        #     jerk1 = jerk2 = 0
        #     if vel_profile.t1 > 0:
        #         jerk1 = LON_ACC_TO_JERK_FACTOR * ((vel_profile.v_mid - vel_profile.v_init) ** 2) / (vel_profile.t1 ** 3)
        #     if vel_profile.t3 > 0:
        #         jerk2 = LON_ACC_TO_JERK_FACTOR * ((vel_profile.v_tar - vel_profile.v_mid) ** 2) / (vel_profile.t3 ** 3)
        #     lon_cost = (jerk1 + jerk2) * LON_JERK_COST_WEIGHT

        return lon_cost, lat_cost


class BP_RightLaneMetric:
    @staticmethod
    def calc_cost(time_period: float, lane_idx: int) -> float:
        return BP_RIGHT_LANE_COST_WEIGHT * lane_idx * time_period


class BP_LaneDeviationMetric:
    @staticmethod
    def calc_cost(lat_dev: float) -> float:
        return BP_METRICS_LANE_DEVIATION_COST_WEIGHT * lat_dev * lat_dev
