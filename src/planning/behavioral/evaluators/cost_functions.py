import numpy as np
from decision_making.src.planning.behavioral.data_objects import ActionSpec

from decision_making.src.global_constants import BP_RIGHT_LANE_COST_WEIGHT, BP_EFFICIENCY_COST_WEIGHT, \
    LAT_JERK_COST_WEIGHT, LON_JERK_COST_WEIGHT, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, \
    BP_METRICS_LANE_DEVIATION_COST_WEIGHT, BP_EFFICIENCY_COST_DERIV_ZERO_DESIRED_RATIO
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

        avg_deviation = (vel_profile.t_first * deviation1 + vel_profile.t_flat * deviation2 + vel_profile.t_last * deviation3) / \
                        profile_time
        avg_vel = des_vel - avg_deviation

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
        r = BP_EFFICIENCY_COST_DERIV_ZERO_DESIRED_RATIO  # C'(0)/C'(vel_des) = P'(1)/P'(0)
        # the following two lines are the solution of two equations on a and b: P(1) = 1, P'(1)/P'(0) = r
        coef_2 = (r-1)/(r+1)    # coefficient of x^2
        coef_1 = 2/(r+1)        # coefficient of x
        normalized_vel = np.absolute(1 - vel / BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED)
        costs = normalized_vel * (coef_2 * normalized_vel + coef_1)
        return costs

    @staticmethod
    def _calc_avg_vel_deviation(v_init: float, v_end: float, v_des: float) -> float:
        """
        given a velocity segment, calculate average deviation from the desired velocity
        :param v_init: [m/s] initial segment velocity
        :param v_end: [m/s] terminal segment velocity
        :param v_des: [m/s] desired velocity
        :return: [m/s] average deviation
        """
        if (v_init - v_des) * (v_end - v_des) >= 0:  # v1 and v2 are on the same side of des_vel
            return abs(0.5 * (v_init + v_end) - v_des)  # simple average deviation
        else:  # v1 and v2 are on different sides of des_vel
            return 0.5 * ((v_init - v_des) ** 2 + (v_end - v_des) ** 2) / abs(v_end - v_init)


class BP_ComfortMetric:
    @staticmethod
    def calc_cost(ego_fstate: np.array, spec: ActionSpec, T_d: float) -> [float, float]:
        """
        Calculate comfort cost for lateral and longitudinal movement
        :param ego_fstate: initial ego Frenet state
        :param spec: action spec
        :param T_d: maximal permitted lateral movement time, bounded according to the safety
        :return: comfort cost in units of the general performance metrics cost
        """
        lat_cost = lon_cost = 0
        # lateral jerk
        if 0. < T_d < np.inf:
            dist = spec.d - ego_fstate[FS_DX]
            lat_jerk1 = QuinticPoly1D.cumulative_jerk_from_constraints(
                ego_fstate[FS_DA], ego_fstate[FS_DV], 0, dist, T_d)
            lat_jerk2 = np.inf
            if spec.t > 0:
                lat_jerk2 = QuinticPoly1D.cumulative_jerk_from_constraints(
                    ego_fstate[FS_DA], ego_fstate[FS_DV], 0, dist, min(2*T_d, spec.t))
            lat_cost = min(lat_jerk1, lat_jerk2) * LAT_JERK_COST_WEIGHT

        # longitudinal jerk
        if spec.t > 0:
            lon_jerk = QuinticPoly1D.cumulative_jerk_from_constraints(
                ego_fstate[FS_SA], ego_fstate[FS_SV], spec.v, spec.s - ego_fstate[FS_SX], spec.t)
            lon_cost = lon_jerk * LON_JERK_COST_WEIGHT

        return lon_cost, lat_cost


class BP_RightLaneMetric:
    @staticmethod
    def calc_cost(time_period: float, lane_idx: int) -> float:
        """
        Calculate non-right lane cost for the given lane
        :param time_period: [s] time period of the action
        :param lane_idx: lane index (0 means the rightest lane)
        :return: non-right lane cost
        """
        return BP_RIGHT_LANE_COST_WEIGHT * lane_idx * time_period


class BP_LaneDeviationMetric:
    @staticmethod
    def calc_cost(lat_dev: float) -> float:
        """
        Calculate lane deviation cost for an action
        :param lat_dev: [m] maximal lateral deviation during an action
        :return: lane deviation cost
        """
        return BP_METRICS_LANE_DEVIATION_COST_WEIGHT * lat_dev * lat_dev
