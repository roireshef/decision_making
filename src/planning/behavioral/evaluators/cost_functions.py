import numpy as np
from decision_making.src.planning.behavioral.data_objects import ActionSpec

from decision_making.src.global_constants import BP_RIGHT_LANE_COST_WEIGHT, BP_EFFICIENCY_COST_WEIGHT, \
    LAT_JERK_COST_WEIGHT, LON_JERK_COST_WEIGHT, BP_DEFAULT_DESIRED_SPEED, \
    BP_METRICS_LANE_DEVIATION_COST_WEIGHT, BP_EFFICIENCY_COST_CONVEXITY_RATIO
from decision_making.src.planning.types import FS_SA, FS_SV, FS_SX, FS_DA, FS_DV, FS_DX
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D


class BP_CostFunctions:
    @staticmethod
    def calc_efficiency_cost(ego_fstate: np.array, spec: ActionSpec) -> float:
        """
        Calculate efficiency cost for a planned following car or lane.
        Do it by calculation of average velocity during achieving the followed car and then following it with
        constant velocity. Total considered time is given by time_horizon.
        :param ego_fstate: initial ego Frenet state
        :param spec: action specification
        :return: the efficiency cost
        """
        avg_vel = (spec.s - ego_fstate[FS_SX]) / spec.t
        efficiency_cost = BP_CostFunctions.calc_efficiency_cost_for_velocities(np.array([avg_vel]))[0]
        return BP_EFFICIENCY_COST_WEIGHT * efficiency_cost * spec.t

    @staticmethod
    def calc_efficiency_cost_for_velocities(vel: np.array) -> np.array:
        """
        calculate efficiency (velocity) cost by parabola function
        C(vel) = P(v) = a*v*v + b*v, where v = abs(1 - vel/vel_des), C(vel_des) = 0, C(0) = 1, C'(0)/C'(vel_des) = r
        :param vel: input velocities: either 1D or 2D array
        :return: array of size vel.shape of efficiency costs per point
        """
        r = BP_EFFICIENCY_COST_CONVEXITY_RATIO  # C'(0)/C'(vel_des) = P'(1)/P'(0)
        # the following two lines are the solution of two equations on a and b: P(1) = 1, P'(1)/P'(0) = r
        coef_2 = (r-1)/(r+1)    # coefficient of x^2
        coef_1 = 2/(r+1)        # coefficient of x
        normalized_vel = np.absolute(1 - vel / BP_DEFAULT_DESIRED_SPEED)
        costs = normalized_vel * (coef_2 * normalized_vel + coef_1)
        return costs

    @staticmethod
    def calc_comfort_cost(ego_fstate: np.array, spec: ActionSpec, min_lat_jerk: float, T_d: float) -> [float, float]:
        """
        Calculate comfort cost for lateral and longitudinal movement
        :param ego_fstate: initial ego Frenet state
        :param spec: action spec
        :param min_lat_jerk: lateral jerked imposed by safety
        :param T_d: [sec] heuristic approximation of lateral time, according to the initial and end constraints
        :return: comfort cost in units of the general performance metrics cost
        """
        lat_cost = lon_cost = 0

        # lateral jerk
        (dx, dv) = (spec.d - ego_fstate[FS_DX], ego_fstate[FS_DV])
        if 0. < T_d < np.inf and (abs(dx) > 0.5 or dx * dv < 0):  # TODO: find another way to prevent singularity for small t
            lat_jerk = QuinticPoly1D.cumulative_jerk_from_constraints(ego_fstate[FS_DA], dv, 0, dx, T_d)
            lat_jerk = max(lat_jerk, min_lat_jerk)
            lat_cost = lat_jerk * LAT_JERK_COST_WEIGHT

        # longitudinal jerk
        if spec.t > 0:
            lon_jerk = QuinticPoly1D.cumulative_jerk_from_constraints(
                ego_fstate[FS_SA], ego_fstate[FS_SV], spec.v, spec.s - ego_fstate[FS_SX], spec.t)
            lon_cost = lon_jerk * LON_JERK_COST_WEIGHT

        return lon_cost, lat_cost

    @staticmethod
    def calc_right_lane_cost(time_period: float, lane_idx: int) -> float:
        """
        Calculate non-right lane cost for the given lane
        :param time_period: [s] time period of the action
        :param lane_idx: lane index (0 means the rightest lane)
        :return: non-right lane cost
        """
        return BP_RIGHT_LANE_COST_WEIGHT * lane_idx * time_period

    @staticmethod
    def calc_lane_deviation_cost(relative_lat_dev: float) -> float:
        """
        Calculate lane deviation cost for an action
        :param relative_lat_dev: maximal relative lateral deviation during an action. The range: [0, 1].
        :return: lane deviation cost
        """
        return BP_METRICS_LANE_DEVIATION_COST_WEIGHT * relative_lat_dev * relative_lat_dev
