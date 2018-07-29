from typing import List

import numpy as np
from decision_making.src.planning.behavioral.data_objects import ActionSpec

from decision_making.src.global_constants import BP_RIGHT_LANE_COST_WEIGHT, BP_EFFICIENCY_COST_WEIGHT, \
    LAT_JERK_COST_WEIGHT, LON_JERK_COST_WEIGHT, BP_DEFAULT_DESIRED_SPEED, \
    BP_METRICS_LANE_DEVIATION_COST_WEIGHT, BP_EFFICIENCY_COST_CONVEXITY_RATIO, EPS
from decision_making.src.planning.types import FS_SA, FS_SV, FS_SX, FS_DA, FS_DV, FS_DX
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.state.state import State
from mapping.src.service.map_service import MapService


class BP_CostFunctions:
    @staticmethod
    def calc_efficiency_cost(state: State, specs: List[ActionSpec],
                             cost_weight: float=BP_EFFICIENCY_COST_WEIGHT,
                             des_vel: float=BP_DEFAULT_DESIRED_SPEED) -> np.array:
        """
        Calculate efficiency cost for a planned following car or lane.
        Do it by calculation of average velocity during achieving the followed car and then following it with
        constant velocity. Total considered time is given by time_horizon.
        :param state: current state
        :param specs: action specifications list
        :return: the efficiency costs array
        """
        ego_fstate = state.ego_state.map_state.road_fstate
        TS = np.array([np.array([spec.t, spec.s]) for spec in specs])
        specs_t, specs_s = np.split(TS, 2, axis=1)
        avg_vel = (specs_s - ego_fstate[FS_SX]) / specs_t
        efficiency_cost = BP_CostFunctions._calc_efficiency_cost_for_velocities(avg_vel, des_vel)
        return cost_weight * efficiency_cost * specs_t

    @staticmethod
    def _calc_efficiency_cost_for_velocities(vel: np.array, des_vel: float=BP_DEFAULT_DESIRED_SPEED) -> np.array:
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
        normalized_vel = np.absolute(1 - vel / des_vel)
        costs = normalized_vel * (coef_2 * normalized_vel + coef_1)
        return costs

    @staticmethod
    def calc_comfort_cost(state: State, specs: List[ActionSpec], T_d_max: np.array=None) -> [np.array, np.array]:
        """
        Calculate comfort cost for lateral and longitudinal movement
        :param state: current state
        :param specs: action specifications list
        :param T_d_max: [sec] array: the largest possible lateral time imposed by safety. T_d_max=spec.t if it's not imposed
        :return: comfort cost in units of the general performance metrics cost
        """
        ego_fstate = state.ego_state.map_state.road_fstate
        TSVD = np.array([np.array([spec.t, spec.s, spec.v, spec.d]) for spec in specs])
        specs_t, specs_s, specs_v, specs_d = np.split(TSVD, 4, axis=1)

        T_d = T_s = np.clip(specs_t, EPS, 1./EPS)
        if T_d_max is not None:
            T_d = np.clip(np.minimum(specs_t, T_d_max), EPS, 1./EPS)

        # lateral jerk
        (dx, dv) = (specs_d - ego_fstate[FS_DX], ego_fstate[FS_DV])
        # if ego is very close to the target lane center, T_d_max=0 should not cause infinite lateral jerk
        dx[np.where(np.abs(dx) < 0.1)] = 0
        lat_jerks = QuinticPoly1D.cumulative_jerk_from_constraints(ego_fstate[FS_DA], dv, 0, dx, T_d)
        lat_costs = lat_jerks * LAT_JERK_COST_WEIGHT

        # longitudinal jerk
        lon_jerks = QuinticPoly1D.cumulative_jerk_from_constraints(
            ego_fstate[FS_SA], ego_fstate[FS_SV], specs_v, specs_s - ego_fstate[FS_SX], T_s)
        lon_costs = lon_jerks * LON_JERK_COST_WEIGHT

        return lon_costs, lat_costs

    @staticmethod
    def calc_right_lane_cost(state: State, specs: List[ActionSpec], cost_weight: float=BP_RIGHT_LANE_COST_WEIGHT) -> \
            np.array:
        """
        Calculate non-right lane cost for the given lane
        :param state: current state
        :param specs: action specifications list
        :return: non-right lane costs array
        """
        TD = np.array([np.array([spec.t, spec.d]) for spec in specs])
        specs_t, specs_d = np.split(TD, 2, axis=1)
        lane_width = MapService.get_instance().get_road(state.ego_state.map_state.road_id).lane_width
        lane_idxs = np.floor(specs_d / lane_width)
        return cost_weight * lane_idxs * specs_t

    @staticmethod
    def calc_lane_deviation_cost(state: State, specs: List[ActionSpec],
                                 cost_weight: float=BP_METRICS_LANE_DEVIATION_COST_WEIGHT) -> np.array:
        """
        Calculate lane deviation costs for an actions
        :param state: current state
        :param specs: action specifications list
        :return: lane deviation costs array
        """
        curr_d = state.ego_state.map_state.road_fstate[FS_DX]
        specs_d = np.array([spec.d for spec in specs])
        lane_width = MapService.get_instance().get_road(state.ego_state.map_state.road_id).lane_width
        # for deviation of half lane, normalized_lane_dev = 1
        normalized_lane_dev = np.minimum(1., 2 * np.abs(specs_d - curr_d) / lane_width)
        return cost_weight * normalized_lane_dev * normalized_lane_dev
