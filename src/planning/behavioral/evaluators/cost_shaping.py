from typing import Tuple

import numpy as np

from decision_making.src.global_constants import SPECIFICATION_MARGIN_TIME_DELAY, SAFETY_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.evaluators.cost_functions import BP_CostFunctions
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D, Poly1D


def cost_shaping():
    efficiency_weight = efficiency_vs_comfort()
    lane_dev_weight, right_lane_weight = efficiency_vs_right_lane(efficiency_weight)
    print('comfort weight=1, efficiency_weight=%.2f, lane_dev_weight=%.2f, right_lane_weight=%.2f' %
          (efficiency_weight, lane_dev_weight, right_lane_weight))


def efficiency_vs_comfort() -> float:
    """
    Let a static action: acceleration from 0 to 100 km/h. T is unknown.
    Let the desired maximal acceleration for the action is LON_ACC_LIMITS[1] (3 m/s^2).
    Choose such cost weights, for which the best total cost is achieved for an action with maximal acceleration 3 m/s^2.
    Implementation:
        Let comfort cost weight = 1.
        Find T for the static action, for which the maximal acceleration is achieved.
        Loop on efficiency weights w:
            Loop on spec.t:
                Calculate efficiency & comfort costs, using current efficiency weight
            Find spec.t giving the best total cost.
        Choose such w, for which spec.t = T.
        Return the efficiency weight w.
    :return: efficiency cost weight
    """
    desired_max_acceleration = LON_ACC_LIMITS[1]
    vT = 27.8  # 100 km/h

    efficiency_weights_list = np.arange(0.4, 6, 0.2)
    efficiency_costs = np.full(30, np.inf)
    comfort_costs = np.full(30, np.inf)
    max_accel = np.full(30, np.inf)

    for T in range(8, 30):
        constraints_s = np.array([0, 0, 0, vT, 0])
        poly_s = WerlingPlanner._solve_1d_poly(constraints_s[np.newaxis], T, QuarticPoly1D)[0]
        fstates = Poly1D.polyval_with_derivatives(poly_s[np.newaxis], np.arange(0, T, 0.1))
        max_accel[T] = np.max(fstates[..., 2])
    T_by_accel = np.argmin(np.abs(max_accel - desired_max_acceleration))

    dist_from_correct_T = np.zeros((len(efficiency_weights_list)))
    for i, w in enumerate(efficiency_weights_list):
        for T in range(8, 30):
            fstate = np.array([0, 0, 0, 0, 0, 0])
            constraints_s = np.array([0, 0, 0, vT, 0])
            poly_s = WerlingPlanner._solve_1d_poly(constraints_s[np.newaxis], T, QuarticPoly1D)[0]
            sT = Math.polyval2d(poly_s[np.newaxis], np.array([float(T)]))
            spec = ActionSpec(t=T, v=vT, s=sT, d=0)
            efficiency_costs[T] = BP_CostFunctions.calc_efficiency_cost(fstate, spec, w, vT)
            comfort_costs[T], _ = BP_CostFunctions.calc_comfort_cost(fstate, spec, 0, 0)
        dist_from_correct_T[i] = abs(np.argmin(efficiency_costs + comfort_costs) - T_by_accel)

    efficiency_weight = efficiency_weights_list[np.argmin(dist_from_correct_T)]
    return efficiency_weight


def efficiency_vs_right_lane(efficiency_weight: float) -> [float, float]:
    """
    1. Let the agent and the goal are on the left lane and time_to_left_goal (e.g. 30 sec).
            lane1_cost = time_to_left_goal * non_right_weight
            lane0_cost = 2 * (comfort + lane_dev_cost) + T_d * non_right_weight.
        If we assume a boundary situation lane0_cost == lane1_cost, then we get an equation:
        non_right_weight * time_to_left_goal = 2 * (lane_dev_cost + comfort) + non_right_weight * T_d
        2 * (lane_dev_cost + comfort) = non_right_weight * (time_to_left_goal - T_d)

    2. Let the goal is on the right lane in distance D ahead,
       Suppose a boundary situation: another object moves with velocity v = desired_vel - dv on the right lane.
       Suppose a cost of full overtake, i.e. two lane changes (to the left and back to the right), is equivalent
       to a difference of dt in arrival time to the goal.
       For example, if desired_vel = 20 m/s, dv = 2 m/s, dt = 5 sec, then:
       D/20 + 5 sec = D/18, then D = 1233 m.

        lane0_cost = efficiency_cost(dist_to_goal/18, 18)
        lane1_cost = 2 * (lane_dev + comfort) + non_right_cost =
                     2 * (lane_dev + comfort) + non_right_weight * overtake_time =
                     non_right_weight * (time_to_left_goal - T_d) + non_right_weight * overtake_time =   # from (1)
                     non_right_weight * (time_to_left_goal - T_d + overtake_time)

        If we assume a boundary situation: lane0_cost == lane1_cost, then we get an equation:
        non_right_weight = lane0_cost / (time_to_left_goal - T_d + overtake_time)
        From (1) we get:
        lane_dev_weight = lane_dev_cost = 0.5 * non_right_weight * (time_to_left_goal - T_d) - comfort
    """
    slow_v = 18
    desired_v = 20.
    time_loss_equiv_to_lane_change = 5

    dist_to_goal = time_loss_equiv_to_lane_change / (1/slow_v - 1/desired_v)
    long_T = dist_to_goal/slow_v
    fstate = np.array([0, slow_v, 0, 0, 0, 0])
    spec = ActionSpec(t=long_T, v=slow_v, s=dist_to_goal, d=0)
    lane0_costs = BP_CostFunctions.calc_efficiency_cost(fstate, spec, efficiency_weight, desired_v)

    car_length = 4
    lat_comfort_cost, T_d = calc_lane_change_comfort_cost()
    # lane_dev_cost = BP_CostFunctions.calc_lane_deviation_cost(1, lane_dev_weight)
    overtake_dist = desired_v * (SPECIFICATION_MARGIN_TIME_DELAY + SAFETY_MARGIN_TIME_DELAY) + car_length
    overtake_time = overtake_dist / (desired_v - slow_v) - T_d

    time_to_left_goal = 30  # time to goal on the left lane
    right_lane_weight = lane0_costs / (time_to_left_goal - T_d + overtake_time)
    lane_dev_weight = 0.5 * (time_to_left_goal - T_d) * right_lane_weight - lat_comfort_cost
    return lane_dev_weight, right_lane_weight


def calc_lane_change_comfort_cost():
    lane_width = 3.6
    T_d = 7
    fstate = np.array([0, 0, 0, 0, 0, 0])
    spec = ActionSpec(0, 0, 0, d=lane_width)
    _, lat_comfort_cost = BP_CostFunctions.calc_comfort_cost(fstate, spec, T_d, T_d)
    return lat_comfort_cost, T_d


cost_shaping()
