from logging import Logger
import numpy as np
import pytest
import copy

from decision_making.src.global_constants import BP_DEFAULT_DESIRED_SPEED
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.evaluators.cost_functions import BP_CostFunctions
from decision_making.src.planning.behavioral.evaluators.velocity_profile import VelocityProfile


def test_calcEfficiencyCost_differentDeviationsFromDesiredVel_costsComplyEfficiencyCostLogic():
    """
    Test profiles with different deviations from desired velocity and different times.
    Verify the resulting costs comply the efficiency_cost logic.
    """
    v_des = BP_DEFAULT_DESIRED_SPEED
    # deceleration (5 sec) and acceleration (5 sec)
    vel_profile1 = VelocityProfile(v_init=v_des, t_first=5, v_mid=v_des-5, t_flat=0, t_last=5, v_tar=v_des)
    eff_cost1 = BP_CostFunctions.calc_efficiency_cost(vel_profile1)

    # vel_profile2 is symmetric to vel_profile1 around v_des, then has the same cost
    vel_profile2 = copy.deepcopy(vel_profile1)
    vel_profile2.v_mid = v_des + 5  # velocities: v_des, v_des+5, v_des
    eff_cost2 = BP_CostFunctions.calc_efficiency_cost(vel_profile2)
    assert eff_cost1 == eff_cost2

    # vel_profile3 has lower deviation from v_des than vel_profile2, then has lower cost
    vel_profile3 = copy.deepcopy(vel_profile2)
    vel_profile3.v_init = v_des - 2
    vel_profile3.v_mid  = v_des + 3  # velocities: v_des-2, v_des+3, v_des
    eff_cost3 = BP_CostFunctions.calc_efficiency_cost(vel_profile3)
    assert eff_cost3 < eff_cost2

    # flat profile with desired velocity should have cost = 0
    vel_profile4 = VelocityProfile(v_init=v_des, t_first=0, v_mid=v_des, t_flat=10, t_last=0, v_tar=v_des)
    eff_cost4 = BP_CostFunctions.calc_efficiency_cost(vel_profile4)
    assert eff_cost4 == 0

    # increase the profile time and verify that the cost increases accordingly
    vel_profile5 = copy.deepcopy(vel_profile3)
    vel_profile5.t_first *= 2
    vel_profile5.t_last *= 2
    eff_cost5 = BP_CostFunctions.calc_efficiency_cost(vel_profile5)
    assert eff_cost5 == 2*eff_cost3


def test_calcComfortCost():
    ego_fstate = np.array([0, 0, 0, 0, 0, 0])
    spec = ActionSpec(t=10, v=20, s=100, d=3.5)
    T_d_max = 3
    lon_cost1, lat_cost1 = BP_CostFunctions.calc_comfort_cost(ego_fstate, spec, T_d_max, T_d_approx=T_d_max)
    lon_cost2, lat_cost2 = BP_CostFunctions.calc_comfort_cost(ego_fstate, spec, T_d_max, T_d_approx=T_d_max+3)
    lon_cost3, lat_cost3 = BP_CostFunctions.calc_comfort_cost(ego_fstate, spec, T_d_max, T_d_approx=T_d_max-1)
    assert lat_cost1 == lat_cost2 == lat_cost3 and lon_cost1 == lon_cost2 == lon_cost3

    # if T_d_max is not imposed, the lateral jerk is lower, although the approximated T_d is the same
    lon_cost4, lat_cost4 = BP_CostFunctions.calc_comfort_cost(ego_fstate, spec, np.inf, T_d_approx=T_d_max)
    assert lat_cost1 > lat_cost4 and lon_cost1 == lon_cost4
