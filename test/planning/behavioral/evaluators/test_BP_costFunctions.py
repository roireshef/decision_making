from logging import Logger
import numpy as np
import pytest
import copy

from decision_making.src.global_constants import BP_DEFAULT_DESIRED_SPEED
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.evaluators.cost_functions import BPCosts
from decision_making.src.planning.behavioral.evaluators.velocity_profile import VelocityProfile


def test_calcEfficiencyCost_differentDeviationsFromDesiredVel_costsComplyEfficiencyCostLogic():
    """
    Test profiles with different deviations from desired velocity and different times.
    Verify the resulting costs comply the efficiency_cost logic.
    """
    v_des = BP_DEFAULT_DESIRED_SPEED
    # deceleration (5 sec) and acceleration (5 sec)
    vel_profile1 = VelocityProfile(v_init=v_des, t_first=5, v_mid=v_des-5, t_flat=0, t_last=5, v_tar=v_des)
    eff_cost1 = BPCosts.calc_efficiency_cost(vel_profile1)

    # vel_profile2 is symmetric to vel_profile1 around v_des, then has the same cost
    vel_profile2 = copy.deepcopy(vel_profile1)
    vel_profile2.v_mid = v_des + 5  # velocities: v_des, v_des+5, v_des
    eff_cost2 = BPCosts.calc_efficiency_cost(vel_profile2)
    assert eff_cost1 == eff_cost2

    # vel_profile3 has lower deviation from v_des than vel_profile2, then has lower cost
    vel_profile3 = copy.deepcopy(vel_profile2)
    vel_profile3.v_init = v_des - 2
    vel_profile3.v_mid  = v_des + 3  # velocities: v_des-2, v_des+3, v_des
    eff_cost3 = BPCosts.calc_efficiency_cost(vel_profile3)
    assert eff_cost3 < eff_cost2

    # flat profile with desired velocity should have cost = 0
    vel_profile4 = VelocityProfile(v_init=v_des, t_first=0, v_mid=v_des, t_flat=10, t_last=0, v_tar=v_des)
    eff_cost4 = BPCosts.calc_efficiency_cost(vel_profile4)
    assert eff_cost4 == 0

    # increase the profile time and verify that the cost increases accordingly
    vel_profile5 = copy.deepcopy(vel_profile3)
    vel_profile5.t_first *= 2
    vel_profile5.t_last *= 2
    eff_cost5 = BPCosts.calc_efficiency_cost(vel_profile5)
    assert eff_cost5 == 2*eff_cost3


def test_calcComfortCost_differentMaxApproxLatTimeAndLonDist_maxTimeImposesHighJerk():
    """
    Test lateral and longitudinal comfort costs:
        verify that T_d_max imposes the same high jerk for different T_d_approx
        if T_d_max is not imposed, the lateral jerk is lower, although the approximated T_d is the same
        increasing T_d_approx decreases lateral jerk
        affect of spec.s on the longitudinal jerk
    """
    ego_fstate = np.array([0, 0, 0, 0, 0, 0])
    spec = ActionSpec(t=10, v=15, s=100, d=3.5)
    T_d_max = 3
    # verify that T_d_max imposes the same high jerk for different T_d_approx
    lon_cost1, lat_cost1 = BPCosts.calc_comfort_cost(ego_fstate, spec, T_d_max, T_d_approx=T_d_max)
    lon_cost2, lat_cost2 = BPCosts.calc_comfort_cost(ego_fstate, spec, T_d_max, T_d_approx=T_d_max + 3)
    lon_cost3, lat_cost3 = BPCosts.calc_comfort_cost(ego_fstate, spec, T_d_max, T_d_approx=T_d_max - 1)
    assert lat_cost1 == lat_cost2 == lat_cost3 and lon_cost1 == lon_cost2 == lon_cost3

    # if T_d_max is not imposed, the lateral jerk is lower, although the approximated T_d is the same
    _, lat_cost4 = BPCosts.calc_comfort_cost(ego_fstate, spec, T_d_max=10, T_d_approx=T_d_max)
    assert lat_cost1 > lat_cost4

    # increasing T_d_approx decreases lateral jerk
    _, lat_cost5 = BPCosts.calc_comfort_cost(ego_fstate, spec, T_d_max=10, T_d_approx=T_d_max + 1)
    assert lat_cost5 < lat_cost4

    # in our case increase of spec.s increases longitudinal jerk
    spec_longer_s = copy.deepcopy(spec)
    spec_longer_s.s += 10
    lon_cost6, lat_cost6 = BPCosts.calc_comfort_cost(ego_fstate, spec_longer_s, T_d_max, T_d_approx=T_d_max)
    assert lon_cost6 > lon_cost1 and lat_cost6 == lat_cost1
