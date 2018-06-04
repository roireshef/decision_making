from logging import Logger
import numpy as np
import pytest

from decision_making.src.global_constants import BP_DEFAULT_DESIRED_SPEED
from decision_making.src.planning.behavioral.evaluators.cost_functions import BP_CostFunctions
from decision_making.src.planning.behavioral.evaluators.velocity_profile import VelocityProfile


def test_calcEfficiencyCost():
    v_des = BP_DEFAULT_DESIRED_SPEED
    vel_profile = VelocityProfile(v_init=v_des, t_first=5, v_mid=v_des-5, t_flat=0, t_last=5, v_tar=v_des)
    eff_cost1 = BP_CostFunctions.calc_efficiency_cost(vel_profile)
    vel_profile = VelocityProfile(v_init=v_des, t_first=5, v_mid=v_des+5, t_flat=0, t_last=5, v_tar=v_des)
    eff_cost2 = BP_CostFunctions.calc_efficiency_cost(vel_profile)
    assert eff_cost1 == eff_cost2

    vel_profile = VelocityProfile(v_init=v_des-2, t_first=5, v_mid=v_des+3, t_flat=0, t_last=5, v_tar=v_des)
    eff_cost3 = BP_CostFunctions.calc_efficiency_cost(vel_profile)
    assert eff_cost3 < eff_cost2

    vel_profile = VelocityProfile(v_init=v_des, t_first=0, v_mid=v_des, t_flat=10, t_last=0, v_tar=v_des)
    eff_cost4 = BP_CostFunctions.calc_efficiency_cost(vel_profile)
    assert eff_cost4 == 0

    vel_profile = VelocityProfile(v_init=v_des-2, t_first=10, v_mid=v_des+3, t_flat=0, t_last=10, v_tar=v_des)
    eff_cost5 = BP_CostFunctions.calc_efficiency_cost(vel_profile)
    assert eff_cost5 == 2*eff_cost3


    eff_cost1=eff_cost1