import numpy as np

from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D


def test_isMaintainingDistance_safeSettings_returnsTrue():
    """ see https://www.desmos.com/calculator/t5db9ymokk (D_headwaymargin line) """
    v0 = 10
    vT = 0
    delta_s_init = 35
    headway_specify = 1.5
    headway_safe = 0.7
    safe_margin = 3
    specify_margin = 5

    s_specify = delta_s_init - headway_specify*vT - specify_margin
    constraints_s = np.array([0, v0, 0, s_specify, vT, 0])

    T = 10
    A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_matrix(T))

    poly_host_s = QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
    poly_target_s = np.array([0, 0, 0, 0, vT, delta_s_init])

    assert KinematicUtils.is_maintaining_distance(poly_host_s, poly_target_s, safe_margin, headway_safe, np.array([0, T]))


def test_isMaintainingDistance_unsafeSettings_returnsFalse():
    """ see https://www.desmos.com/calculator/tqyzkbttoq (D_headwaymargin line) """
    v0 = 10
    vT = 0
    delta_s_init = 35
    headway_specify = 1.5
    headway_safe = 0.7
    safe_margin = 3.5
    specify_margin = 5

    s_specify = delta_s_init - headway_specify*vT - specify_margin
    constraints_s = np.array([0, v0, 0, s_specify, vT, 0])

    T = 10
    A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_matrix(T))

    poly_host_s = QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
    poly_target_s = np.array([0, 0, 0, 0, vT, delta_s_init])

    assert not KinematicUtils.is_maintaining_distance(poly_host_s, poly_target_s, safe_margin, headway_safe, np.array([0, T]))
