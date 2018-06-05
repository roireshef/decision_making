from logging import Logger
import numpy as np
import pytest

from decision_making.src.global_constants import LON_ACC_LIMITS
from decision_making.src.planning.behavioral.evaluators.velocity_profile import VelocityProfile


def test_calcProfileDetails_3segmentsProfile_verifyAllDetails():
    """
    Calculate details of a 3 segments profile:
        times per segment,
        cumulative times,
        cumulative distances,
        velocities per segment
        accelerations per segment.
    Verify all details are correct.
    """
    vel_profile = VelocityProfile(v_init=10, t_first=5, v_mid=15, t_flat=5, t_last=5, v_tar=10)
    t, t_cum, s_cum, v, a = vel_profile.calc_profile_details()
    assert np.sum(t) == t_cum[-1]
    assert s_cum[-1] == vel_profile.total_dist()
    assert a[0] * t[0] == v[1] - v[0]
    assert a[2] * t[2] == vel_profile.v_tar - v[2]


def test_sampleAt_sample3segmentsProfile_sampleIn3Points():
    """
    Test sampling of 3 segments profile in 3 different points.
    """
    vel_profile = VelocityProfile(v_init=10, t_first=5, v_mid=15, t_flat=5, t_last=5, v_tar=10)
    t, t_cum, s_cum, v, a = vel_profile.calc_profile_details()
    t1 = 2
    s1, v1 = vel_profile.sample_at(t1)
    assert v1 == vel_profile.v_init + a[0] * t1
    assert s1 == vel_profile.v_init * t1 + 0.5 * a[0] * t1**2
    t2 = t_cum[1] + 2
    s2, v2 = vel_profile.sample_at(t2)
    assert v2 == vel_profile.v_mid
    assert s2 == s_cum[1] + vel_profile.v_mid * (t2 - t_cum[1])
    t3 = t_cum[2] + 2
    s3, v3 = vel_profile.sample_at(t3)
    assert v3 == vel_profile.v_mid + a[2] * (t3 - t_cum[2])
    assert s3 == s_cum[2] + vel_profile.v_mid * (t3 - t_cum[2]) + 0.5 * a[2] * (t3 - t_cum[2])**2


def test_cutByTime_3segmetsProfile_checkCutInTwoPlaces():
    """
    Cut profile in some point and verify its total time and dist are correct.
    """
    vel_profile = VelocityProfile(v_init=10, t_first=5, v_mid=15, t_flat=5, t_last=5, v_tar=10)
    t, t_cum, s_cum, v, a = vel_profile.calc_profile_details()
    t1 = 7
    cut_profile = vel_profile.cut_by_time(t1)
    assert cut_profile.total_time() == t1
    assert cut_profile.total_dist() == 0.5 * (v[0] + v[1]) * t[0] + v[1] * (t1 - t[0])


def test_calProfileGivenAcc_create2SegProfile_checkDistAndAccelerations():
    """
    Calculate vel_profile for given two velocities, distance and acceleration.
    Verify that it's total distance and accelerations are correct.
    """
    dist1 = 50
    v_tar = 10
    acc = 1
    vel_profile = VelocityProfile.calc_profile_given_acc(v_init=5, a=acc, dist=dist1, v_tar=v_tar)
    t, t_cum, s_cum, v, a = vel_profile.calc_profile_details()
    assert abs(vel_profile.total_dist() - (dist1 + v_tar * t_cum[-1])) < 0.01
    assert abs(abs(a[0]) - acc) < 0.01
    assert abs(abs(a[2]) - acc) < 0.01

    dist2 = -10
    vel_profile = VelocityProfile.calc_profile_given_acc(v_init=30, a=acc, dist=dist2, v_tar=v_tar)
    t, t_cum, s_cum, v, a = vel_profile.calc_profile_details()
    assert abs(vel_profile.total_dist() - (dist2 + v_tar * t_cum[-1])) < 0.01
    assert abs(abs(a[0]) - acc) < 0.01
    assert abs(abs(a[2]) - acc) < 0.01


def test_calcProfileGivenT_3DifferentProfiles_timeAndDistAreCorrect():
    """
    Calculate vel_profile for given two velocities, distance and T.
    Verify that it's total distance and accelerations are correct.
    """
    v_init1 = 5
    dist1 = 50
    v_tar = 10
    T1 = 10
    vel_profile1 = VelocityProfile.calc_profile_given_T(v_init=v_init1, T=T1, dist=dist1, v_tar=v_tar)
    assert vel_profile1.v_init == v_init1 and vel_profile1.v_tar == v_tar
    assert vel_profile1.total_time() == T1
    assert abs(vel_profile1.total_dist() - (dist1 + v_tar * T1)) < 0.001

    v_init2 = 5
    dist2 = -10
    T2 = 10
    vel_profile2 = VelocityProfile.calc_profile_given_T(v_init=v_init2, T=T2, dist=dist2, v_tar=v_tar)
    assert vel_profile2.v_init == v_init2 and vel_profile2.v_tar == v_tar
    assert vel_profile2.total_time() == T2
    assert abs(vel_profile2.total_dist() - (dist2 + v_tar * T2)) < 0.001

    v_init3 = 20
    dist3 = 10
    T3 = 10
    vel_profile3 = VelocityProfile.calc_profile_given_T(v_init=v_init3, T=T3, dist=dist3, v_tar=v_tar)
    assert vel_profile3.v_init == v_init3 and vel_profile3.v_tar == v_tar
    assert vel_profile3.total_time() == T3
    assert abs(vel_profile3.total_dist() - (dist3 + v_tar * T3)) < 0.001


def test_calcLastSafeTime_accDecProfile_safeDistAtSafeTimeShouldBeEqualToActualDist():
    """
    Test calculating last safe time of profile.
    Try 3 different distances from front object and 1 distance from back object.
    Verify: safe distance at last safe time t should be equal to the actual distance at t.
    """
    length = 4
    v_init = 5
    dist = 12.5
    v_obj = 10
    T = 15
    td = 2
    init_s_ego = 400

    vel_profile = VelocityProfile.calc_profile_given_T(v_init=v_init, T=T, dist=dist, v_tar=v_obj)

    init_s_obj = init_s_ego + dist + length + v_obj * td
    safe_time = vel_profile.calc_last_safe_time(init_s_ego, length, init_s_obj, v_obj, length, T, td)
    s_ego_td, v_ego_td = vel_profile.sample_at(safe_time + td)
    safety_dist = vel_profile.get_safety_dist(v_obj, v_ego_td,
                                              (init_s_obj + safe_time*v_obj) - (init_s_ego + s_ego_td), 0, length)
    assert abs(safety_dist) < 0.001

    init_s_obj = init_s_ego + dist + length + v_obj * td - 5
    safe_time = vel_profile.calc_last_safe_time(init_s_ego, length, init_s_obj, v_obj, length, T, td)
    s_ego_td, v_ego_td = vel_profile.sample_at(safe_time + td)
    safety_dist = vel_profile.get_safety_dist(v_obj, v_ego_td,
                                              (init_s_obj + safe_time*v_obj) - (init_s_ego + s_ego_td), 0, length)
    assert abs(safety_dist) < 0.001

    init_s_obj = init_s_ego + dist + length + v_obj * td + 5  # safe distance
    safe_time = vel_profile.calc_last_safe_time(init_s_ego, length, init_s_obj, v_obj, length, T, td)
    assert safe_time == np.inf

    # test a back object
    v_tar_back = 20
    init_s_obj = init_s_ego - (100 + length + v_tar_back * td)
    safe_time = vel_profile.calc_last_safe_time(init_s_ego, length, init_s_obj, v_tar_back, length, T, td)
    s_ego, v_ego = vel_profile.sample_at(safe_time)
    safety_dist = vel_profile.get_safety_dist(
        v_ego, v_tar_back, (init_s_ego + s_ego) - (init_s_obj + safe_time*v_tar_back), td, length)
    assert abs(safety_dist) < 0.001


def test_calcLargestSafeTimeForSegment_twoCarsWithConstantAcceleration_safetyDistIsZero():
    """
    Test a solution of quadratic equation that finds the time t, for which back object becomes unsafe w.r.t. front
    object. Verify that at time t the safe distance is equal to the actual distance.
    """
    margin = 4
    s_back = 100
    s_front = s_back + 20
    v_back = 5
    v_front = 10
    a_back = 1
    a_front = 0.5
    t = VelocityProfile._calc_last_safe_time_for_segment(s_front, v_front, a_front, s_back, v_back, a_back, np.inf, margin)
    st_back = s_back + v_back*t + 0.5*a_back*t*t
    st_front = s_front + v_front*t + 0.5*a_front*t*t
    safety_dist = VelocityProfile.get_safety_dist(v_front + a_front*t, v_back + a_back*t, st_front-st_back, 0, margin)
    assert abs(safety_dist) < 0.001
