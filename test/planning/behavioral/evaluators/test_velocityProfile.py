from logging import Logger
import numpy as np
import pytest

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
    assert s_cum[-1] + v[2] * t[2] + 0.5 * a[2] * t[2]**2 == vel_profile.total_dist()
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


def test_findVelDuringDeceleration_differentAccDecOrders_checkExactTime():
    """
    Consider two profiles: (1) acceleration, flat, deceleration; (2) deceleration, acceleration
    Verify that a given velocity is correctly found during deceleration in both profiles.
    """
    vel_profile1 = VelocityProfile(v_init=10, t_first=5, v_mid=15, t_flat=5, t_last=5, v_tar=10)
    t, t_cum, s_cum, v, a = vel_profile1.calc_profile_details()
    v1 = vel_profile1.v_tar + 2
    t1 = vel_profile1.find_vel_during_deceleration(v1)
    assert t1 == t_cum[-1] - (vel_profile1.v_tar - v1) / a[2]

    vel_profile2 = VelocityProfile(v_init=15, t_first=5, v_mid=10, t_flat=0, t_last=5, v_tar=15)
    t, t_cum, s_cum, v, a = vel_profile2.calc_profile_details()
    v2 = vel_profile2.v_init - 2
    t2 = vel_profile2.find_vel_during_deceleration(v2)
    assert t2 == (v2 - vel_profile2.v_init) / a[0]


def test_findVelDuringAcceleration_differentAccDecOrders_checkExactTime():
    """
    Consider two profiles: (1) deceleration, acceleration; (2) acceleration, flat, deceleration.
    Verify that a given velocity is correctly found during acceleration in both profiles.
    """
    vel_profile1 = VelocityProfile(v_init=15, t_first=5, v_mid=10, t_flat=0, t_last=5, v_tar=15)
    t, t_cum, s_cum, v, a = vel_profile1.calc_profile_details()
    v1 = vel_profile1.v_tar - 2
    t1 = vel_profile1.find_vel_during_acceleration(v1)
    assert t1 == t_cum[-1] - (vel_profile1.v_tar - v1) / a[2]

    vel_profile2 = VelocityProfile(v_init=10, t_first=5, v_mid=15, t_flat=5, t_last=5, v_tar=10)
    t, t_cum, s_cum, v, a = vel_profile2.calc_profile_details()
    v2 = vel_profile2.v_init + 2
    t2 = vel_profile2.find_vel_during_acceleration(v2)
    assert t2 == (v2 - vel_profile2.v_init) / a[0]


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


def test_calcProfileGivenT_():
    """
    Calculate vel_profile for given two velocities, distance and T.
    Verify that it's total distance and accelerations are correct.
    """
    dist1 = 50
    v_tar = 10
    T1 = 10
    vel_profile = VelocityProfile.calc_profile_given_T(v_init=5, T=T1, dist=dist1, v_tar=v_tar)
    assert vel_profile.total_time() == T1
    assert abs(vel_profile.total_dist() - (dist1 + v_tar * T1)) < 0.01

    dist2 = -10
    v_tar = 10
    T2 = 10
    vel_profile = VelocityProfile.calc_profile_given_T(v_init=5, T=T2, dist=dist2, v_tar=v_tar)
    assert vel_profile.total_time() == T2
    assert abs(vel_profile.total_dist() - (dist2 + v_tar * T2)) < 0.01
