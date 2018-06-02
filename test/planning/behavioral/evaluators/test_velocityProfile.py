from logging import Logger
import numpy as np
import pytest

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, LON_ACC_LIMITS
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
    v_tar = 10
    T = 15
    td = 2
    init_s_ego = 400

    vel_profile = VelocityProfile.calc_profile_given_T(v_init=v_init, T=T, dist=dist, v_tar=v_tar)

    init_s_obj = init_s_ego + dist + length + v_tar * td
    safe_time = vel_profile.calc_last_safe_time(init_s_ego, length, init_s_obj, v_tar, length, T, td)
    s_ego_td, v_ego_td = vel_profile.sample_at(safe_time + td)
    safety_dist = vel_profile.get_safety_dist(v_tar, v_ego_td,
                                               (init_s_obj + safe_time*v_tar) - (init_s_ego + s_ego_td), 0, length)
    assert abs(safety_dist) < 0.001

    init_s_obj = init_s_ego + dist + length + v_tar * td - 5
    safe_time = vel_profile.calc_last_safe_time(init_s_ego, length, init_s_obj, v_tar, length, T, td)
    s_ego_td, v_ego_td = vel_profile.sample_at(safe_time + td)
    safety_dist = vel_profile.get_safety_dist(v_tar, v_ego_td,
                                               (init_s_obj + safe_time*v_tar) - (init_s_ego + s_ego_td), 0, length)
    assert abs(safety_dist) < 0.001

    init_s_obj = init_s_ego + dist + length + v_tar * td + 1  # safe distance
    safe_time = vel_profile.calc_last_safe_time(init_s_ego, length, init_s_obj, v_tar, length, T, td)
    assert safe_time == np.inf

    # test a back object
    v_tar_back = 20
    init_s_obj = init_s_ego - (100 + length + v_tar_back * td)
    safe_time = vel_profile.calc_last_safe_time(init_s_ego, length, init_s_obj, v_tar_back, length, T, td)
    s_ego, v_ego = vel_profile.sample_at(safe_time)
    safety_dist = vel_profile.get_safety_dist(
        v_ego, v_tar_back, (init_s_ego + s_ego) - (init_s_obj + safe_time*v_tar_back), td, length)
    assert abs(safety_dist) < 0.001


def test_isSafeProfile_twoProfiles_checkSafetyForBoundaryCases():
    """
    Check if the profile is fully safe by testing it in boundary cases near the minimal safe distance.
    """
    length = 4
    v_init = 5
    v_tar = 10
    dist = 12.5
    T = 15
    td = 2
    init_s_ego = 100
    vel_profile = VelocityProfile.calc_profile_given_T(v_init=v_init, T=T, dist=dist, v_tar=v_tar)

    # a bit larger than the minimal safe distance
    init_s_obj = init_s_ego + dist + length + v_tar * td + 0.1
    assert vel_profile.is_safe_profile(init_s_ego, init_s_obj, v_tar, length, np.inf, td)

    # a bit smaller than the minimal safe distance
    init_s_obj -= 0.2
    assert not vel_profile.is_safe_profile(init_s_ego, init_s_obj, v_tar, length, np.inf, td)

    # check profile that first decelerates
    vel_profile = VelocityProfile(v_init=20, v_mid=10, v_tar=20, t_first=10, t_flat=0, t_last=10)

    max_brake = -LON_ACC_LIMITS[0]
    init_s_obj = init_s_ego + vel_profile.total_dist() - v_tar * vel_profile.total_time() + \
                 (vel_profile.v_tar**2 - v_tar**2)/(2*max_brake) + vel_profile.v_tar*td + length + 0.1
    assert vel_profile.is_safe_profile(init_s_ego, init_s_obj, v_tar, length, np.inf, td)
    init_s_obj -= 0.2
    assert not vel_profile.is_safe_profile(init_s_ego, init_s_obj, v_tar, length, np.inf, td)


def test_calcLargestTimeForSegment_twoCarsWithConstantAcceleration_safetyDistIsZero():
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
    t = VelocityProfile._calc_largest_time_for_segment(s_front, v_front, a_front, s_back, v_back, a_back, np.inf, margin)
    st_back = s_back + v_back*t + 0.5*a_back*t*t
    st_front = s_front + v_front*t + 0.5*a_front*t*t
    safety_dist = VelocityProfile.get_safety_dist(v_front + a_front*t, v_back + a_back*t, st_front-st_back, 0, margin)
    assert abs(safety_dist) < 0.001
