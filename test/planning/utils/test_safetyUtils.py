from logging import Logger

import numpy as np
import time

from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.src.state.state import ObjectSize

obj_size = [ObjectSize(4, 2, 0)]
ego_size = ObjectSize(4, 2, 0)
lane_wid = 3.5
T = 10.
times_step = 0.1
times_num = int(T / times_step) + 1
time_range = np.arange(0, T + 0.001, times_step)
ego_lon = 200
center_lane_lats = lane_wid / 2 + np.arange(0, 2 * lane_wid + 0.001, lane_wid)  # 3 lanes


def create_trajectory(v0: float, vT: float, lane0: int, laneT: int, lon0: float=0, T_d: float=T):
    """
    Create trajectory (for ego or object) for given start/end velocities, start/end lane centers, relative longitude, T_d
    :param v0: initial velocity
    :param vT: end velocity
    :param lane0: initial lane
    :param laneT: end lane
    :param lon0: initial relative longitude (wrt initial ego longitude ego_lon)
    :param T_d:
    :return:
    """
    constraints_s = np.array([[ego_lon + lon0, v0, 0, vT, 0]])
    constraints_d = np.array([[center_lane_lats[lane0], 0, 0, center_lane_lats[laneT], 0, 0]])
    poly_s = WerlingPlanner._solve_1d_poly(constraints_s, T, QuarticPoly1D)
    fstates_s = QuarticPoly1D.polyval_with_derivatives(poly_s, time_range)
    poly_d = WerlingPlanner._solve_1d_poly(constraints_d, T_d, QuinticPoly1D)
    fstates_d = QuinticPoly1D.polyval_with_derivatives(poly_d, time_range)
    if T_d < T:  # fill all samples beyond T_d by the sample at T_d
        T_d_sample = int(T_d / times_step)
        fstates_d[:, T_d_sample+1:, :] = fstates_d[:, T_d_sample:T_d_sample+1, :]
    ftraj = np.dstack((fstates_s, fstates_d))
    return ftraj


def test_calcSafetyForTrajectories_safetyWrtL_allCasesShouldComplyRSS():
    """
    Test safety of ego trajectories w.r.t. object L moving on three lanes with different
    velocities. All velocities are constant along trajectories.
    The test checks safety for whole trajectories.
    """
    ego_ftraj = create_trajectory(v0=10, vT=10, lane0=0, laneT=0)
    obj_ftraj = create_trajectory(v0=20, vT=20, lane0=1, laneT=1)  # faster L
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # move on different lanes, then safe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=20, vT=20, lane0=1, laneT=1)  # slower L
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # ego 30 m/s changes lane from 0 to 1, unsafe wrt L 20 m/s on lane 1 from same lon as ego

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=10, vT=10, lane0=1, laneT=1)  # much slower L
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # ego 30 m/s changes lane from 0 to 1, safe wrt L 10 m/s on lane 1 from same lon as ego

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=2, laneT=1)  # L
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # obj & ego move laterally one towards the other to lane 1, becomes unsafe laterally

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=2)
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # obj & ego move laterally to the left, keeping lateral distance, and always safe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1, T_d=5)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=2)  # L
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # ego moves laterally to the left faster than obj, then unsafe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=0)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=0)
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # obj moves laterally to ego (obj's blame)

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=0)
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # obj & ego move laterally one toward another, then unsafe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=0, T_d=4)
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    # at blame time ego moves laterally very slow and the object moves much faster, but the action is
    # towards the object
    assert not safe_times  # obj & ego move laterally one toward another, then unsafe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=2, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=0)
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # obj & ego move laterally to the right, keeping lateral distance, and always safe

    ego_ftraj = create_trajectory(v0=10, vT=20, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=1)  # fast L becomes LF
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # mid-vel ego (10-20 m/s) is safe w.r.t. close and fast LF (30 m/s)

    # ego moves on its lane, L performs unsafe cut-in
    ego_ftraj = create_trajectory(v0=20, vT=30, lane0=0, laneT=0)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=0)  # close fast L
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # accelerating ego (20->30 m/s) is safe w.r.t. close LF (30 m/s)


def test_calcSafetyForTrajectories_safetyWrtLF_allCasesShouldComplyRSS():
    """
    Test safety of ego trajectories w.r.t. object LF with different velocities and starting from different longitudes.
    The test checks safety for whole trajectories.
    """
    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=1, lon0=20)  # close and fast LF
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # fast ego (30 m/s) is unsafe w.r.t. close LF with same velocity

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=1, lon0=40)  # not close fast LF
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # fast ego (30 m/s) is safe w.r.t. not close LF with same velocity

    ego_ftraj = create_trajectory(v0=20, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=1, lon0=20)  # close fast LF
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # accelerating ego (20->30 m/s) is safe w.r.t. close LF (30 m/s)

    # ego moves on its lane, LF performs safe cut-in
    ego_ftraj = create_trajectory(v0=20, vT=30, lane0=0, laneT=0)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=0, lon0=10)  # close fast LF
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # accelerating ego (20->30 m/s) is safe w.r.t. close LF (30 m/s)


def test_calcSafetyForTrajectories_safetyWrtF_allCasesShouldComplyRSS():
    """
    Test safety of lane change trajectories w.r.t. object F. Try different T_d for slow F.
    The test checks safety for whole trajectories.
    """
    ego_ftraj = create_trajectory(v0=10, vT=10, lane0=0, laneT=0)
    obj_ftraj = create_trajectory(v0=10, vT=10, lane0=0, laneT=0, lon0=15)  # F
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # move with the same velocity 10 and start on the safe distance

    ego_ftraj = create_trajectory(v0=20, vT=20, lane0=0, laneT=0)
    obj_ftraj = create_trajectory(v0=10, vT=10, lane0=0, laneT=0, lon0=15)  # slow F
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # the object ahead is slower, therefore unsafe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=0, lon0=35)
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # F moves with the same velocity as ego, then safe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=20, vT=20, lane0=0, laneT=0, lon0=130)
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    # becomes unsafe longitudinally, before it becomes safe laterally
    assert not safe_times  # ego_v=30 overtake obj_v=20, lon=135 m, T_d = 10     unsafe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1, T_d=5)
    obj_ftraj = create_trajectory(v0=20, vT=20, lane0=0, laneT=0, lon0=130)
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    # becomes unsafe longitudinally, after it becomes safe laterally
    assert safe_times  # ego_v=30 overtake obj_v=20, lon=135 m, T_d = 5     safe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1, T_d=5)
    obj_ftraj = create_trajectory(v0=20, vT=20, lane0=0, laneT=0, lon0=125)
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    # becomes unsafe longitudinally, before it becomes safe laterally
    assert not safe_times  # ego_v=30 overtake obj_v=20, lon=130 m, T_d = 5   unsafe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1, T_d=2.6)
    obj_ftraj = create_trajectory(v0=10, vT=10, lane0=0, laneT=0, lon0=165)
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # ego_v=30 overtake obj_v=10, 175 m, T_d = 2.6   unsafe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1, T_d=2.5)
    obj_ftraj = create_trajectory(v0=10, vT=10, lane0=0, laneT=0, lon0=165)
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # ego_v=30 overtake obj_v=10, 175 m, T_d = 2.5   safe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1, T_d=5)
    obj_ftraj = create_trajectory(v0=10, vT=10, lane0=0, laneT=0, lon0=190)
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # ego_v=30 overtake obj_v=10, 210 m, T_d = 5     unsafe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1, T_d=5)
    obj_ftraj = create_trajectory(v0=10, vT=10, lane0=0, laneT=0, lon0=195)
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # ego_v=30 overtake obj_v=10, 215 m, T_d = 5     safe


def test_calcSafetyForTrajectories_safetyWrtLB_allCasesShouldComplyRSS():
    """
    Test safety of lane change trajectories w.r.t. object LB.
    Try different longitudinal accelerations, different T_d and different initial longitudes of LB.
    The test checks safety for whole trajectories.
    """
    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=1, lon0=-120)  # fast LB
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # fast ego (30 m/s) is safe w.r.t. fast LB (30 m/s, 120 m behind ego)

    ego_ftraj = create_trajectory(v0=20, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=1, lon0=-120)  # fast LB
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # accelerating ego (20->30 m/s) is unsafe w.r.t. LB (30 m/s, 120 m behind ego)

    # ego performs dangerous cut-in of LB
    ego_ftraj = create_trajectory(v0=10, vT=20, lane0=0, laneT=1, T_d=6)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=1, lon0=-190)  # far LB
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # slow ego (10->20 m/s, T_d = 6) is unsafe w.r.t. far LB, since ego enters to corridor while lon_unsafe

    ego_ftraj = create_trajectory(v0=10, vT=20, lane0=0, laneT=1, T_d=3)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=1, lon0=-190)  # far LB
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # slow ego (10->20 m/s, T_d = 3) is safe w.r.t. far LB, since ego enters to corridor while lon_safe

    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=20, vT=20, lane0=1, laneT=1, lon0=-15)  # slower close LB
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # safe wrt to close LB, because it becomes safe longitudinally before it becomes unsafe laterally

    # ego performs dangerous cut-in of LB
    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=20, vT=20, lane0=1, laneT=1, lon0=-10)  # slower close LB
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # safe wrt to close LB, because it becomes safe longitudinally after it becomes unsafe laterally

    ego_ftraj = create_trajectory(v0=10, vT=30, lane0=0, laneT=1, T_d=6)
    obj_ftraj = create_trajectory(v0=9, vT=9, lane0=1, laneT=1, lon0=-20)  # slow close LB
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # ego (10->30 m/s, T_d = 6) is safe wrt close slow LB (9 m/s, 20 m behind ego)

    # unsafe LB, but ego is safe
    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=0)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=0, lon0=-20)  # unsafe LB
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # LB moves to ego's lane, behind ego on unsafe distance, but ego is safe because obj is behind

    # ego & LB move one towards another laterally; unsafe because of blame
    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=0, lon0=-30)  # LB becomes RB
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # ego & LB move one towards another laterally: unsafe since ego has a blame

    # ego & LB move one towards another laterally, but LB changes lane very quickly;
    # At blame time ego moves laterally slower than thresh, but its actions is toward LB, then it's blamed
    ego_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=1)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=1, laneT=0, lon0=-30, T_d=3)  # LB becomes RB
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # ego & LB move one towards another laterally: unsafe since ego has a blame


def test_calcSafetyForTrajectories_safetyWrtB_allCasesShouldComplyRSS():
    """
    Test safety of lane change trajectories w.r.t. rear object B.
    Try different longitudinal accelerations, different T_d and different initial longitudes of LB.
    The test checks safety for whole trajectories.
    """
    # B is unsafe wrt ego, but accident may occur only in the worst case. Then ego is considered safe
    ego_ftraj = create_trajectory(v0=20, vT=20, lane0=0, laneT=1, T_d=5)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=0, lon0=-45)  # B
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert safe_times  # unsafe wrt to rear object B, because B is blamed for danger

    # accident will occur according to the trajectories, then unsafe
    ego_ftraj = create_trajectory(v0=10, vT=10, lane0=0, laneT=1, T_d=5)
    obj_ftraj = create_trajectory(v0=30, vT=30, lane0=0, laneT=0, lon0=-45)  # B
    safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, obj_ftraj, obj_size).all(axis=-1)
    assert not safe_times  # unsafe wrt to rear object B, because B becomes in front of ego (accident)
