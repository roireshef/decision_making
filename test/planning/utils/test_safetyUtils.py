import numpy as np
import pytest

from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.src.state.state import ObjectSize
from decision_making.test.planning.utils.trajectory_utils import TrajectoryUtils
from decision_making.test.planning.utils.trajectory_utils import T_s


@pytest.fixture(scope='function')
def default_object_size():
    return ObjectSize(4, 2, 0)


@pytest.mark.parametrize(
    'index, ego_v0, ego_vT, ego_lane0, ego_laneT, ego_T_d, obj_v0, obj_vT, obj_lane0, obj_laneT, obj_lon0, obj_T_d, expected', [
     # |           EGO          |              OBJECT         |
  # idx| v0  vT lane0 laneT T_d | v0, vT lane0 laneT lon0 T_d |expected
     (0, 10, 10, 0, 0, T_s(), 10, 10, 0, 0, 15, T_s(), True),# move with the same velocity 10 and start on the safe distance
     (1, 20, 20, 0, 0, T_s(), 10, 10, 0, 0, 15, T_s(), False),# the object ahead is slower, therefore unsafe
     (2, 30, 30, 0, 1, T_s(), 30, 30, 0, 0, 35, T_s(), True),# F moves with the same velocity as ego, then safe
     (3, 30, 30, 0, 1, T_s(), 20, 20, 0, 0, 130, T_s(), False),# becomes unsafe longitudinally, before it becomes safe laterally
     (4, 30, 30, 0, 1, 5, 20, 20, 0, 0, 130, T_s(), True),# becomes unsafe longitudinally, after it becomes safe laterally, since T_d=5
     (5, 30, 30, 0, 1, 3.1, 10, 10, 0, 0, 165, T_s(), False),# ego_overtakes slow obj with T_d not small enough
     (6, 30, 30, 0, 1, 3.0, 10, 10, 0, 0, 165, T_s(), True),# ego_overtakes slow obj with small enough T_d
     (7, 30, 30, 0, 1, 5, 10, 10, 0, 0, 185, T_s(), False),# ego_overtakes slow obj that is not far enough for T_d=5
     (8, 30, 30, 0, 1, 5, 10, 10, 0, 0, 190, T_s(), True)# ego_overtakes slow obj that is far enough for T_d=5
    ]
)
def test_calcSafetyForTrajectories_safetyWrtFrontObject_allCasesShouldComplyRSS(
        index: int,
        ego_v0: float, ego_vT: float, ego_lane0: int, ego_laneT: int, ego_T_d: float,
        obj_v0: float, obj_vT: float, obj_lane0: int, obj_laneT: int, obj_lon0: float, obj_T_d: float,
        expected: bool):
    """
    Test safety of lane change trajectories w.r.t. object F. Try different T_d for slow F.
    The test checks safety for whole trajectories.
    :param index: serial number of iteration created by @pytest.mark.parametrize. Enables to debug a specific scenario
    :param ego_v0: [m/s] ego initial velocity
    :param ego_vT: [m/s] ego final velocity
    :param ego_lane0: [int] initial lane index of ego
    :param ego_laneT: [int] final lane index of ego
    :param ego_T_d: [sec] lateral time for lane change
    :param obj_v0: [m/s] object's initial velocity
    :param obj_vT: [m/s] object's final velocity
    :param obj_lane0: [int] initial lane index of the object
    :param obj_laneT: [int] final lane index of the object
    :param obj_lon0: [m] initial object's longitude
    :param obj_T_d: [m] final object's longitude
    :param expected: (bool ground truth) expected safety result for every scenario in the list @pytest.mark.parametrize
    """
    obj_size = default_object_size()
    ego_ftraj = TrajectoryUtils.create_ftrajectory(ego_v0, ego_vT, ego_lane0, ego_laneT, T_d=ego_T_d)
    obj_ftraj = TrajectoryUtils.create_ftrajectory(obj_v0, obj_vT, obj_lane0, obj_laneT, lon0=obj_lon0, T_d=obj_T_d)
    actual_safe = (SafetyUtils.get_safety_costs(ego_ftraj, obj_size, obj_ftraj, [obj_size]) < 1).all(axis=-1)[0][0]
    assert actual_safe == expected


@pytest.mark.parametrize(
    'index, ego_v0, ego_vT, ego_lane0, ego_laneT, ego_T_d, obj_v0, obj_vT, obj_lane0, obj_laneT, obj_lon0, obj_T_d, expected', [
     # |           EGO          |              OBJECT         |
  # idx| v0  vT lane0 laneT T_d | v0, vT lane0 laneT lon0 T_d |expected
     (0, 30, 30, 0, 1, T_s(), 30, 30, 1, 1, -120, T_s(), True),# fast ego is safe wrt fast LB (120 m behind ego)
     (1, 20, 30, 0, 1, T_s(), 30, 30, 1, 1, -120, T_s(), False),# accelerating ego is unsafe wrt faster LB (120 m behind ego)
     (2, 10, 20, 0, 1, 6, 30, 30, 1, 1, -230, T_s(), False),# ego performs dangerous cut-in of LB: ego enters to corridor while lon_unsafe
     (3, 10, 20, 0, 1, 3, 30, 30, 1, 1, -240, T_s(), True),# ego enters to corridor of LB when it safe lon.
     (4, 30, 30, 0, 1, T_s(), 20, 20, 1, 1, -20, T_s(), True),# safe wrt to close LB: becomes safe lon. before it becomes unsafe lat
     (5, 30, 30, 0, 1, T_s(), 20, 20, 1, 1, -15, T_s(), False),# dangerous cut-in of LB: becomes safe lon. after it becomes unsafe laterally
     (6, 10, 30, 0, 1, 6, 9, 9, 1, 1, -40, T_s(), True),# ego is safe wrt close slow LB (9 m/s, 40 m behind ego)
     (7, 10, 30, 0, 1, T_s(), 30, 30, 1, 1, -20, T_s(), True),# LB moves to ego's lane behind ego on unsafe distance, but ego is safe because its rear obj's blame
     (8, 30, 30, 0, 1, T_s(), 30, 30, 1, 0, -30, T_s(), False),# ego & LB move one towards another laterally; unsafe because of ego blame
     (9, 30, 30, 0, 1, T_s(), 30, 30, 1, 0, -30, 3, False)# ego & LB move one towards another laterally. At blame time ego moves laterally slower than thresh, but its actions is toward LB, then it's blamed
    ]
)
def test_calcSafetyForTrajectories_safetyWrtLeftBackObject_allCasesShouldComplyRSS(
        index: int,
        ego_v0: float, ego_vT: float, ego_lane0: int, ego_laneT: int, ego_T_d: float,
        obj_v0: float, obj_vT: float, obj_lane0: int, obj_laneT: int, obj_lon0: float, obj_T_d: float,
        expected: bool):
    """
    Test safety of lane change trajectories w.r.t. object LB.
    Try different longitudinal accelerations, different T_d and different initial longitudes of LB.
    The test checks safety for whole trajectories.
    :param index: serial number of iteration created by @pytest.mark.parametrize. Enables to debug a specific scenario
    :param ego_v0: [m/s] ego initial velocity
    :param ego_vT: [m/s] ego final velocity
    :param ego_lane0: [int] initial lane index of ego
    :param ego_laneT: [int] final lane index of ego
    :param ego_T_d: [sec] lateral time for lane change
    :param obj_v0: [m/s] object's initial velocity
    :param obj_vT: [m/s] object's final velocity
    :param obj_lane0: [int] initial lane index of the object
    :param obj_laneT: [int] final lane index of the object
    :param obj_lon0: [m] initial object's longitude
    :param obj_T_d: [m] final object's longitude
    :param expected: (bool ground truth) expected safety result for every scenario in the list @pytest.mark.parametrize
    """
    obj_size = default_object_size()
    ego_ftraj = TrajectoryUtils.create_ftrajectory(ego_v0, ego_vT, ego_lane0, ego_laneT, T_d=ego_T_d)
    obj_ftraj = TrajectoryUtils.create_ftrajectory(obj_v0, obj_vT, obj_lane0, obj_laneT, lon0=obj_lon0, T_d=obj_T_d)
    actual_safe = (SafetyUtils.get_safety_costs(ego_ftraj, obj_size, obj_ftraj, [obj_size]) < 1).all(axis=-1)[0][0]
    assert actual_safe == expected


@pytest.mark.parametrize(
    'index, ego_v0, ego_vT, ego_lane0, ego_laneT, ego_T_d, obj_v0, obj_vT, obj_lane0, obj_laneT, obj_lon0, obj_T_d, expected', [
     # |           EGO          |               OBJECT        |
  # idx| v0  vT lane0 laneT T_d | v0, vT lane0 laneT lon0 T_d |expected
     (0, 30, 30, 0, 1, T_s(), 30, 30, 1, 1, 20, T_s(), False),# fast ego unsafe wrt too close LF with same velocity
     (1, 30, 30, 0, 1, T_s(), 30, 30, 1, 1, 40, T_s(), True),# fast ego safe wrt not too close LF with same velocity
     (2, 20, 30, 0, 1, T_s(), 30, 30, 1, 1, 20, T_s(), True),# accelerating ego is safe wrt close faster LF
     (3, 20, 30, 0, 0, T_s(), 30, 30, 1, 0, 10, T_s(), True),# faster LF performs safe cut-in of ego
     (4, 30, 30, 0, 0, T_s(), 30, 30, 1, 0, 10, T_s(), False)# LF performs unsafe cut-in of ego
    ]
)
def test_calcSafetyForTrajectories_safetyWrtLeftFrontObject_allCasesShouldComplyRSS(
        index: int,
        ego_v0: float, ego_vT: float, ego_lane0: int, ego_laneT: int, ego_T_d: float,
        obj_v0: float, obj_vT: float, obj_lane0: int, obj_laneT: int, obj_lon0: float, obj_T_d: float,
        expected: bool):
    """
    Test safety of ego trajectories w.r.t. object LF with different velocities and starting from different longitudes.
    The test checks safety for whole trajectories.
    :param index: serial number of iteration created by @pytest.mark.parametrize. Enables to debug a specific scenario
    :param ego_v0: [m/s] ego initial velocity
    :param ego_vT: [m/s] ego final velocity
    :param ego_lane0: [int] initial lane index of ego
    :param ego_laneT: [int] final lane index of ego
    :param ego_T_d: [sec] lateral time for lane change
    :param obj_v0: [m/s] object's initial velocity
    :param obj_vT: [m/s] object's final velocity
    :param obj_lane0: [int] initial lane index of the object
    :param obj_laneT: [int] final lane index of the object
    :param obj_lon0: [m] initial object's longitude
    :param obj_T_d: [m] final object's longitude
    :param expected: (bool ground truth) expected safety result for every scenario in the list @pytest.mark.parametrize
    """
    obj_size = default_object_size()
    ego_ftraj = TrajectoryUtils.create_ftrajectory(ego_v0, ego_vT, ego_lane0, ego_laneT, T_d=ego_T_d)
    obj_ftraj = TrajectoryUtils.create_ftrajectory(obj_v0, obj_vT, obj_lane0, obj_laneT, lon0=obj_lon0, T_d=obj_T_d)
    actual_safe = (SafetyUtils.get_safety_costs(ego_ftraj, obj_size, obj_ftraj, [obj_size]) < 1).all(axis=-1)[0][0]
    assert actual_safe == expected


@pytest.mark.parametrize(
    'index, ego_v0, ego_vT, ego_lane0, ego_laneT, ego_T_d, obj_v0, obj_vT, obj_lane0, obj_laneT, obj_lon0, obj_T_d, expected', [
     # |           EGO          |               OBJECT        |
  # idx| v0  vT lane0 laneT T_d | v0, vT lane0 laneT lon0 T_d |expected
     (0, 10, 10, 0, 0, T_s(), 20, 20, 1, 1, 0, T_s(), True),# move on different lanes, then safe
     (1, 30, 30, 0, 1, T_s(), 20, 20, 1, 1, 0, T_s(), False),# ego changes lane from 0 to 1, unsafe wrt slower L on lane 1, starting from same lon as ego
     (2, 30, 30, 0, 1, T_s(), 10, 10, 1, 1, 0, T_s(), True),# ego changes lane from 0 to 1, safe wrt slow L on lane 1, starting from same lon as ego
     (3, 30, 30, 0, 1, T_s(), 30, 30, 2, 1, 0, T_s(), False),# obj & ego move laterally one towards the other to lane 1, becomes unsafe laterally
     (4, 30, 30, 0, 1, T_s(), 30, 30, 1, 2, 0, T_s(), True),# obj & ego move laterally to the left, keeping lateral distance, and always safe
     (5, 30, 30, 0, 1, 5, 30, 30, 1, 2, 0, T_s(), False),# ego moves laterally to the left faster than obj, then unsafe
     (6, 30, 30, 0, 0, T_s(), 30, 30, 1, 0, 0, T_s(), False),# obj moves laterally to ego (obj's blame). Ego is unsafe since L is not rear object
     (7, 30, 30, 0, 1, T_s(), 30, 30, 1, 0, 0, T_s(), False),# obj & ego move laterally one toward another, then unsafe
     (8, 30, 30, 0, 1, T_s(), 30, 30, 1, 0, 0, 4, False),# at blame time ego moves laterally very slow, but the action is towards the object, then unsafe
     (9, 30, 30, 2, 1, T_s(), 30, 30, 1, 0, 0, T_s(), True),# obj & ego move laterally to the right, keeping lateral distance, and always safe
    (10, 10, 20, 0, 1, T_s(), 30, 30, 1, 1, 0, T_s(), True),# ego is safe wrt close but much faster L (becomes LF)
    (11, 25, 30, 0, 0, T_s(), 30, 30, 1, 0, 0, T_s(), False)# ego moves on its lane, L performs unsafe cut-in; ego is unsafe since L is not rear object
    ]
)
def test_calcSafetyForTrajectories_safetyWrtLeftObject_allCasesShouldComplyRSS(
        index: int,
        ego_v0: float, ego_vT: float, ego_lane0: int, ego_laneT: int, ego_T_d: float,
        obj_v0: float, obj_vT: float, obj_lane0: int, obj_laneT: int, obj_lon0: float, obj_T_d: float,
        expected: bool):
    """
    Test safety of ego trajectories w.r.t. object L moving on three lanes with different
    velocities. All velocities are constant along trajectories.
    The test checks safety for whole trajectories.
    :param index: serial number of iteration created by @pytest.mark.parametrize. Enables to debug a specific scenario
    :param ego_v0: [m/s] ego initial velocity
    :param ego_vT: [m/s] ego final velocity
    :param ego_lane0: [int] initial lane index of ego
    :param ego_laneT: [int] final lane index of ego
    :param ego_T_d: [sec] lateral time for lane change
    :param obj_v0: [m/s] object's initial velocity
    :param obj_vT: [m/s] object's final velocity
    :param obj_lane0: [int] initial lane index of the object
    :param obj_laneT: [int] final lane index of the object
    :param obj_lon0: [m] initial object's longitude
    :param obj_T_d: [m] final object's longitude
    :param expected: (bool ground truth) expected safety result for every scenario in the list @pytest.mark.parametrize
    """
    obj_size = default_object_size()
    ego_ftraj = TrajectoryUtils.create_ftrajectory(ego_v0, ego_vT, ego_lane0, ego_laneT, T_d=ego_T_d)
    obj_ftraj = TrajectoryUtils.create_ftrajectory(obj_v0, obj_vT, obj_lane0, obj_laneT, lon0=obj_lon0, T_d=obj_T_d)
    actual_safe = (SafetyUtils.get_safety_costs(ego_ftraj, obj_size, obj_ftraj, [obj_size]) < 1).all(axis=-1)[0][0]
    assert actual_safe == expected


@pytest.mark.parametrize(
    'index, ego_v0, ego_vT, ego_lane0, ego_laneT, ego_T_d, obj_v0, obj_vT, obj_lane0, obj_laneT, obj_lon0, obj_T_d, expected', [
     # |           EGO          |               OBJECT        |
  # idx| v0  vT lane0 laneT T_d | v0, vT lane0 laneT lon0 T_d |expected
     (0, 20, 20, 0, 1, 5, 30, 30, 0, 0, -45, T_s(), True),# B is unsafe but ego safe wrt to B, because B is blamed
     (0, 10, 10, 0, 1, 5, 30, 30, 0, 0, -45, T_s(), False)# B is much faster, accident will occur according to the prediction, then unsafe
    ]
)
def test_calcSafetyForTrajectories_safetyWrtRearObject_allCasesShouldComplyRSS(
        index: int,
        ego_v0: float, ego_vT: float, ego_lane0: int, ego_laneT: int, ego_T_d: float,
        obj_v0: float, obj_vT: float, obj_lane0: int, obj_laneT: int, obj_lon0: float, obj_T_d: float,
        expected: bool):
    """
    Test safety of lane change trajectories w.r.t. rear object B.
    Try different longitudinal accelerations, different T_d and different initial longitudes of LB.
    The test checks safety for whole trajectories.
    :param index: serial number of iteration created by @pytest.mark.parametrize. Enables to debug a specific scenario
    :param ego_v0: [m/s] ego initial velocity
    :param ego_vT: [m/s] ego final velocity
    :param ego_lane0: [int] initial lane index of ego
    :param ego_laneT: [int] final lane index of ego
    :param ego_T_d: [sec] lateral time for lane change
    :param obj_v0: [m/s] object's initial velocity
    :param obj_vT: [m/s] object's final velocity
    :param obj_lane0: [int] initial lane index of the object
    :param obj_laneT: [int] final lane index of the object
    :param obj_lon0: [m] initial object's longitude
    :param obj_T_d: [m] final object's longitude
    :param expected: (bool ground truth) expected safety result for every scenario in the list @pytest.mark.parametrize
    """
    obj_size = default_object_size()
    ego_ftraj = TrajectoryUtils.create_ftrajectory(ego_v0, ego_vT, ego_lane0, ego_laneT, T_d=ego_T_d)
    obj_ftraj = TrajectoryUtils.create_ftrajectory(obj_v0, obj_vT, obj_lane0, obj_laneT, lon0=obj_lon0, T_d=obj_T_d)
    actual_safe = (SafetyUtils.get_safety_costs(ego_ftraj, obj_size, obj_ftraj, [obj_size]) < 1).all(axis=-1)[0][0]
    assert actual_safe == expected
