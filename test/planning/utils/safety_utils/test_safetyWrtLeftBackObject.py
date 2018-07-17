import pytest
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.test.planning.utils.safety_utils.test_safetyUtils import create_trajectory, EGO_SIZE, OBJ_SIZE, \
    T_S


@pytest.mark.parametrize(
    'index, ego_v0, ego_vT, ego_lane0, ego_laneT, ego_T_d, obj_v0, obj_vT, obj_lane0, obj_laneT, obj_lon0, obj_T_d, expected', [
     # |           EGO          |              OBJECT         |
  # idx| v0  vT lane0 laneT T_d | v0, vT lane0 laneT lon0 T_d |expected
     (0, 30, 30,  0,    1,  T_S,  30, 30,  1,    1, -120, T_S, True),# fast ego is safe wrt fast LB (120 m behind ego)
     (1, 20, 30,  0,    1,  T_S,  30, 30,  1,    1, -120, T_S, False),# accelerating ego is unsafe wrt faster LB (120 m behind ego)
     (2, 10, 20,  0,    1,    6,  30, 30,  1,    1, -190, T_S, False),# ego performs dangerous cut-in of LB: ego enters to corridor while lon_unsafe
     (3, 10, 20,  0,    1,    3,  30, 30,  1,    1, -190, T_S, True),# ego enters to corridor of LB when it safe lon.
     (4, 30, 30,  0,    1,  T_S,  20, 20,  1,    1,  -15, T_S, True),# safe wrt to close LB: becomes safe lon. before it becomes unsafe lat
     (5, 30, 30,  0,    1,  T_S,  20, 20,  1,    1,  -10, T_S, False),# dangerous cut-in of LB: becomes safe lon. after it becomes unsafe laterally
     (6, 10, 30,  0,    1,    6,   9,  9,  1,    1,  -20, T_S, True),# ego is safe wrt close slow LB (9 m/s, 20 m behind ego)
     (7, 10, 30,  0,    1,  T_S,  30, 30,  1,    1,  -20, T_S, True),# LB moves to ego's lane behind ego on unsafe distance, but ego is safe because its rear obj's blame
     (8, 30, 30,  0,    1,  T_S,  30, 30,  1,    0,  -30, T_S, False),# ego & LB move one towards another laterally; unsafe because of ego blame
     (9, 30, 30,  0,    1,  T_S,  30, 30,  1,    0,  -30,   3, False)# ego & LB move one towards another laterally. At blame time ego moves laterally slower than thresh, but its actions is toward LB, then it's blamed
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
    """
    ego_ftraj = create_trajectory(ego_v0, ego_vT, ego_lane0, ego_laneT, T_d=ego_T_d)
    obj_ftraj = create_trajectory(obj_v0, obj_vT, obj_lane0, obj_laneT, lon0=obj_lon0, T_d=obj_T_d)
    actual_safe = SafetyUtils.get_safe_times(ego_ftraj, EGO_SIZE, obj_ftraj, OBJ_SIZE).all(axis=-1)[0][0]
    assert actual_safe == expected
