import pytest
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.test.planning.utils.safety_utils.test_safetyUtils import create_trajectory, EGO_SIZE, OBJ_SIZE, \
    T_S


@pytest.mark.parametrize(
    'index, ego_v0, ego_vT, ego_lane0, ego_laneT, ego_T_d, obj_v0, obj_vT, obj_lane0, obj_laneT, obj_lon0, obj_T_d, expected', [
     # |           EGO          |               OBJECT        |
  # idx| v0  vT lane0 laneT T_d | v0, vT lane0 laneT lon0 T_d |expected
     (0, 20, 20,  0,    1,    5,  30, 30,  0,    0,  -45, T_S, True),# B is unsafe but ego safe wrt to B, because B is blamed
     (0, 10, 10,  0,    1,    5,  30, 30,  0,    0,  -45, T_S, False)# B is much faster, accident will occur according to the prediction, then unsafe
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
    """
    ego_ftraj = create_trajectory(ego_v0, ego_vT, ego_lane0, ego_laneT, T_d=ego_T_d)
    obj_ftraj = create_trajectory(obj_v0, obj_vT, obj_lane0, obj_laneT, lon0=obj_lon0, T_d=obj_T_d)
    actual_safe = SafetyUtils.get_safe_times(ego_ftraj, EGO_SIZE, obj_ftraj, OBJ_SIZE).all(axis=-1)[0][0]
    assert actual_safe == expected
