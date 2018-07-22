import pytest
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.test.planning.utils.safety_utils.test_safetyUtils import create_trajectory, EGO_SIZE, OBJ_SIZE, \
    T_S


@pytest.mark.parametrize(
    'index, ego_v0, ego_vT, ego_lane0, ego_laneT, ego_T_d, obj_v0, obj_vT, obj_lane0, obj_laneT, obj_lon0, obj_T_d, expected', [
     # |           EGO          |               OBJECT        |
  # idx| v0  vT lane0 laneT T_d | v0, vT lane0 laneT lon0 T_d |expected
     (0, 30, 30,  0,    1,  T_S,  30, 30,  1,    1,   20, T_S, False),# fast ego unsafe wrt too close LF with same velocity
     (1, 30, 30,  0,    1,  T_S,  30, 30,  1,    1,   40, T_S, True),# fast ego safe wrt not too close LF with same velocity
     (2, 20, 30,  0,    1,  T_S,  30, 30,  1,    1,   20, T_S, True),# accelerating ego is safe wrt close faster LF
     (3, 20, 30,  0,    0,  T_S,  30, 30,  1,    0,   10, T_S, True),# faster LF performs safe cut-in of ego
     (4, 30, 30,  0,    0,  T_S,  30, 30,  1,    0,   10, T_S, False)# LF performs unsafe cut-in of ego
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
    """
    ego_ftraj = create_trajectory(ego_v0, ego_vT, ego_lane0, ego_laneT, T_d=ego_T_d)
    obj_ftraj = create_trajectory(obj_v0, obj_vT, obj_lane0, obj_laneT, lon0=obj_lon0, T_d=obj_T_d)
    actual_safe = SafetyUtils.get_safe_times(ego_ftraj, EGO_SIZE, obj_ftraj, OBJ_SIZE).all(axis=-1)[0][0]
    assert actual_safe == expected
