import pytest
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.test.planning.utils.safety_utils.test_safetyUtils import create_trajectory, EGO_SIZE, OBJ_SIZE, \
    T_S


@pytest.mark.parametrize(
    'index, ego_v0, ego_vT, ego_lane0, ego_laneT, ego_T_d, obj_v0, obj_vT, obj_lane0, obj_laneT, obj_lon0, obj_T_d, expected', [
     # |           EGO          |               OBJECT        |
  # idx| v0  vT lane0 laneT T_d | v0, vT lane0 laneT lon0 T_d |expected
     (0, 10, 10,  0,    0,  T_S,  20, 20,  1,    1,    0, T_S, True),# move on different lanes, then safe
     (1, 30, 30,  0,    1,  T_S,  20, 20,  1,    1,    0, T_S, False),# ego changes lane from 0 to 1, unsafe wrt slower L on lane 1, starting from same lon as ego
     (2, 30, 30,  0,    1,  T_S,  10, 10,  1,    1,    0, T_S, True),# ego changes lane from 0 to 1, safe wrt slow L on lane 1, starting from same lon as ego
     (3, 30, 30,  0,    1,  T_S,  30, 30,  2,    1,    0, T_S, False),# obj & ego move laterally one towards the other to lane 1, becomes unsafe laterally
     (4, 30, 30,  0,    1,  T_S,  30, 30,  1,    2,    0, T_S, True),# obj & ego move laterally to the left, keeping lateral distance, and always safe
     (5, 30, 30,  0,    1,    5,  30, 30,  1,    2,    0, T_S, False),# ego moves laterally to the left faster than obj, then unsafe
     (6, 30, 30,  0,    0,  T_S,  30, 30,  1,    0,    0, T_S, False),# obj moves laterally to ego (obj's blame). Ego is unsafe since L is not rear object
     (7, 30, 30,  0,    1,  T_S,  30, 30,  1,    0,    0, T_S, False),# obj & ego move laterally one toward another, then unsafe
     (8, 30, 30,  0,    1,  T_S,  30, 30,  1,    0,    0,   4, False),# at blame time ego moves laterally very slow, but the action is towards the object, then unsafe
     (9, 30, 30,  2,    1,  T_S,  30, 30,  1,    0,    0, T_S, True),# obj & ego move laterally to the right, keeping lateral distance, and always safe
    (10, 10, 20,  0,    1,  T_S,  30, 30,  1,    1,    0, T_S, True),# ego is safe wrt close but much faster L (becomes LF)
    (11, 20, 30,  0,    0,  T_S,  30, 30,  1,    0,    0, T_S, False)# ego moves on its lane, L performs unsafe cut-in; ego is unsafe since L is not rear object
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
    """
    ego_ftraj = create_trajectory(ego_v0, ego_vT, ego_lane0, ego_laneT, T_d=ego_T_d)
    obj_ftraj = create_trajectory(obj_v0, obj_vT, obj_lane0, obj_laneT, lon0=obj_lon0, T_d=obj_T_d)
    actual_safe = SafetyUtils.get_safe_times(ego_ftraj, EGO_SIZE, obj_ftraj, OBJ_SIZE).all(axis=-1)[0][0]
    assert actual_safe == expected
