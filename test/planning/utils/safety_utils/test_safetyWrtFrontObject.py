import pytest
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.test.planning.utils.safety_utils.test_safetyUtils import create_trajectory, OBJ_SIZE, EGO_SIZE, \
    T_S


@pytest.mark.parametrize(
    'index, ego_v0, ego_vT, ego_lane0, ego_laneT, ego_T_d, obj_v0, obj_vT, obj_lane0, obj_laneT, obj_lon0, obj_T_d, expected', [
     # |           EGO          |              OBJECT         |
  # idx| v0  vT lane0 laneT T_d | v0, vT lane0 laneT lon0 T_d |expected
     (0, 10, 10,  0,    0,  T_S,  10, 10,  0,    0,   15, T_S, True),# move with the same velocity 10 and start on the safe distance
     (1, 20, 20,  0,    0,  T_S,  10, 10,  0,    0,   15, T_S, False),# the object ahead is slower, therefore unsafe
     (2, 30, 30,  0,    1,  T_S,  30, 30,  0,    0,   35, T_S, True),# F moves with the same velocity as ego, then safe
     (3, 30, 30,  0,    1,  T_S,  20, 20,  0,    0,  130, T_S, False),# becomes unsafe longitudinally, before it becomes safe laterally
     (4, 30, 30,  0,    1,    5,  20, 20,  0,    0,  130, T_S, True),# becomes unsafe longitudinally, after it becomes safe laterally, since T_d=5
     (5, 30, 30,  0,    1,  2.6,  10, 10,  0,    0,  165, T_S, False),# ego_overtakes slow obj with T_d not small enough
     (6, 30, 30,  0,    1,  2.5,  10, 10,  0,    0,  165, T_S, True),# ego_overtakes slow obj with small enough T_d
     (7, 30, 30,  0,    1,    5,  10, 10,  0,    0,  190, T_S, False),# ego_overtakes slow obj that is not far enough for T_d=5
     (8, 30, 30,  0,    1,    5,  10, 10,  0,    0,  195, T_S, True)# ego_overtakes slow obj that is far enough for T_d=5
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
    """
    ego_ftraj = create_trajectory(ego_v0, ego_vT, ego_lane0, ego_laneT, T_d=ego_T_d)
    obj_ftraj = create_trajectory(obj_v0, obj_vT, obj_lane0, obj_laneT, lon0=obj_lon0, T_d=obj_T_d)
    actual_safe = SafetyUtils.get_safe_times(ego_ftraj, EGO_SIZE, obj_ftraj, OBJ_SIZE).all(axis=-1)[0][0]
    assert actual_safe == expected
