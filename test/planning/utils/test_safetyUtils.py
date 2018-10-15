import numpy as np
import pytest

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY
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
     (1, 10, 20, 0, 0, T_s(), 10, 10, 0, 0, 110, T_s(), True),# the object ahead is slower but far enough
     (2, 10, 20, 0, 0, T_s(), 10, 10, 0, 0, 100, T_s(), False) # the object ahead is slower and not far enough
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
    actual_safe = SafetyUtils._get_lon_safety(ego_ftraj, SAFETY_MARGIN_TIME_DELAY, obj_ftraj,
                                              SPECIFICATION_MARGIN_TIME_DELAY, obj_size.length).all(axis=-1)[0]
    assert actual_safe == expected
