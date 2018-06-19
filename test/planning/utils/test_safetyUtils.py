from logging import Logger

import numpy as np

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.types import C_V, C_YAW, C_Y, C_X, C_K, C_A
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.src.state.state import EgoState, ObjectSize, DynamicObject, State
from mapping.src.service.map_service import MapService


def test_calcSafetyForTrajectories_egoAndObjectsOnTwoLanes():
    """
    Test safety of 3 different ego trajectories w.r.t. 5 different objects moving on two lanes with different velocities
    and starting from different longitudes.
    One of ego trajectories changes lane.
    The test checks safety for whole trajectories.
    """
    ego_size = np.array([4, 2])
    obj_sizes = np.array([[4, 2], [6, 2], [6, 2], [4, 2], [4, 2]])
    lane_wid = 3.

    t = 10.
    times_step = 1.
    times_num = int(t/times_step)
    zeros = np.zeros(times_num)
    time_range = np.arange(0, t, times_step)
    ego_sv1 = np.repeat(10., times_num)
    ego_sv2 = np.repeat(20., times_num)
    ego_sv3 = np.repeat(30., times_num)
    ego_sx1 = time_range * ego_sv1[0]
    ego_sx2 = time_range * ego_sv2[0]
    ego_sx3 = time_range * ego_sv3[0]
    ego_dx = np.repeat(2., times_num)
    ego_dv3 = np.repeat(lane_wid/t, times_num)
    ego_dx3 = ego_dx[0] + time_range * ego_dv3[0]

    # ego has 3 trajectories:
    #   all trajectories start from lon=0 and from the rightest lane
    #   first trajectory moves with constant velocity 10 m/s
    #   second trajectory moves with constant velocity 20 m/s
    #   third trajectory moves with constant velocity 30 m/s and moves laterally from the first lane to the second lane.
    ego_ftraj = np.array([np.c_[ego_sx1, ego_sv1, zeros, ego_dx, zeros, zeros],
                          np.c_[ego_sx2, ego_sv2, zeros, ego_dx, zeros, zeros],
                          np.c_[ego_sx3, ego_sv3, zeros, ego_dx3, ego_dv3, zeros]])

    # both objects start from lon=16 m ahead of ego
    #   the first object moves with velocity 10 m/s on the rightest lane
    #   the second object moves with velocity 20 m/s on the second lane
    #   the third object moves with velocity 30 m/s on the right lane
    #   the fourth object moves with velocity 20 m/s on the right lane, and starts from lon=150
    #   the fifth object moves with velocity 20 m/s on the right lane, and starts from lon=165
    obj_ftraj = np.array([np.c_[ego_sx1 + ego_sv1[0] + (ego_size[0] + obj_sizes[0, 0])/2 + 1, ego_sv1, zeros, ego_dx, zeros, zeros],
                          np.c_[ego_sx2, ego_sv2, zeros, ego_dx + lane_wid, zeros, zeros],
                          np.c_[ego_sx3 + ego_sv3[0] + (ego_size[0] + obj_sizes[2, 0])/2 + 1, ego_sv3, zeros, ego_dx, zeros, zeros],
                          np.c_[ego_sx2 + 5 * ego_sv3[0], ego_sv2, zeros, ego_dx, zeros, zeros],
                          np.c_[ego_sx2 + 5.5 * ego_sv3[0], ego_sv2, zeros, ego_dx, zeros, zeros]])

    safe_times = SafetyUtils.calc_safety_for_trajectories(ego_ftraj, ego_size, obj_ftraj, obj_sizes)

    assert safe_times[0][0].all()  # move with the same velocity and start on the safe distance
    assert safe_times[0][1].all()  # object is ahead and faster, therefore safe
    assert not safe_times[1][0].all()  # the object ahead is slower, therefore unsafe
    assert safe_times[1][1].all()  # move on different lanes, then safe

    assert not safe_times[2][0].all()  # ego is faster
    assert not safe_times[2][1].all()  # ego is faster
    assert safe_times[2][2].all()  # move with the same velocity
    assert not safe_times[2][3].all()  # obj becomes unsafe longitudinally at time 6, before it becomes safe laterally
    assert safe_times[2][4].all()  # obj becomes unsafe longitudinally at time 7, exactly when it becomes safe laterally
