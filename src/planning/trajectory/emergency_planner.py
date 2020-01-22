from typing import List

import numpy as np
import rte.python.profiler as prof
from decision_making.src.global_constants import WERLING_TIME_RESOLUTION, LON_ACC_LIMITS
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.types import FS_SV, \
    FS_SA, FS_SX, FS_DX, LIMIT_MIN, FrenetState2D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.state.state import State


class EmergencyPlanner:
    def __init__(self, dt=WERLING_TIME_RESOLUTION):
        self._dt = dt

    @property
    def dt(self):
        return self._dt

    @prof.ProfileFunction()
    def plan(self, state: State, reference_route: FrenetSerret2DFrame) -> List[SamplableTrajectory]:
        """ see base class """

        # The reference_route, the goal, ego and the dynamic objects are given in the global coordinate-frame.
        # The vehicle doesn't need to lay parallel to the road.

        ego_frenet_state: FrenetState2D = reference_route.cstate_to_fstate(state.ego_state.cartesian_state)

        a_0 = ego_frenet_state[FS_SA]
        v_0 = ego_frenet_state[FS_SV]
        s_0 = ego_frenet_state[FS_SX]
        MIN_ACC = LON_ACC_LIMITS[LIMIT_MIN]

        J_j = -5

        T_j = (MIN_ACC - a_0) / J_j
        poly_s_j = np.array([J_j/6, a_0/2, v_0, s_0])

        poly_v_j = np.polyval(np.polyder(poly_s_j, m=2), T_j)
        poly_a_j = np.polyder(poly_s_j, m=1)

        # reaching decel limit with constant jerk already reaches v=0
        if poly_v_j < 0:
            # find v=0 time
            roots = Math.find_real_roots_in_limits(poly_v_j, np.array([0, np.inf]))
            T_v_0 = roots[0]

            trajectories = [SamplableWerlingTrajectory(timestamp_in_sec=state.ego_state.timestamp_in_sec,
                                                       T_s=T_v_0,
                                                       T_d=T_v_0,
                                                       T_extended=max(T_v_0, 2),
                                                       frenet_frame=reference_route,
                                                       poly_s_coefs=poly_s_j,
                                                       poly_d_coefs=np.array([ego_frenet_state[FS_DX]]))]

        # need to apply constant accel
        else:
            # constant decel (MIN_ACC) profile
            v_j = np.polyval(poly_v_j, T_j)
            s_j = np.polyval(poly_s_j, T_j)
            poly_s_a = np.array([MIN_ACC/2, v_j, s_j])
            poly_v_a = np.polyder(poly_s_a, m=1)

            roots = Math.find_real_roots_in_limits(poly_v_a, np.array([0, np.inf]))
            T_v_0 = roots[0]

            trajectories = [SamplableWerlingTrajectory(timestamp_in_sec=state.ego_state.timestamp_in_sec,
                                                       T_s=T_j,
                                                       T_d=T_j,
                                                       T_extended=T_j,
                                                       frenet_frame=reference_route,
                                                       poly_s_coefs=poly_s_j,
                                                       poly_d_coefs=np.array([ego_frenet_state[FS_DX]])),
                            SamplableWerlingTrajectory(timestamp_in_sec=state.ego_state.timestamp_in_sec + T_j,
                                                       T_s=T_v_0,
                                                       T_d=T_v_0,
                                                       T_extended=max(T_v_0, 2 - T_j),
                                                       frenet_frame=reference_route,
                                                       poly_s_coefs=poly_s_a,
                                                       poly_d_coefs=np.array([ego_frenet_state[FS_DX]]))
                            ]

        return trajectories