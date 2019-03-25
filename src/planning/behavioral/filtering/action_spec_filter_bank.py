import numpy as np
from typing import List

from decision_making.src.global_constants import BP_ACTION_T_LIMITS
from decision_making.src.global_constants import EPS, WERLING_TIME_RESOLUTION, VELOCITY_LIMITS, LON_ACC_LIMITS, \
    LAT_ACC_LIMITS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.types import C_A, C_V, C_K, FS_SX, FS_SV
from decision_making.src.planning.types import FS_SA, FS_DX, LIMIT_MIN
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
import rte.python.profiler as prof


class FilterIfNone(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        return [(action_spec and behavioral_state) is not None for action_spec in action_specs]


class FilterForKinematics(ActionSpecFilter):
    @prof.ProfileFunction()
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        relative_lanes = np.array([spec.relative_lane for spec in action_specs])

        initial_fstates = np.array([behavioral_state.projected_ego_fstates[lane] for lane in relative_lanes])
        terminal_fstates = np.array([spec.as_fstate() for spec in action_specs])
        T = np.array([spec.t for spec in action_specs])

        constraints_s = np.concatenate((initial_fstates[:, :(FS_SA+1)], terminal_fstates[:, :(FS_SA+1)]), axis=1)
        constraints_d = np.concatenate((initial_fstates[:, FS_DX:], terminal_fstates[:, FS_DX:]), axis=1)

        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T))
        poly_coefs_s = QuinticPoly1D.zip_solve(A_inv, constraints_s)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv, constraints_d)

        are_valid = []
        for poly_s, poly_d, t, lane, s_T, v_T in zip(poly_coefs_s, poly_coefs_d, T, relative_lanes,
                                                     terminal_fstates[:, FS_SX], terminal_fstates[:, FS_SV]):
            if np.isnan(t):
                are_valid.append(False)
                continue

            time_samples = np.arange(0, t + EPS, WERLING_TIME_RESOLUTION)
            frenet_frame = behavioral_state.extended_lane_frames[lane]
            total_time = max(BP_ACTION_T_LIMITS[LIMIT_MIN], t)

            samplable_trajectory = SamplableWerlingTrajectory(0, t, t, total_time, frenet_frame, poly_s, poly_d)
            samples = samplable_trajectory.sample(time_samples)

            is_valid = KinematicUtils.filter_by_cartesian_limits(samples[np.newaxis,...],
                                                                 VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS)[0]

            are_valid.append(is_valid)

        return are_valid
