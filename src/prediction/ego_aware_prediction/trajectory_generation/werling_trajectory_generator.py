import numpy as np
from decision_making.src.planning.trajectory.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner, \
    SamplableWerlingTrajectory
from decision_making.src.planning.types import FS_SX, FS_SV, FS_SA, FS_DX, FS_DV, FS_DA, S5, D5
from decision_making.src.prediction.ego_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.prediction.ego_aware_prediction.trajectory_generation.trajectory_generator import \
    TrajectoryGenerator
from rte.ctm.pythonwrappers.src.FrenetSerret2DFrame import FrenetSerret2DFrame


class WerlingTrajectoryGenerator(TrajectoryGenerator):

    def generate_trajectory(self, timestamp_in_sec: float, frenet_frame: FrenetSerret2DFrame,
                            predicted_maneuver_spec: ManeuverSpec) -> SamplableWerlingTrajectory:
        """
        Generate a trajectory in Frenet coordiantes, according to object's Frenet frame
        :param timestamp_in_sec: [sec] global timestamp *in seconds* to use as a reference
                (other timestamps will be given relative to it)
        :param frenet_frame: Frenet reference frame of object
        :param predicted_maneuver_spec: specification of the trajectory in Frenet frame.

        :return: Trajectory in Frenet frame.
        """

        fconstraints_init = FrenetConstraints(sx=predicted_maneuver_spec.init_state[FS_SX],
                                              sv=predicted_maneuver_spec.init_state[FS_SV],
                                              sa=predicted_maneuver_spec.init_state[FS_SA],
                                              dx=predicted_maneuver_spec.init_state[FS_DX],
                                              dv=predicted_maneuver_spec.init_state[FS_DV],
                                              da=predicted_maneuver_spec.init_state[FS_DA])

        fconstraints_final = FrenetConstraints(sx=predicted_maneuver_spec.final_state[FS_SX],
                                               sv=predicted_maneuver_spec.final_state[FS_SV],
                                               sa=predicted_maneuver_spec.final_state[FS_SA],
                                               dx=predicted_maneuver_spec.final_state[FS_DX],
                                               dv=predicted_maneuver_spec.final_state[FS_DV],
                                               da=predicted_maneuver_spec.final_state[FS_DA])

        # solve problem in Frenet-frame
        ftrajectories, poly_coefs, T_d_vals = WerlingPlanner._solve_optimization(fconst_0=fconstraints_init,
                                                                                 fconst_t=fconstraints_final,
                                                                                 T_s=predicted_maneuver_spec.T_s,
                                                                                 T_d_vals=np.array(
                                                                                     [predicted_maneuver_spec.T_d]),
                                                                                 dt=np.minimum(
                                                                                     predicted_maneuver_spec.T_d,
                                                                                     predicted_maneuver_spec.T_s))

        poly_s = poly_coefs[0, S5:D5]
        poly_d = poly_coefs[0, D5:]

        return SamplableWerlingTrajectory(timestamp_in_sec=timestamp_in_sec,
                                          T_s=predicted_maneuver_spec.T_s,
                                          T_d=predicted_maneuver_spec.T_d,
                                          frenet_frame=frenet_frame,
                                          poly_s_coefs=poly_s,
                                          poly_d_coefs=poly_d)
