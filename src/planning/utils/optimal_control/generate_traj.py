import numpy as np
from decision_making.src.planning.types import FrenetState2D, FS_SX, FS_SV, FS_SA, FS_DX, FS_DV, FS_DA
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D


class TrajectoriesGenerator:

    @staticmethod
    def calc_longitudinal_trajectories(ego_init_fstate: FrenetState2D,
                                       target_t: np.array, target_v: np.array, target_s: np.array,
                                       time_samples: np.array) -> [np.array, np.array]:
        """
        Calculate longitudinal ego trajectories for the given time samples.
        :param ego_init_fstate: initial frenet state of ego
        :param target_t: 1D array of time periods for getting to the target longitude and velocity
        :param target_v: array of target velocities (shape = target_t.shape)
        :param target_s: array of target longitudes (shape = target_t.shape)
        :param time_samples: 1D array of time samples for trajectories generation
        :return: 2D array of trajectories' longitudes, 2D array of velocities.
        Each array of size len(target_t) x len(time_samples)
        """
        # TODO: Acceleration is not calculated.

        # duplicate time_samples array actions_num times
        actions_num = target_t.shape[0]
        dup_time_samples = np.repeat(time_samples, actions_num).reshape(len(time_samples), actions_num)

        ds = target_s - ego_init_fstate[FS_SX]
        # profiles for the cases, when dynamic object is in front of ego
        sx = QuinticPoly1D.distance_by_constraints(a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV],
                                                   v_T=target_v, ds=ds, T=target_t, t=time_samples)
        # set inf to samples outside specs_t
        outside_samples = np.where(dup_time_samples > target_t)
        sx[outside_samples[1], outside_samples[0]] = np.inf

        sv = QuinticPoly1D.velocity_by_constraints(a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV],
                                                   v_T=target_v, ds=ds, T=target_t, t=time_samples)
        return ego_init_fstate[FS_SX] + sx, sv


    @staticmethod
    def calc_lateral_trajectories(ego_init_fstate: FrenetState2D, target_d: np.array,
                                  T_d: np.array, time_samples: np.array) -> [np.array, np.array]:
        """
        Calculate all lateral ego trajectories for the given time samples, for all actions and all T_d periods.
        :param ego_init_fstate: initial frenet state of ego
        :param target_d: array of target latitudes
        :param T_d: array of possible lateral movement times
        :param time_samples: 1D array of time samples for trajectories generation
        :return: 2D array of trajectories' latitudes, 2D array of velocities.
        Two 2D arrays of size len(target_t)*len(T_d) x len(time_samples)
        """
        # TODO: Acceleration is not calculated.

        actions_num = target_d.shape[0]
        T_d_num = T_d.shape[0]
        A_inv = QuinticPoly1D.inv_time_constraints_tensor(T_d)
        (zeros, ones) = (np.zeros(len(target_d)), np.ones(len(target_d)))
        constraints_d = np.array([ego_init_fstate[FS_DX] * ones, ego_init_fstate[FS_DV] * ones,
                                  ego_init_fstate[FS_DA] * ones, target_d, zeros, zeros])  # 6 x len(specs_d)

        # shape: Actions * T_d_num  x  6
        poly_coefs_d = np.fliplr(np.einsum('ijk,kl->lij', A_inv, constraints_d).reshape(actions_num * T_d_num, 6))

        ftraj_d = QuinticPoly1D.polyval_with_derivatives(poly_coefs_d, time_samples)[:, :, :2]

        # fill all elements of ftraj_d beyond T_d by the values of ftraj_d at T_d
        for i, td in enumerate(T_d):
            last_sample = np.where(time_samples >= td)[0][0]
            ftraj_d[i::T_d_num, last_sample+1:, :] = ftraj_d[i::T_d_num, last_sample:last_sample+1, :]

        return ftraj_d[:, :, 0], ftraj_d[:, :, 1]
