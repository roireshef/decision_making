import numpy as np

from decision_making.src.global_constants import TRAJECTORY_ARCLEN_RESOLUTION
from decision_making.src.planning.types import FP_SX, FP_DX, CartesianPoint2D, \
    FrenetTrajectory, CartesianPath2D, FrenetTrajectories, CartesianExtendedTrajectories, CartesianPoint3D, FS_SX, \
    FS_SV, FS_SA, FS_DX, FS_DV, FS_DA
from mapping.src.transformations.geometry_utils import CartesianFrame, Euclidean


class FrenetSerretFrame:
    def __init__(self, points: CartesianPath2D, s_max: float, ds: float = TRAJECTORY_ARCLEN_RESOLUTION):
        # TODO: move this outside
        self.s_max = s_max
        self.ds = ds

        self.O, _ = CartesianFrame.resample_curve(curve=points, step_size=ds,
                                                  desired_curve_len=s_max, preserve_step_size=True)

        self.T, self.N, self.k = FrenetSerretFrame.fit_frenet(self.O)
        self.s_cumm = np.linspace(0.0, s_max, len(self.O))

    @staticmethod
    def fit_frenet(xy: CartesianPath2D):
        if xy.shape[0] == 0:
            raise ValueError('xyz array cannot be empty')

        dxy = np.gradient(xy)[0]
        ddxy = np.gradient(dxy)[0]

        # magintudes
        dxy_norm = np.linalg.norm(dxy, axis=1)

        # Tangent
        T = np.divide(dxy, np.c_[dxy_norm])

        # Derivative of Tangent
        dT = np.gradient(T)[0]
        dT_norm = np.linalg.norm(dT, axis=1)

        # Normal - robust to zero-curvature
        N = FrenetSerretFrame._row_wise_normal(T)

        # Curvature
        cross_norm = np.abs(np.sum(dxy * FrenetSerretFrame._row_wise_normal(ddxy), axis=1))
        k = np.zeros(len(T))
        k[dxy_norm > 0] = np.c_[cross_norm[dxy_norm > 0]] / (np.c_[dxy_norm[dxy_norm > 0]] ** 3)
        return T, N, np.c_[k]

    @staticmethod
    def _row_wise_normal(mat: np.ndarray):
        return np.c_[-mat[:, 1], mat[:, 0]]

    def taylor_interp(self, s: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, float):
        """Given arbitrary s tensor (of shape D) of values in the range [0, self.s_max], this function uses taylor
        approximation to return a(s), T(s), N(s), k(s), where:
        a(s) is the map to Cartesian-frame (a point on the curve. will have shape of Dx2),
        T(s) is the tangent unit vector (will have shape of Dx2)
        N(s) is the normal unit vector (will have shape of Dx2)
        k(s) is the curvature (scalar) - assumed to be constant in the neighborhood of the points in self.O and thus
        taken from the nearest point in self.O (will have shape of D)
        """
        assert np.all(np.bitwise_and(0 <= s, s <= self.s_max))

        progress_ds = s / self.ds
        O_idx = np.round(progress_ds).astype(np.int)
        delta_s = np.expand_dims((progress_ds - O_idx) * self.ds, axis=len(s.shape))

        a_s = self.O[O_idx] + \
              delta_s * self.T[O_idx] + \
              delta_s ** 2 / 2 * self.k[O_idx] * self.N[O_idx] - \
              delta_s ** 3 / 6 * self.k[O_idx] ** 2 * self.T[O_idx]

        T_s = self.T[O_idx] + \
              delta_s * self.k[O_idx] * self.N[O_idx] - \
              delta_s ** 2 / 2 * self.k[O_idx] ** 2 * self.T[O_idx]

        N_s = self.N[O_idx] - \
              delta_s * self.k[O_idx] * self.T[O_idx] - \
              delta_s ** 2 / 2 * self.k[O_idx] ** 2 * self.N[O_idx]

        return a_s, T_s, N_s, self.k[O_idx]

    def fpoints_to_cpoints(self, fpoints: FrenetTrajectory) -> CartesianPath2D:
        """
        Transforms frenet-frame point to cartesian-frame point (using self.curve) \n
        :param fpoint: numpy array of frenet-point [sx, dx]
        :return: cartesian-frame point [x, y]
        """
        a_s, _, N_s, _ = self.taylor_interp(fpoints[:, FP_SX])
        return a_s + N_s * fpoints[:, [FP_DX]]

    def cpoints_to_fpoints(self, cpoints: CartesianPoint2D) -> FrenetTrajectory:
        O_idx, delta_s = Euclidean.project_on_piecewise_linear_curve(cpoints, self.O)
        s_approx = (O_idx + delta_s) * self.ds

        # TODO: replace this with GD for finding more accurate s
        s_exact = s_approx

        a_s, _, N_s, _ = self.taylor_interp(s_exact)

        # project cpoints on the normals at a_s
        d = np.einsum('ij,ij->i', cpoints - a_s, N_s)

        return np.c_[s_exact, d]

    def ftrajectories_to_ctrajectories(self, ftrajectories: FrenetTrajectories) -> CartesianExtendedTrajectories:
        """
        Transforms Frenet-frame trajectories to cartesian-frame trajectories, using tensor operations
        :param ftrajectories: Frenet-frame trajectories (tensor)
        :return: Cartesian-frame trajectories (tensor)
        """
        s_x = ftrajectories[:, :, FS_SX]
        s_v = ftrajectories[:, :, FS_SV]
        s_a = ftrajectories[:, :, FS_SA]
        d_x = ftrajectories[:, :, FS_DX]
        d_v = ftrajectories[:, :, FS_DV]
        d_a = ftrajectories[:, :, FS_DA]

        pos_r, T_r, N_r, k_r = FrenetSerretFrame.taylor_interp(ftrajectories[:, :, FS_SX])
        theta_r =

    def ctrajectories_to_ftrajectories(self, ctrajectories: CartesianExtendedTrajectories) -> FrenetTrajectories:
        """
        Transforms Cartesian-frame trajectories to Frenet-frame trajectories, using tensor operations
        :param ctrajectories: Cartesian-frame trajectories (tensor)
        :return: Frenet-frame trajectories (tensor)
        """
        pass