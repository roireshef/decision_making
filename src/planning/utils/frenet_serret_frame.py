from dipy.tracking.metrics import frenet_serret

from decision_making.src.global_constants import TRAJECTORY_ARCLEN_RESOLUTION, TRAJECTORY_CURVE_INTERP_TYPE
from decision_making.src.planning.types import CartesianPath3D, C_X, C_Y, FP_SX, FP_DX, FrenetPoint, CartesianPoint2D, \
    FrenetTrajectory, CartesianPath2D, FrenetTrajectories, CartesianExtendedTrajectories, FS_SX, FS_SV, FS_SA, FS_DV, \
    FS_DX, FS_DA, CartesianPoint3D
from mapping.src.transformations.geometry_utils import CartesianFrame, Euclidean

import numpy as np
from scipy import interpolate as interp
import math

class FrenetSerretFrame:
    def __init__(self, points: CartesianPath2D, s_max: float, ds: float = TRAJECTORY_ARCLEN_RESOLUTION):
        # TODO: move this outside
        self.s_max = s_max
        self.ds = ds

        self.O, _ = CartesianFrame.resample_curve(curve=points, step_size=ds,
                                                  desired_curve_len=s_max, preserve_step_size=True)

        # T, N, _, self.k, _ = frenet_serret(np.c_[self.O, np.zeros(len(self.O))])
        # self.T = T[:, :(C_Y+1)]
        # self.N = N[:, :(C_Y+1)]

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
        N = FrenetSerretFrame.row_wise_normal(T)

        # Curvature
        cross_norm = np.abs(np.sum(dxy * FrenetSerretFrame.row_wise_normal(ddxy), axis=1))
        k = np.zeros(len(T))
        k[dxy_norm > 0] = np.c_[cross_norm[dxy_norm > 0]] / (np.c_[dxy_norm[dxy_norm > 0]] ** 3)
        return T, N, np.c_[k]

    @staticmethod
    def row_wise_normal(mat: np.ndarray):
        return np.c_[-mat[:, 1], mat[:, 0]]

    # @staticmethod
    # def norm(vec: np.ndarray):
    #     norm = np.linalg.norm(vec, axis=1)
    #     is_zero = np.where(norm == 0)
    #     norm[is_zero] = np.finfo(float).eps
    #
    #     return norm

    def taylor_interp(self, s: np.ndarray) -> (CartesianPoint2D, CartesianPoint3D, CartesianPoint3D, float):
        """Given arbitrary s in the range [0, self.s_max], this function uses taylor approximation to return
        a(s), T(s), N(s), k(s), where:
        a(s) is the map to Cartesian-frame (a point on the curve),
        T(s) is the tangent unit vector
        N(s) is the normal unit vector
        k(s) is the curvature (scalar) - assumed to be constant in the neighborhood of the points in self.O and thus
        taken from the nearest point in self.O
        """
        assert np.all(np.bitwise_and(0 <= s, s <= self.s_max))

        progress_ds = s / self.ds
        O_idx = np.round(progress_ds).astype(np.int)
        delta_s = ((progress_ds - O_idx) * self.ds)[:, np.newaxis]

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
        # sx, dx = fpoints[:, FP_SX], fpoints[:, FP_DX]
        # O_idx, s_leftover = (sx // self._ds).astype(np.int), sx % self._ds
        #
        # exact_O = self.O[O_idx, :2] + s_leftover * self.T[O_idx, :2]
        #
        # return exact_O + self.N[O_idx, :2] * dx

        a_s, _, N_s, _ = self.taylor_interp(fpoints[:, FP_SX])
        return a_s + N_s * fpoints[:, [FP_DX]]

    def cpoints_to_fpoints(self, cpoints: CartesianPoint2D) -> FrenetTrajectory:
        # O_idx, progress = Euclidean.project_on_piecewise_linear_curve(cpoints)
        # s_leftover = progress * self.segments_longitudes[O_idx]
        # sx = self._ds * O_idx + s_leftover
        #
        # exact_O = self.O[O_idx, :2] + s_leftover * self.T[O_idx, :2]
        #
        # # this is an approximation in cases where a point lies in the funnel of two segments
        # dx = np.linalg.norm(cpoints - exact_O, axis=1)
        # return np.dstack((sx, dx))

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
        pass