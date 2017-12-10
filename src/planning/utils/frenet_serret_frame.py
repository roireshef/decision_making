import numpy as np

from decision_making.src.global_constants import TRAJECTORY_ARCLEN_RESOLUTION
from decision_making.src.planning.types import FP_SX, FP_DX, CartesianPoint2D, \
    FrenetTrajectory, CartesianPath2D, FrenetTrajectories, CartesianExtendedTrajectories, CartesianPoint3D, FS_SX, \
    FS_SV, FS_SA, FS_DX, FS_DV, FS_DA, C_Y, C_X, CartesianExtendedTrajectory, FrenetPoint
from mapping.src.model.constants import EPSILON
from mapping.src.transformations.geometry_utils import CartesianFrame, Euclidean


class FrenetSerret2DFrame:
    def __init__(self, points: CartesianPath2D, ds: float = TRAJECTORY_ARCLEN_RESOLUTION):
        # TODO: move this outside
        self.s_max = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1), axis=0)

        self.O, _ = CartesianFrame.resample_curve(curve=points, step_size=ds,
                                                  desired_curve_len=self.s_max,
                                                  preserve_step_size=True)

        self.ds = ds
        self.T, self.N, self.k = FrenetSerret2DFrame._fit_frenet(self.O)

    def get_yaw(self, s: np.ndarray):
        _, T_r, _, _, _ = self._taylor_interp(s)
        return np.arctan2(T_r[..., C_Y], T_r[..., C_X])

    def ftrajectory_to_ctrajectory(self, ftrajectory: FrenetTrajectory) -> CartesianExtendedTrajectory:
        """
        Transforms Frenet-frame trajectory to cartesian-frame trajectory, using tensor operations
        :param ftrajectory: a frenet-frame trajectory
        :return: a cartesian-frame trajectory (given in the coordinate frame of self.points)
        """
        return self.ftrajectories_to_ctrajectories(np.array([ftrajectory]))[0]

    def ctrajectory_to_ftrajectory(self, ctrajectory: CartesianExtendedTrajectory) -> FrenetTrajectory:
        """
        Transforms Cartesian-frame trajectory to Frenet-frame trajectory, using tensor operations
        :param ctrajectory: a cartesian-frame trajectory (in the coordinate frame of self.points)
        :return: a frenet-frame trajectory
        """
        return self.ctrajectories_to_ftrajectories(np.array([ctrajectory]))[0]

    def fpoint_to_cpoint(self, fpoints: FrenetPoint) -> CartesianPoint2D:
        return self.fpoints_to_cpoints(fpoints[np.newaxis, :])[0]

    def cpoint_to_fpoint(self, cpoints: CartesianPoint2D) -> FrenetPoint:
        return self.cpoints_to_fpoints(cpoints[np.newaxis, :])[0]

    def fpoints_to_cpoints(self, fpoints: FrenetTrajectory) -> CartesianPath2D:
        """
        Transforms frenet-frame point to cartesian-frame point (using self.curve) \n
        :param fpoint: numpy array of frenet-point [sx, dx]
        :return: cartesian-frame point [x, y]
        """
        a_s, _, N_s, _, _ = self._taylor_interp(fpoints[:, FP_SX])
        return a_s + N_s * fpoints[:, [FP_DX]]

    def cpoints_to_fpoints(self, cpoints: CartesianPath2D) -> FrenetTrajectory:
        s, a_s, _, N_s, _, _ = np.apply_along_axis(self._project_cartesian_point, 0, cpoints)

        # project cpoints on the normals at a_s
        d = np.einsum('ij,ij->i', cpoints - a_s, N_s)

        return np.c_[s, d]

    def ftrajectories_to_ctrajectories(self, ftrajectories: FrenetTrajectories) -> CartesianExtendedTrajectories:
        """
        Transforms Frenet-frame trajectories to cartesian-frame trajectories, using tensor operations
        :param ftrajectories: Frenet-frame trajectories (tensor)
        :return: Cartesian-frame trajectories (tensor)
        """
        num_t = ftrajectories.shape[0]
        num_p = ftrajectories.shape[1]

        s_x = ftrajectories[:, :, FS_SX]
        s_v = ftrajectories[:, :, FS_SV]
        s_a = ftrajectories[:, :, FS_SA]
        d_x = ftrajectories[:, :, FS_DX]
        d_v = ftrajectories[:, :, FS_DV]
        d_a = ftrajectories[:, :, FS_DA]

        a_r, T_r, N_r, k_r, k_r_tag = self._taylor_interp(s_x)
        theta_r = np.arctan2(T_r[..., C_Y], T_r[..., C_X])

        cached_term = 1 - k_r * d_x  # pre-compute terms to use below
        d_tag = d_v / s_v  # 1st derivative of d_x by distance
        tan_delta_theta = d_tag / cached_term
        delta_theta = np.arctan2(d_tag, cached_term)
        cos_delta_theta = np.cos(delta_theta)

        # compute v_x (velocity in the heading direction)
        v_x = np.divide(s_v * cached_term, cos_delta_theta)

        # compute k_x (curvature)
        d_tagtag = (d_a - d_tag * s_a) / (s_v ** 2)  # 2nd derivative of d_x by distance
        k_x = d_tagtag + (k_r_tag * k_r * d_tag) * tan_delta_theta * cos_delta_theta ** 3 / cached_term ** 2 + \
              k_r * cos_delta_theta / cached_term

        # compute a_x (curvature)
        delta_theta_tag = v_x / s_v * k_x - k_r  # derivative of delta_theta (via chain rule: d(sx)->d(t)->d(s))
        a_x = s_a * cached_term / cos_delta_theta + \
              s_v ** 2 / cos_delta_theta * (cached_term * tan_delta_theta * delta_theta_tag - (k_r_tag * k_r * d_tag))

        # compute position (cartesian)
        pos_x = a_r + N_r * d_x[..., np.newaxis]

        # compute theta_x
        theta_x = theta_r + delta_theta

        return np.concatenate((pos_x.reshape([num_t, num_p, 2]), theta_x.reshape([num_t, num_p, 1]),
                               v_x.reshape([num_t, num_p, 1]), a_x.reshape([num_t, num_p, 1]),
                               k_x.reshape([num_t, num_p, 1])), axis=2)

    def ctrajectories_to_ftrajectories(self, ctrajectories: CartesianExtendedTrajectories) -> FrenetTrajectories:
        """
        Transforms Cartesian-frame trajectories to Frenet-frame trajectories, using tensor operations
        :param ctrajectories: Cartesian-frame trajectories (tensor)
        :return: Frenet-frame trajectories (tensor)
        """
        pass

    ## UTILITIES ##

    def _project_cartesian_point(self, point: CartesianPoint2D) -> \
            (float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """Given a 2D point in cartesian frame (same origin as self.O) this function uses taylor approximation to return
        s*, a(s*), T(s*), N(s*), k(s*), k'(s*), where:
        s* is the progress along the curve where the point is projected
        a(s*) is the Cartesian-coordinates (x,y) of the projection on the curve,
        T(s*) is the tangent unit vector (dx,dy) of the projection on the curve
        N(s*) is the normal unit vector (dx,dy) of the projection on the curve
        k(s*) is the curvature (scalar) - assumed to be constant in the neighborhood of the points in self.O and thus
        taken from the nearest point in self.O
        k'(s*) is the derivative of the curvature (by distance d(s))
        """
        # TODO: replace this with GD for finding more accurate s
        O_idx, delta_s = Euclidean.project_on_piecewise_linear_curve(np.array([point]), self.O)[0]
        s_approx = (O_idx + delta_s) * self.ds

        s_exact = np.zeros(cpoints.shape[0])
        for cpoint_idx in range(cpoints.shape[0]):
            step = 1
            s_approx = (O_idx[cpoint_idx] + delta_s[cpoint_idx]) * self.ds
            while step > EPSILON:
                a_s, _, N_s, k_s, _ = self._taylor_interp(np.array([s_approx]))
                k = k_s[cpoint_idx]
                if k < EPSILON:
                    break
                N = N_s[cpoint_idx]  # normal vector in s_approx
                A = a_s[cpoint_idx]  # cartesian point of s_approx
                P = cpoints[cpoint_idx+1][:2]  # input cartesian point
                radius = 1/k  # circle radius according to the curvature
                center_to_P = P - A + N * radius  # vector from the circle center to P
                cos = np.dot(N, center_to_P) / np.linalg.norm(center_to_P)  # cos(angle between N and this vector)
                step = np.math.acos(cos) * radius  # arc length from A to the new guess point
                s_approx = s_approx + step  # next s_approx of the current point
            s_exact[cpoint_idx] = s_approx

        a_s, T_s, N_s, k_s, k_s_tag = self._taylor_interp(np.array([s_approx]))[0]
        return s_approx, a_s, T_s, N_s, k_s, k_s_tag

    def _taylor_interp(self, s: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """Given arbitrary s tensor (of shape D) of values in the range [0, self.s_max], this function uses taylor
        approximation to return a(s), T(s), N(s), k(s), k'(s), where:
        a(s) is the map to Cartesian-frame (a point on the curve. will have shape of Dx2),
        T(s) is the tangent unit vector (will have shape of Dx2)
        N(s) is the normal unit vector (will have shape of Dx2)
        k(s) is the curvature (scalar) - assumed to be constant in the neighborhood of the points in self.O and thus
        taken from the nearest point in self.O (will have shape of D)
        k'(s) is the derivative of the curvature (by distance d(s))
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

        k_s = self.k[O_idx] + \
              delta_s * np.gradient(self.k, axis=0)[O_idx] + \
              delta_s ** 2 / 2 * np.gradient(np.gradient(self.k, axis=0), axis=0)[O_idx]

        k_s_tag = np.gradient(self.k, axis=0)[O_idx] + delta_s * np.gradient(np.gradient(self.k, axis=0), axis=0)[O_idx]

        return a_s, T_s, N_s, k_s[..., 0], k_s_tag[..., 0]

    @staticmethod
    def _fit_frenet(xy: CartesianPath2D):
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
        N = FrenetSerret2DFrame._row_wise_normal(T)

        # Curvature
        cross_norm = np.abs(np.sum(dxy * FrenetSerret2DFrame._row_wise_normal(ddxy), axis=1))
        k = np.zeros(len(T))
        k[dxy_norm > 0] = np.c_[cross_norm[dxy_norm > 0]] / (np.c_[dxy_norm[dxy_norm > 0]] ** 3)
        return T, N, np.c_[k]

    @staticmethod
    def _row_wise_normal(mat: np.ndarray):
        return np.c_[-mat[:, 1], mat[:, 0]]
