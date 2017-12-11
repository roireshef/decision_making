import numpy as np

from decision_making.src.global_constants import TRAJECTORY_ARCLEN_RESOLUTION, TRAJECTORY_CURVE_INTERP_TYPE
from decision_making.src.planning.types import FP_SX, FP_DX, CartesianPoint2D, \
    FrenetTrajectory, CartesianPath2D, FrenetTrajectories, CartesianExtendedTrajectories, FS_SX, \
    FS_SV, FS_SA, FS_DX, FS_DV, FS_DA, C_Y, C_X, CartesianExtendedTrajectory, FrenetPoint, C_THETA, C_K, C_V, C_A
from mapping.src.transformations.geometry_utils import CartesianFrame, Euclidean


class FrenetSerret2DFrame:
    def __init__(self, points: CartesianPath2D, ds: float = TRAJECTORY_ARCLEN_RESOLUTION):
        # TODO: move this outside
        self.s_max = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1), axis=0)

        self.O, _ = CartesianFrame.resample_curve(curve=points, step_size=ds,
                                                  desired_curve_len=self.s_max,
                                                  preserve_step_size=True,
                                                  interp_type=TRAJECTORY_CURVE_INTERP_TYPE)

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
        s = np.zeros(shape=cpoints.shape[0])
        a_s = np.zeros(shape=cpoints.shape)
        N_s = np.zeros(shape=cpoints.shape)

        for i in range(len(cpoints)):
            s[i], a_s[i], _, N_s[i], _, _ = self._project_cartesian_point(cpoints[i])

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

        radius_ratio = 1 - k_r * d_x  # pre-compute terms to use below
        d_tag = d_v / s_v  # 1st derivative of d_x by distance
        tan_delta_theta = d_tag / radius_ratio
        delta_theta = np.arctan2(d_tag, radius_ratio)
        cos_delta_theta = np.cos(delta_theta)

        # compute v_x (velocity in the heading direction)
        v_x = np.divide(s_v * radius_ratio, cos_delta_theta)

        # compute k_x (curvature)
        d_tagtag = (d_a - d_tag * s_a) / (s_v ** 2)  # 2nd derivative of d_x by distance
        k_x = d_tagtag + (k_r_tag * k_r * d_tag) * tan_delta_theta * cos_delta_theta ** 3 / radius_ratio ** 2 + \
              k_r * cos_delta_theta / radius_ratio

        # compute a_x (curvature)
        delta_theta_tag = radius_ratio / np.cos(delta_theta) * k_x - k_r  # derivative of delta_theta (via chain rule: d(sx)->d(t)->d(s))
        a_x = s_a * radius_ratio / cos_delta_theta + \
              s_v ** 2 / cos_delta_theta * (radius_ratio * tan_delta_theta * delta_theta_tag - (k_r_tag * k_r * d_tag))

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
        pos_x = ctrajectories[:, :, [C_X, C_Y]]
        theta_x = ctrajectories[:, :, C_THETA]
        k_x = ctrajectories[:, :, C_K]
        v_x = ctrajectories[:, :, C_V]
        a_x = ctrajectories[:, :, C_A]

        new_shape = np.append(ctrajectories.shape[:2], [2])
        s_x = np.zeros(shape=new_shape[:2])
        a_r = np.zeros(shape=new_shape)
        T_r = np.zeros(shape=new_shape)
        N_r = np.zeros(shape=new_shape)
        k_r = np.zeros(shape=new_shape[:2])
        k_r_tag = np.zeros(shape=new_shape[:2])

        for i in range(ctrajectories.shape[0]):
            for j in range(ctrajectories.shape[1]):
                s_x[i,j], a_r[i,j], T_r[i,j], N_r[i,j], k_r[i,j], k_r_tag[i,j] = \
                    self._project_cartesian_point(ctrajectories[i,j,[C_X, C_Y]])

        d_x = np.einsum('tpi,tpi->tp', ctrajectories[:, :, [C_X, C_Y]] - a_r, N_r)

        radius_ratio = 1 - k_r * d_x  # pre-compute terms to use below

        theta_r = np.arctan2(T_r[..., C_Y], T_r[..., C_X])
        delta_theta = theta_x - theta_r

        s_v = v_x * np.cos(delta_theta) / radius_ratio
        d_v = v_x * np.sin(delta_theta)

        delta_theta_tag = radius_ratio / np.cos(delta_theta) * k_x - k_r  # derivative of delta_theta (via chain rule: d(sx)->d(t)->d(s))

        d_tag = (radius_ratio * np.sin(delta_theta)) ** 2
        d_tag_tag = -(k_r_tag * d_x + k_r * d_tag) * np.tan(delta_theta) + \
                    radius_ratio / np.cos(delta_theta) ** 2 * (k_x * radius_ratio/np.cos(delta_theta) - k_r)

        s_a = (a_x - s_v ** 2 / np.cos(delta_theta) *
               (radius_ratio * np.tan(delta_theta) * delta_theta_tag - (k_r_tag * d_x + k_r * d_tag))) * \
              np.cos(delta_theta) / radius_ratio
        d_a = d_tag_tag * s_v ** 2 + d_tag * s_a

        return np.dstack((s_x, s_v, s_a, d_x, d_v, d_a))

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
        O_idx, delta_s = Euclidean.project_on_piecewise_linear_curve(np.array([point]), self.O)
        s_approx = (O_idx[0] + delta_s[0]) * self.ds

        a_s, T_s, N_s, k_s, k_s_tag = self._taylor_interp(np.array([s_approx]))
        return s_approx, a_s[0], T_s[0], N_s[0], k_s[0], k_s_tag[0]

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
              delta_s * np.gradient(self.k, axis=0)[O_idx]
              # delta_s ** 2 / 2 * np.gradient(np.gradient(self.k, axis=0), axis=0)[O_idx]

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

        # SIGNED (!) Curvature
        cross_norm = np.sum(FrenetSerret2DFrame._row_wise_normal(dxy) * ddxy, axis=1)
        k = np.zeros(len(T))
        k[dxy_norm > 0] = np.c_[cross_norm[dxy_norm > 0]] / (np.c_[dxy_norm[dxy_norm > 0]] ** 3)
        return T, N, np.c_[k]

    @staticmethod
    def _row_wise_normal(mat: np.ndarray):
        return np.c_[-mat[:, 1], mat[:, 0]]
