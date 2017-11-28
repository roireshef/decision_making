import numpy as np

from decision_making.src.global_constants import TRAJECTORY_ARCLEN_RESOLUTION, TRAJECTORY_CURVE_INTERP_TYPE
from decision_making.src.planning.types import FS_SX, FS_SV, FS_SA, FS_DX, FS_DV, FS_DA, CURVE_THETA, CURVE_K, CURVE_K_TAG, \
    FP_SX, \
    FP_DX, FrenetPoint, FrenetTrajectory, CartesianExtendedTrajectory, FrenetTrajectories, CartesianExtendedTrajectories, ExtendedCurve
from mapping.src.transformations.geometry_utils import CartesianFrame


class FrenetMovingFrame:
    """
    A 2D Frenet moving coordinate frame. Within this class: fpoint, fstate and ftrajectory are in frenet frame;
    cpoint, cstate, and ctrajectory are in cartesian coordinate frame
    """

    def __init__(self, curve_xy: ExtendedCurve, resolution=TRAJECTORY_ARCLEN_RESOLUTION):
        # TODO: consider moving curve-interpolation outside of this class
        resampled_curve, self._ds = CartesianFrame.resample_curve(curve=curve_xy, step_size=resolution / 4,
                                                                  interp_type=TRAJECTORY_CURVE_INTERP_TYPE)
        resampled_curve, self._ds = CartesianFrame.resample_curve(curve=resampled_curve, step_size=resolution,
                                                                  interp_type='linear')
        self._curve = CartesianFrame.add_yaw_and_derivatives(resampled_curve)
        self._h_tensor = np.array([CartesianFrame.homo_matrix_2d(self._curve[s_idx, 2], self._curve[s_idx, 0:2])
                                   for s_idx in range(len(self._curve))])

    @property
    def curve(self):
        return self._curve.copy()

    @property
    def resolution(self):
        return self._ds

    @property
    def length(self):
        return len(self.curve)

    def sx_to_s_idx(self, sx: float):
        return int(sx / self._ds)

    def get_homo_matrix_2d(self, s_idx: float) -> np.ndarray:
        """
        Returns the homogeneuos matrix (rotation+translation) for the FrenetFrame at a point along the curve
        :param s_idx: distance travelled from the beginning of the curve (in self.__ds units)
        :return: numpy array of shape [3,3] of homogeneous matrix
        """
        if self._h_tensor.size > s_idx:
            return self._h_tensor[s_idx]
        else:
            raise ValueError('index ' + str(s_idx) + 'is not found in __h_tensor (probably __h_tensor is not cached)')

    def cpoint_to_fpoint(self, cpoint: np.ndarray) -> FrenetPoint:
        """
        Transforms cartesian-frame point [x, y] to frenet-frame point (using self.curve) \n
        :param cpoint: cartesian coordinate frame state-vector [x, y, ...]
        :return: numpy vector of FrenetPoint instance [sx, dx]
        """
        # (for each cpoint:) find the (index of the) closest point on the Frenet-curve to serve as the Frenet-origin
        norm_dists = np.linalg.norm(self._curve[:, 0:2] - cpoint, axis=1)
        s_idx = np.argmin(norm_dists)
        if s_idx.size > 1:
            s_idx = s_idx[0]

        h = self.get_homo_matrix_2d(s_idx)  # projection from global coord-frame to the Frenet-origin

        # the point in the cartesian-frame of the frenet-origin
        point = np.dot(np.linalg.inv(h), np.append(cpoint, [1]))

        # the value in the y-axis is frenet-frame dx, the x-axis is the offset in the tangential direction to the curve
        tan_offset, dx = point[0], point[1]

        if s_idx == len(self._curve) and tan_offset > 0:
            raise ArithmeticError('Extrapolation beyond the frenet curve is not supported. tangential offset is ' +
                                  str(tan_offset))

        return np.array([s_idx * self._ds, dx])

    def fpoint_to_cpoint(self, fpoint: np.ndarray) -> np.ndarray:
        """
        Transforms frenet-frame point to cartesian-frame point (using self.curve) \n
        :param fpoint: numpy array of frenet-point [sx, dx]
        :return: cartesian-frame point [x, y]
        """
        sx, dx = fpoint[FP_SX], fpoint[FP_DX]
        s_idx = self.sx_to_s_idx(sx)
        h = self.get_homo_matrix_2d(s_idx)  # projection from global coord-frame to the Frenet-origin
        abs_point = np.dot(h, [0, dx, 1])[:2]

        return np.array(abs_point)

    # currently this is implemented. We should implement the next method and make this one wrap a single trajectory
    # and send it to the next method.
    def ftrajectory_to_ctrajectory(self, ftrajectory: FrenetTrajectory) -> CartesianExtendedTrajectory:
        """
        Transforms Frenet-frame trajectory to cartesian-frame trajectory, using tensor operations
        :param ftrajectory: a frenet-frame trajectory
        :return: a cartesian-frame trajectory in car's coordinate frame
        """
        return self.ftrajectories_to_ctrajectories(np.array([ftrajectory]))[0]

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

        s_idx = np.array(np.divide(s_x, self._ds), dtype=int)  # index of frenet-origin
        theta_r = self._curve[s_idx, CURVE_THETA]  # yaw of frenet-origin
        k_r = self._curve[s_idx, CURVE_K]  # curvature of frenet-origin
        k_r_tag = self._curve[s_idx, CURVE_K_TAG]  # derivative by distance (curvature is already in ds units)

        # pre-compute terms to use below
        term1 = (1 - k_r * d_x)
        d_x_tag = d_v / s_v  # 1st derivative of d_x by distance
        d_x_tagtag = (d_a - d_x_tag * s_a) / (s_v ** 2)  # 2nd derivative of d_x by distance
        tan_theta_diff = d_x_tag / term1
        theta_diff = np.arctan2(d_x_tag, term1)
        cos_theta_diff = np.cos(theta_diff)

        # compute x, y (position)
        norm = np.reshape(np.concatenate((d_x.reshape([num_t, num_p, 1]), np.ones([num_t, num_p, 1])), axis=2),
                          [num_t, num_p, 2, 1])
        xy_abs = np.einsum('tijk, tikl -> tijl', self._h_tensor[s_idx, 0:2, 1:3], norm)

        # compute v (velocity)
        v = np.divide(s_v * term1, cos_theta_diff)

        # compute k (curvature)
        k = np.divide(cos_theta_diff ** 3 * (d_x_tagtag + tan_theta_diff * (k_r_tag * d_x + k_r * d_x_tag)) +
                      cos_theta_diff * term1, term1 ** 2)

        # compute theta
        theta_x = theta_r + theta_diff

        # compute a (acceleration) via pseudo derivative
        a = np.diff(v, axis=1)
        a_col = np.concatenate((a, np.array([a[:, -1]]).reshape(num_t, 1)), axis=1)

        return np.concatenate((xy_abs.reshape([num_t, num_p, 2]), theta_x.reshape([num_t, num_p, 1]),
                               v.reshape([num_t, num_p, 1]), a_col.reshape([num_t, num_p, 1]),
                               k.reshape([num_t, num_p, 1])), axis=2)


