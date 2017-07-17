import numpy as np
from scipy import interpolate as interp
from src.planning.global_constants import *
from src.planning.utils.columns import *


class CartesianFrame:
    @staticmethod
    def homo_matrix_2d(rotation_angle: float, translation: np.ndarray) -> np.ndarray:
        """
        Generates a 2D homogeneous matrix for cartesian frame projections
        :param rotation_angle: yaw in radians
        :param translation: a [x, y] translation vector
        :return: a 3x3 numpy matrix for projection (translation+rotation)
        """
        cos, sin = np.cos(rotation_angle), np.sin(rotation_angle)
        return np.array([
            [cos, -sin, translation[0]],
            [sin, cos, translation[1]],
            [0, 0, 1]
        ])

    @staticmethod
    def homo_matrix_3d(rotation_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """
        Generates a 3D homogeneous matrix for cartesian frame projections
        :param rotation_angle: 3D rotation matrix
        :param translation: a [x, y, z] translation vector
        :return: a 3x3 numpy matrix for projection (translation+rotation)
        """
        t = translation.reshape([-1, 1])  #translation vector
        return np.vstack((np.hstack((rotation_matrix, t[:3, 1])), [0, 0, 0, 1]))

    @staticmethod
    def homo_matrix_3d_from_quaternion(quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """
        Generates a 3D homogeneous matrix for cartesian frame projections
        :param quaternion: numpy array of quaternion rotation
        :param translation: a [x, y, z] translation vector
        :return: a 3x3 numpy matrix for projection (translation+rotation)
        """
        # rotation_matrix = tf.transformations.quaternion_matrix(quaternion)     #rotation matrix
        # return CartesianFrame.homo_matrix_3d(rotation_matrix, translation)
        pass

    @staticmethod
    def add_yaw_and_derivatives(xy_points: np.ndarray) -> np.ndarray:
        """
        Takes a matrix of curve points ([x, y] only) and adds columns: [yaw, curvature, derivative of curvature]
        by computing pseudo-derivatives
        :param xy_points: a numpy matrix of shape [n, 2]
        :return: a numpy matrix of shape [n, 4]
        """
        xy_dot = np.diff(xy_points, axis=0)
        xy_dotdot = np.diff(xy_dot, axis=0)

        xy_dot = np.concatenate((xy_dot, np.array([xy_dot[-1, :]])), axis=0)
        xy_dotdot = np.concatenate((np.array([xy_dotdot[0, :]]), xy_dotdot, np.array([xy_dotdot[-1, :]])), axis=0)

        theta = np.arctan2(xy_dot[:, 1], xy_dot[:, 0])                                              # orientation
        k = (xy_dot * xy_dotdot[:, [1, 0]]).dot([1, -1]) / (np.linalg.norm(xy_dot, axis=1) ** 3)    # curvature

        theta_col = theta.reshape([-1, 1])
        k_col = k.reshape([-1, 1])

        k_tag = np.diff(k_col, axis=0)
        k_tag_col = np.concatenate((k_tag, [k_tag[-1]])).reshape([-1, 1])

        return np.concatenate((xy_points, theta_col, k_col, k_tag_col), 1)

    @staticmethod
    def resample_curve(curve: np.ndarray, step_size: float, include_last_point: bool=True,
                       interp_type: str='linear') -> np.ndarray:
        """
        Takes a discrete set of points [x, y] and perform interpolation (with a constant step size)
        :param curve: numpy array of shape [n, k] where the first two columns are x and y coordinates
        :param step_size: interpolation (resampling) step-size
        :param include_last_point: whether to include the last point in the curve
        :param sampling_points:
        :param normalize_sampling_points:
        :param interp_type:
        :return:
        """
        num_cols = curve.shape[1]

        arc_len_vec = np.concatenate(([.0], np.cumsum(np.linalg.norm(np.diff(curve[:, :2], axis=0), axis=1))))

        ## TODO: make this block prettier
        if include_last_point:
            last_point_factor = 1e-6
        else:
            last_point_factor = 0
        interp_arc = np.arange(0, arc_len_vec[-1]+last_point_factor, step=step_size)
        max_index_in_bounds = np.where(interp_arc <= arc_len_vec[-1])[0][-1]  # remove values out of bounds of route
        interp_arc = interp_arc[:max_index_in_bounds + 1]
        ##

        interp_route = np.zeros(shape=[len(interp_arc), num_cols])

        for col in range(num_cols):
            route_func = interp.interp1d(arc_len_vec, curve[:, col], kind=interp_type)
            interp_route[:, col] = route_func(interp_arc)

        return interp_route


class FrenetMovingFrame:
    """
    A 2D Frenet moving coordinate frame. Within this class: fpoint, fstate and ftrajectory are in frenet frame;
    cpoint, cstate, and ctrajectory are in cartesian coordinate frame
    """
    # TODO: consider moving curve-interpolation outside of this class
    def __init__(self, curve_xy, resolution=TRAJECTORY_ARCLEN_RESOLUTION):
        resampled_curve = CartesianFrame.resample_curve(curve_xy, resolution, interp_type=TRAJECTORY_CURVE_INTERP_TYPE)
        self._curve = CartesianFrame.add_yaw_and_derivatives(resampled_curve)
        self._ds = resolution
        self._h_tensor = np.array([CartesianFrame.homo_matrix_2d(self._curve[s_idx, 2], self._curve[s_idx, 0:2])
                                   for s_idx in range(len(self._curve))])

    @property
    def curve(self): return self._curve.copy()

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

    def cpoint_to_fpoint(self, cpoint: np.ndarray) -> np.ndarray:
        """
        Transforms cartesian-frame point [x, y] to frenet-frame point (using self.curve) \n
        :param cpoint: cartesian coordinate frame state-vector [x, y, ...]
        :return: numpy vector of FrenetPoint instance [sx, dx]
        """
        # (for each cpoint:) find the (index of the) closest point on the Frenet-curve to serve as the Frenet-origin
        norm_dists = np.linalg.norm(self._curve[:, 0:2] - cpoint, axis=1)
        sx = np.argmin(norm_dists)
        if sx.size > 1:
            sx = sx[0]

        H = self.get_homo_matrix_2d(sx)     # projection from global coord-frame to the Frenet-origin
        dx = np.dot(np.linalg.inv(H), np.append(cpoint, [1]))[1]

        return np.array([sx * self._ds, dx])

    def fpoint_to_cpoint(self, fpoint: np.ndarray) -> np.ndarray:
        """
        Transforms frenet-frame point to cartesian-frame point (using self.curve) \n
        :param fpoint: numpy array of frenet-point [sx, dx]
        :return: cartesian-frame point [x, y]
        """
        sx, dx = fpoint[0], fpoint[1]
        s_idx = int(sx / self._ds)
        H = self.get_homo_matrix_2d(s_idx)  # projection from global coord-frame to the Frenet-origin
        abs_point = np.dot(H, [0, dx, 1])[:2]

        return np.array(abs_point)

    # currently this is implemented. We should implement the next method and make this one wrap a single trajectory
    # and send it to the next method.
    def ftrajectory_to_ctrajectory(self, ftrajectory: np.ndarray) -> np.ndarray:
        """
        Transforms Frenet-frame trajectory to cartesian-frame trajectory, using tensor operations
        :param ftrajectory: a numpy matrix of rows of the form [sx, sv, sa, dx, dv, da]
        :return: a numpy matrix of rows of the form [x, y, theta, v, a, k] in car's coordinate frame
        """
        n = len(ftrajectory)

        s_x = ftrajectory[:, F_SX]
        s_v = ftrajectory[:, F_SV]
        s_a = ftrajectory[:, F_SA]
        d_x = ftrajectory[:, F_DX]
        d_v = ftrajectory[:, F_DV]
        d_a = ftrajectory[:, F_DA]

        s_idx = np.array(s_x / self._ds, dtype=int) # index of frenet-origin
        theta_r = self._curve[s_idx, C_THETA]       # yaw of frenet-origin
        k_r = self._curve[s_idx, C_K]               # curvature of frenet-origin
        k_r_tag = self._curve[s_idx, C_K_TAG]       # derivative by distance (curvature is already in ds units)

        # pre-compute terms to use below
        term1 = (1 - k_r * d_x)
        d_x_tag = d_v / s_v                                 # 1st derivative of d_x by distance
        d_x_tagtag = (d_a - d_x_tag * s_a) / (s_v ** 2)     # 2nd derivative of d_x by distance
        tan_theta_diff = d_x_tag / term1
        theta_diff = np.arctan2(d_x_tag, term1)
        cos_theta_diff = np.cos(theta_diff)

        # compute x, y (position)
        norm = np.reshape(np.concatenate((d_x.reshape([n, 1]), np.ones([n, 1])), axis=1), [n, 2, 1])
        xy_abs = np.einsum('ijk, ikl -> ijl', self._h_tensor[s_idx, 0:2, 1:3], norm)

        # compute v (velocity)
        v = np.divide(s_v * term1, cos_theta_diff)

        # compute k (curvature)
        k = np.divide(cos_theta_diff ** 3 * (d_x_tagtag + tan_theta_diff * (k_r_tag * d_x + k_r * d_x_tag)) +
                      cos_theta_diff * term1, term1 ** 2)

        # compute theta
        theta_x = theta_r + theta_diff

        # compute a (acceleration) via pseudo derivative
        a = np.diff(v, axis=0)
        a_col = np.concatenate((a, [a[-1]]))

        return np.concatenate((xy_abs.reshape([n, 2]), theta_x.reshape([n, 1]), v.reshape([n, 1]),
                               a_col.reshape([n, 1]), k.reshape([n, 1])), axis=1)

    def ftrajectories_to_ctrajectories(self, ftrajectories: np.ndarray) -> np.ndarray:
        """
        Transforms Frenet-frame trajectories to cartesian-frame trajectories, using tensor operations
        :param ftrajectories: a numpy tensor with dimensions [0 - trajectories, 1 - trajectory points,
            2 - frenet-state [sx, sv, sa, dx, dv, da])
        :return: a numpy tensor with dimensions [0 - trajectories, 1 - trajectory points,
            2 - cartesian-state [x, y, theta, v, a, k]) in car's coordinate frame
        """
        pass