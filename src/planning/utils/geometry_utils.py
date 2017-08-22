from typing import Union

import numpy as np
from scipy import interpolate as interp

from decision_making.src.global_constants import *
from decision_making.src.planning.utils import tf_transformations
from decision_making.src.planning.utils.columns import *
from decision_making.src.planning.utils.math import Math


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
        :param rotation_matrix: 3D rotation matrix
        :param translation: a [x, y, z] translation vector
        :return: a 3x3 numpy matrix for projection (translation+rotation)
        """
        t = translation.reshape([-1, 1])  # translation vector
        return np.vstack((np.hstack((rotation_matrix, t[:3, 1])), [0, 0, 0, 1]))

    @staticmethod
    def homo_matrix_3d_from_quaternion(quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """
        Generates a 3D homogeneous matrix for cartesian frame projections
        :param quaternion: numpy array of quaternion rotation
        :param translation: a [x, y, z] translation vector
        :return: a 3x3 numpy matrix for projection (translation+rotation)
        """
        # TODO: reimplement without tf (ROS) dependency, if needed
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

        theta = np.arctan2(xy_dot[:, 1], xy_dot[:, 0])  # orientation
        k = (xy_dot * xy_dotdot[:, [1, 0]]).dot([1, -1]) / (np.linalg.norm(xy_dot, axis=1) ** 3)  # curvature

        theta_col = theta.reshape([-1, 1])
        k_col = k.reshape([-1, 1])

        k_tag = np.diff(k_col, axis=0)
        k_tag_col = np.concatenate((k_tag, [k_tag[-1]])).reshape([-1, 1])

        return np.concatenate((xy_points, theta_col, k_col, k_tag_col), 1)

    @staticmethod
    def resample_curve(curve: np.ndarray, step_size: float, desired_curve_len: Union[None, float] = None,
                       preserve_step_size: bool = False, interp_type: str = 'linear') -> (np.ndarray, float):
        """
        Takes a discrete set of points [x, y] and perform interpolation (with a constant step size). \n
        Note: user may specify a desired final curve length. If she doesn't, it uses the total distance travelled over a
        linear fit of 'curve'. \n
        :param curve:
        :param step_size:
        :param desired_curve_len:
        :param preserve_step_size: If True - once desired_curve_len is not a multiply of step_size, desired_curve_len is
        trimmed to its nearest multiply. If False - once desired_curve_len is not a multiply of step_size, the step_size
        is efficiently "stretched" to preserve the exact given desired_curve_len.
        :param interp_type: order of interpolation (either: 'linear', 'bilinear', 'cubic', etc)
        :return: (the resampled curve, the effective step size)
        """

        # accumulated distance travelled on the original curve (linear fit over the discrete points)
        org_s = np.concatenate(([.0], np.cumsum(np.linalg.norm(np.diff(curve[:, :2], axis=0), axis=1))))

        # if desired curve length is not specified, use the total distance travelled over curve (via linear fit)
        if desired_curve_len is None:
            desired_curve_len = org_s[-1]

        # this method does not support extrapolation
        if desired_curve_len > org_s[-1]:
            raise ValueError('resampled_curve_len (' + str(desired_curve_len) + ') is greater than the accumulated ' +
                             'actual arc length (' + org_s + ')')

        # handles cases where we suffer from floating point division inaccuracies
        num_samples = Math.div(desired_curve_len, step_size) + 1

        # if step_size must be preserved but desired_curve_len is not a multiply of step_size - trim desired_curve_len
        if Math.mod(desired_curve_len, step_size) > 0 and preserve_step_size:
            desired_curve_len = (num_samples - 1) * step_size
            effective_step_size = step_size
        else:
            effective_step_size = desired_curve_len / (num_samples - 1)

        # the newly sampled accumulated distance travelled (used for interpolation)
        s = np.linspace(0, desired_curve_len, num_samples)

        # interpolation each of the dimensions in curve over samples from s
        interp_curve = np.zeros(shape=[len(s), curve.shape[1]])
        for col in range(curve.shape[1]):
            curve_func = interp.interp1d(org_s, curve[:, col], kind=interp_type)
            interp_curve[:, col] = curve_func(s)

        return interp_curve, effective_step_size

    @staticmethod
    def calc_point_segment_dist(p: np.ndarray, p_start: np.ndarray, p_end: np.ndarray) -> (int, float, float):
        """
        Given point p and directed segment p1->p2, calculate:
            1. from which side p is located relatively to the line p1->p2,
            2. the closest distance from p to the segment,
            3. length of the projection of p on the segment (zero if the projection is outside the segment).
        :param p: 2D Point
        :param p_start: first edge of 2D segment
        :param p_end: second edge of 2D segment
        :return: signed distance between the point p and the segment p1->p2; length of the projection of p on the segment
        """
        segment = p_end - p_start
        segment_start_to_p = p - p_start
        segment_p_to_end = p_end - p
        if segment[0] == 0 and segment[1] == 0:
            return 0, np.linalg.norm(segment_start_to_p), 0
        dot1 = np.dot(segment, segment_start_to_p)
        dot2 = np.dot(segment, segment_p_to_end)
        normal = np.array([-segment[1], segment[0]])  # normal of v toward left if v looks up
        dotn = np.dot(normal, segment_start_to_p)
        sign = np.sign(dotn)
        proj = 0
        if dot1 > 0 and dot2 > 0:  # then p is between p1,p2, so calc dist to the line
            one_over_vnorm = 1. / np.linalg.norm(segment)
            dist = dotn * one_over_vnorm * sign  # always >= 0
            proj = dot1 * one_over_vnorm  # length of projection of v1 on v
        elif dot1 <= 0:
            dist = np.linalg.norm(segment_start_to_p)
        else:
            dist = np.linalg.norm(segment_p_to_end)
        return sign, dist, proj

    @staticmethod
    def get_vector_in_objective_frame(target_vector: np.array, ego_position: np.array,
                                      ego_orientation: Union[float, np.array]):
        """
        :param target_vector: (x,y,z) array of size [3,]
        :param ego_position: translation of ego frame: (x,y,z) array of size [3,]
        :param ego_orientation: orientation of ego frame. Can be either yaw scalar, or quaternion vector
        :return:
        """
        if hasattr(ego_orientation, "__len__"):
            # Orientation is quaternion numpy array
            quaternion = ego_orientation
        else:
            # Orientation contains yaw
            quaternion = tf_transformations.quaternion_from_euler(0, 0, ego_orientation, 'ryxz')

        car_rotation = tf_transformations.quaternion_matrix(quaternion)
        car_position = np.array(ego_position).reshape([3, -1])
        if len(target_vector.shape) == 1:
            target_vector = target_vector.reshape([3, -1])
        elif target_vector.shape[0] != 3:
            target_vector = target_vector.transpose()
        target_pos_in_obj_frame = np.dot(np.linalg.pinv(car_rotation[0:3, 0:3]), target_vector - car_position)

        return target_pos_in_obj_frame

    @staticmethod
    def convert_yaw_to_quaternion(yaw: float):
        """
        :param yaw: angle in [rad]
        :return: quaternion
        """
        return tf_transformations.quaternion_from_euler(0, 0, yaw, 'ryxz')


class FrenetMovingFrame:
    """
    A 2D Frenet moving coordinate frame. Within this class: fpoint, fstate and ftrajectory are in frenet frame;
    cpoint, cstate, and ctrajectory are in cartesian coordinate frame
    """

    def __init__(self, curve_xy: np.ndarray, resolution=TRAJECTORY_ARCLEN_RESOLUTION):
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

    def cpoint_to_fpoint(self, cpoint: np.ndarray) -> np.ndarray:
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
        sx, dx = fpoint[0], fpoint[1]
        s_idx = self.sx_to_s_idx(sx)
        h = self.get_homo_matrix_2d(s_idx)  # projection from global coord-frame to the Frenet-origin
        abs_point = np.dot(h, [0, dx, 1])[:2]

        return np.array(abs_point)

    # currently this is implemented. We should implement the next method and make this one wrap a single trajectory
    # and send it to the next method.
    def ftrajectory_to_ctrajectory(self, ftrajectory: np.ndarray) -> np.ndarray:
        """
        Transforms Frenet-frame trajectory to cartesian-frame trajectory, using tensor operations
        :param ftrajectory: a numpy matrix of rows of the form [sx, sv, sa, dx, dv, da]
        :return: a numpy matrix of rows of the form [x, y, theta, v, a, k] in car's coordinate frame
        """
        return self.ftrajectories_to_ctrajectories(np.array([ftrajectory]))[0]

    def ftrajectories_to_ctrajectories(self, ftrajectories: np.ndarray) -> np.ndarray:
        """
        Transforms Frenet-frame trajectories to cartesian-frame trajectories, using tensor operations
        :param ftrajectories: a numpy tensor with dimensions [0 - trajectories, 1 - trajectory points,
            2 - frenet-state [sx, sv, sa, dx, dv, da])
        :return: a numpy tensor with dimensions [0 - trajectories, 1 - trajectory points,
            2 - cartesian-state [x, y, theta, v, a, k]) in car's coordinate frame
        """
        num_t = ftrajectories.shape[0]
        num_p = ftrajectories.shape[1]

        s_x = ftrajectories[:, :, F_SX]
        s_v = ftrajectories[:, :, F_SV]
        s_a = ftrajectories[:, :, F_SA]
        d_x = ftrajectories[:, :, F_DX]
        d_v = ftrajectories[:, :, F_DV]
        d_a = ftrajectories[:, :, F_DA]

        s_idx = np.array(np.divide(s_x, self._ds), dtype=int)  # index of frenet-origin
        theta_r = self._curve[s_idx, R_THETA]  # yaw of frenet-origin
        k_r = self._curve[s_idx, R_K]  # curvature of frenet-origin
        k_r_tag = self._curve[s_idx, R_K_TAG]  # derivative by distance (curvature is already in ds units)

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


class Dynamics:
    """
    predicting location & velocity for moving objects
    """

    @staticmethod
    def predict_dynamics(x: float, y: float, yaw: float, v_x: float, v_y: float, accel_lon: float, turn_radius: float,
                         dt: float) -> tuple((float, float, float, float, float)):
        """
        Predict the object's location, yaw and velocity after a given time period.
        The object may accelerate and move in circle with given radius.
        :param x: starting x in meters
        :param y: starting y in meters
        :param yaw: starting yaw in radians
        :param v_x: starting v_x in m/s
        :param v_y: starting v_y in m/s
        :param accel_lon: constant longitudinal acceleration in m/s^2
        :param turn_radius: in meters; positive CW, negative CCW, zero means straight motion
        :param dt: time period in seconds
        :return: goal x, y, yaw, v_x, v_y
        """
        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)
        start_vel = np.sqrt(v_x * v_x + v_y * v_y)
        # if the object will stop before goal_timestamp, then set dt to be until the stop time
        if accel_lon < -0.01 and accel_lon * dt < -start_vel:
            dt = start_vel / (-accel_lon)

        if turn_radius is not None and turn_radius != 0:  # movement by circle arc (not straight)
            # calc distance the object passes until goal_timestamp
            dist = start_vel * dt + 0.5 * accel_lon * dt * dt
            # calc yaw change (turn angle) in radians
            d_yaw = dist / turn_radius
            goal_yaw = yaw + d_yaw
            sin_next_yaw = np.sin(goal_yaw)
            cos_next_yaw = np.cos(goal_yaw)
            # calc the circle center
            circle_center = [x - turn_radius * sin_yaw, y + turn_radius * cos_yaw]
            # calc the end location
            goal_x = circle_center[0] + turn_radius * sin_next_yaw
            goal_y = circle_center[1] - turn_radius * cos_next_yaw
            # calc the end velocity
            end_vel = start_vel + accel_lon * dt
            goal_v_x = end_vel * cos_next_yaw
            goal_v_y = end_vel * sin_next_yaw
        else:  # straight movement
            acc_x = accel_lon * cos_yaw
            acc_y = accel_lon * sin_yaw
            goal_x = x + v_x * dt + 0.5 * acc_x * dt * dt
            goal_y = y + v_y * dt + 0.5 * acc_y * dt * dt
            goal_v_x = v_x + dt * acc_x
            goal_v_y = v_y + dt * acc_y
            goal_yaw = yaw

        return tuple((goal_x, goal_y, goal_yaw, goal_v_x, goal_v_y))

    @staticmethod
    def rotate_and_shift_point(x: float, y: float, cosa: float, sina: float, dx: float, dy: float) \
            -> tuple((float, float)):
        """
        calculate new point location after rotation & shift
        :param x: original point location
        :param y:
        :param cosa: cos of rotation angle
        :param sina: sin of rotation angle
        :param dx: shift
        :param dy:
        :return: new point location
        """
        return tuple((x * cosa - y * sina + dx, x * sina + y * cosa + dy))

    @staticmethod
    def rotate_and_shift_points(points: np.ndarray, cosa: float, sina: float, dx: float, dy: float) -> np.ndarray:
        """
        calculate new point location after rotation & shift
        :param points: Nx3 matrix of the original points
        :param cosa: cos of rotation angle
        :param sina: sin of rotation angle
        :param dx: shift
        :param dy:
        :return: Nx3 matrix: new points location
        """
        return np.c_[points[:, 0] * cosa - points[:, 1] * sina + dx,
                     points[:, 0] * sina + points[:, 1] * cosa + dy,
                     points[:, 2]]
