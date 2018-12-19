from typing import Union, Tuple

import numpy as np
from scipy import interpolate as interp
from scipy.interpolate.fitpack2 import UnivariateSpline

from decision_making.src.map_exceptions import OutOfSegmentBack, OutOfSegmentFront
from decision_making.src.mapping.global_constants import SPLINE_POINT_DEVIATION
from decision_making.src.mapping.transformations import tf_transformations
from decision_making.src.mapping.transformations.math_utils import Math


class Euclidean:
    @staticmethod
    def project_on_segment_2d(point, seg_start, seg_end):
        # type:(np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
        """
        Projects an arbitrary point onto the segment seg_start->seg_end, or throw an error point's projection is outside
        the segment.
        :param point: 2D arbitrary point in space (2x1)
        :param seg_start: 2D init point of the segment (2x1)
        :param seg_end: 2D end point of the segment (2x1)
        :return: 2D projection of point onto the segment seg_start->seg_end
        """
        seg_vector = seg_end - seg_start
        seg_length = np.linalg.norm(seg_vector)
        # 1D progress of the projection of the point on the segment (or the line extending it)
        progress = np.dot(point - seg_start, seg_vector) / seg_length ** 2

        if progress < 0.0:
            raise OutOfSegmentBack("Can't project point [{}] on segment [{}]->[{}]".format(point, seg_start, seg_end))
        if progress > 1.0:
            raise OutOfSegmentFront("Can't project point [{}] on segment [{}]->[{}]".format(point, seg_start, seg_end))

        return seg_start + progress * seg_vector  # progress from <seg_start> towards <seg_end>

    @staticmethod
    def dist_to_segment_2d(point, seg_start, seg_end):
        # type: (np.ndarray, np.ndarray, np.ndarray) -> float
        """
        Compute distance from point to *segment* seg_start->seg_end. if the point can't be projected onto the segment,
        the distance is either from the segment's init-point or end-point
        :param point: 2D arbitrary point in space
        :param seg_start: 2D init point of the segment
        :param seg_end: 2D end point of the segment
        :return: distance from point to the segment
        """
        try:
            projection = Euclidean.project_on_segment_2d(point, seg_start, seg_end)
            return np.linalg.norm(point - projection)
        except OutOfSegmentBack:
            return np.linalg.norm(point - seg_start)
        except OutOfSegmentFront:
            return np.linalg.norm(point - seg_end)

    @staticmethod
    def signed_dist_to_line_2d(point, seg_start, seg_end):
        # type (np.ndarray, np.ndarray, np.ndarray) -> float
        """
        Compute the signed distance of point to the line extending the segment seg_start->seg_end
        :param point: 2D arbitrary point in space
        :param seg_start: 2D init point of the segment
        :param seg_end: 2D end point of the segment
        :return: signed distance from point to the line (+ if point is to the left of the line, - if to the right)
        """
        if np.linalg.norm(seg_start - seg_end) == 0.0:
            return np.linalg.norm(point - seg_start)
        seg_vector = seg_end - seg_start  # vector from segment start to its end
        normal = [-seg_vector[1], seg_vector[0]]  # normal vector of the segment at its start point
        return np.divide(np.dot(point - seg_start, normal), np.linalg.norm(normal))

    @staticmethod
    def get_indexes_of_closest_segments_to_point(xy, path_points):
        # type: (np.ndarray, np.ndarray)->np.ndarray
        """
        This functions finds the indexes of the segment before and after the nearest point to (x,y)
        :param xy: a 2d cartesian point to project
        :param path_points: numpy array of size [Nx2] of path coordinates (x,y)
        :return: the indexes of the closest segments to (x,y): numpy array of size [Nx2] N={0,1,2}
        """
        # find the closest point of the road to (x,y)
        distance_to_road_points = np.linalg.norm(np.array(path_points) - xy, axis=1)
        closest_point_ind = np.argmin(distance_to_road_points)

        # the point (x,y) should be projected either onto the segment before the closest point or onto the one after it.
        closest_point_idx_pairs = np.array([[closest_point_ind - 1, closest_point_ind],
                                            [closest_point_ind, closest_point_ind + 1]])

        # filter out non-existing indices
        closest_point_idx_pairs = closest_point_idx_pairs[np.greater_equal(closest_point_idx_pairs[:, 0], 0.0) &
                                                          np.less(closest_point_idx_pairs[:, 1], len(path_points))]

        return closest_point_idx_pairs

    @staticmethod
    def project_on_piecewise_linear_curve(points, path_points):
        # type: (np.ndarray, np.ndarray) -> (np.ndarray, np.ndarray)
        """
        Projects a set of points (any tensor shape) onto a piecewise-linear curve. In cases where a point lies within
        a funnel created by the normals of two successive segments, the point is projected on the end-point of the first
        one. Note that the inverse transformation of this projection will not result exactly in the original point!
        :param points: a set of points to project (any tensor shape)
        :param path_points: a path of 2D points to project onto (2D matrix with shape [N,2])
        :return: (tensor of segment index per point in <points>,
        tensor of progress of projection of each point in <points> on its relevant segment)
        """
        segments_vec = np.diff(path_points, axis=0)
        segments_length = np.linalg.norm(segments_vec, axis=1)
        segments_start = path_points[:-1]
        num_segments = len(segments_start)

        # points reshaped to 2D
        points_matrix = points.reshape(np.prod(points.shape[:-1]).astype(np.int), points.shape[-1])

        # matrix that holds the progress of projection of each point [rows] on each segment [columns]
        progress_matrix = np.apply_along_axis(
            lambda point_i: np.sum((point_i - segments_start) * segments_vec, axis=1) / segments_length ** 2, -1, points_matrix)

        # clips projections to the range [0,1]
        clipped_progress = np.clip(progress_matrix, a_min=0, a_max=1)

        # 3D tensor of [points, segments, clipped projection coordinates]
        clipped_projections = segments_start[np.newaxis, :, :] + np.einsum('ij,jk->ijk', clipped_progress, segments_vec)

        # euclidean distance from each point [rows] to the corresponding clipped-projection on each segment [columns]
        dinstances_to_clipped_projections = np.linalg.norm(points_matrix[..., np.newaxis, :] - clipped_projections, axis=-1)

        # 1D for each point, hold the index of the closest segment
        closest_segment_idxs = np.argmin(dinstances_to_clipped_projections, axis=-1)

        # 1D for each point, hold the clipped-progress (in the range [0, 1]) on the closest segment
        closest_clipped_progress = clipped_progress[np.arange(len(points_matrix)), closest_segment_idxs]

        is_point_in_front_of_curve = np.any((closest_segment_idxs == num_segments - 1) * progress_matrix[:, -1] > 1, axis=-1)
        if np.any(is_point_in_front_of_curve):
            raise OutOfSegmentFront("Can't project point(s) %s on curve [%s, ..., %s]" % (
                str(points_matrix[[is_point_in_front_of_curve]]).replace('\n', ', '),
                str(path_points[0]), str(path_points[-1])))

        is_point_in_back_of_curve = np.any((closest_segment_idxs == 0) * progress_matrix[:, 0] < 0, axis=-1)
        if np.any(is_point_in_back_of_curve):
            raise OutOfSegmentBack("Can't project point(s) %s on curve [%s, ..., %s]" % (
                str(points_matrix[[is_point_in_back_of_curve]]).replace('\n', ', '),
                str(path_points[0]), str(path_points[-1])))

        return closest_segment_idxs.reshape(points.shape[:-1]), closest_clipped_progress.reshape(points.shape[:-1])


class CartesianFrame:
    @staticmethod
    def homo_tensor_2d(rotation_angle, translation):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        """
        Generates a tensor of 2D homogeneous matrices for cartesian frame projections
        :param rotation_angle: numpy array of arbitrary shape [S]
        :param translation: numpy array of arbitrary shape [S,2] - last dimension is [x, y] translation vector
        :return: a numpy array of shape [S,3,3] for projection (translation+rotation)
        """
        cos, sin = np.cos(rotation_angle), np.sin(rotation_angle)
        zeros = np.zeros(rotation_angle.shape)
        ones = np.ones(rotation_angle.shape)
        return np.stack((
            np.stack((cos, -sin, translation[..., 0]), axis=-1),
            np.stack((sin, cos, translation[..., 1]), axis=-1),
            np.stack((zeros, zeros, ones), axis=-1)
                 ), axis=-2)

    @staticmethod
    def homo_matrix_2d(rotation_angle, translation):
        # type: (float, np.ndarray) -> np.ndarray
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
    def homo_matrix_3d(rotation_matrix, translation):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        """
        Generates a 3D homogeneous matrix for cartesian frame projections
        :param rotation_matrix: 3D rotation matrix
        :param translation: a [x, y, z] translation vector
        :return: a 3x3 numpy matrix for projection (translation+rotation)
        """
        t = translation.reshape([3, 1])  # translation vector
        return np.vstack((np.hstack((rotation_matrix, t)), [0, 0, 0, 1]))

    @staticmethod
    def homo_matrix_3d_from_quaternion(quaternion, translation):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        """
        Generates a 3D homogeneous matrix for cartesian frame projections
        :param quaternion: numpy array of quaternion rotation
        :param translation: a [x, y, z] translation vector
        :return: a 3x3 numpy matrix for projection (translation+rotation)
        """
        rotation_matrix = tf_transformations.quaternion_matrix(quaternion)
        return CartesianFrame.homo_matrix_3d(rotation_matrix[:3, :3], translation)

    @staticmethod
    def homo_matrix_3d_from_euler(x_rot, y_rot, z_rot, translation):
        # type: (float, float, float, np.ndarray) -> np.ndarray
        """
        Generates a 3D homogeneous matrix for cartesian frame projections
        :param x_rot: euler rotation around x-axis (0 for no rotation)
        :param y_rot: euler rotation around y-axis (0 for no rotation)
        :param z_rot: euler rotation around z-axis (0 for no rotation)
        :param translation: a [x, y, z] translation vector
        :return: a 3x3 numpy matrix for projection (translation+rotation)
        """
        quaternion = tf_transformations.quaternion_from_euler(x_rot, y_rot, z_rot)
        return CartesianFrame.homo_matrix_3d_from_quaternion(quaternion, translation)

    @staticmethod
    def add_yaw(xy_points):
        # type: (np.ndarray) -> np.ndarray
        """
        Takes a matrix of curve points ([x, y] only) and adds a yaw column
        :param xy_points: a numpy matrix of shape [n, 2]
        :return: a numpy matrix of shape [n, 3]
        """
        xy_dot = np.diff(xy_points, axis=0)
        xy_dot = np.concatenate((xy_dot, np.array([xy_dot[-1, :]])), axis=0)
        theta = np.arctan2(xy_dot[:, 1], xy_dot[:, 0])  # orientation

        return np.concatenate((xy_points, theta.reshape([-1, 1])), axis=1)

    @staticmethod
    def add_yaw_and_derivatives(xy_points):
        # type: (np.ndarray) -> np.ndarray
        """
        Takes a matrix of curve points ([x, y] only) and adds columns: [yaw, curvature, derivative of curvature]
        by computing pseudo-derivatives
        :param xy_points: a numpy matrix of shape [n, 2]
        :return: a numpy matrix of shape [n, 5]
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

        return np.concatenate((xy_points, theta_col, k_col, k_tag_col), axis=1)

    @staticmethod
    def resample_curve(curve, arbitrary_curve_sampling_points=None,step_size= None, desired_curve_len=None,
                       preserve_step_size=False,spline_order=1):
        # type: (np.ndarray,Union[None, np.array],Union[None, float],Union[None, float],bool,int) -> (UnivariateSpline , np.ndarray, Union[float, None])
        """
        Takes a discrete set of points [x, y] and perform interpolation (with a constant step size). \n
        Note: user may specify a desired final curve length. If she doesn't, it uses the total distance travelled over a
        linear fit of 'curve'. \n
        :param curve: a numpy array of points (Nx2)
        :param arbitrary_curve_sampling_points: In case we don't want to resample the curve in uniform step size, we can
         specify arbitrary sampling points
        :param step_size: the distance to sample the points in
        :param desired_curve_len: sample the curve until the desired len
        :param preserve_step_size: If True - once desired_curve_len is not a multiply of step_size, desired_curve_len is
        trimmed to its nearest multiply. If False - once desired_curve_len is not a multiply of step_size, the step_size
        is efficiently "stretched" to preserve the exact given desired_curve_len.
        :param spline_order: order of spline fit (1<=spline_order<=5)
        :return: (the spline object, the resampled curve numpy array Nx2, the effective step size [m])
        """

        # interp.UnivariateSpline enables spline order no larger than 5.
        assert 5 >= spline_order >= 1

        if step_size is None and arbitrary_curve_sampling_points is None:
            raise Exception('resample_curve must get either step_size or arbitrary_curve_sampling_points, '
                            'but none of them was provided.')

        # accumulated distance travelled on the original curve (linear fit over the discrete points)
        org_s = np.concatenate(([.0], np.cumsum(np.linalg.norm(np.diff(curve[:, :2], axis=0), axis=1))))

        # if desired curve length is not specified, use the total distance travelled over curve (via linear fit)
        if desired_curve_len is None:
            desired_curve_len = org_s[-1]

        # this method does not support extrapolation
        if arbitrary_curve_sampling_points is not None:
            if np.max(arbitrary_curve_sampling_points) > org_s[-1]:
                raise ValueError(
                    'max cell of arbitrary_curve_sampling_points ({}) is greater than the accumulated actual arc length ({})'
                        .format(np.max(arbitrary_curve_sampling_points), org_s[-1]))
        else:
            if desired_curve_len - org_s[-1] > np.finfo(np.float32).eps:
                raise ValueError('desired_curve_len ({}) is greater than the accumulated actual arc length ({})'
                                 .format(desired_curve_len, org_s[-1]))

        # the newly sampled accumulated distance travelled (used for interpolation)
        if arbitrary_curve_sampling_points is None:
            # if step_size must be preserved but desired_curve_len is not a multiply of step_size -
            # trim desired_curve_len
            if Math.mod(desired_curve_len, step_size) > 0 and preserve_step_size:
                desired_curve_len = (desired_curve_len // step_size) * step_size
                effective_step_size = step_size
            else:
                effective_step_size = desired_curve_len / (desired_curve_len // step_size)

            # set the samples vector to be with uniform step size
            desired_curve_sampling_points = np.linspace(0, desired_curve_len, int(desired_curve_len // step_size) + 1)
        else:
            # use the arbitrary samples vector from argument
            desired_curve_sampling_points = arbitrary_curve_sampling_points
            effective_step_size = None

        # spline approx. for each of the dimensions in curve over samples from 'desired_curve_sampling_points'
        resampled_curve = np.zeros(shape=[len(desired_curve_sampling_points), curve.shape[1]])
        splines = []
        smooth_factor = 0 if (spline_order == 1) else SPLINE_POINT_DEVIATION
        for col in range(curve.shape[1]):
            spline = interp.UnivariateSpline(org_s, curve[:, col], k=spline_order, s=smooth_factor*curve.shape[0])
            resampled_curve[:, col] = spline(desired_curve_sampling_points)
            splines.append(spline)

        return splines, resampled_curve, effective_step_size

    @staticmethod
    def convert_global_to_relative_frame(global_pos, global_yaw,frame_position, frame_orientation):
        # type:(np.array, Union[float, np.array],np.array, float) -> Tuple[np.ndarray, Union[float, np.array]]
        """
        Convert point in global-frame to a point in relative-frame
        :param global_pos: (x,y,z) point(s) in global coordinate system. Either an array array of size [3,] or a matrix
         of size [?, 3]
        :param global_yaw: [rad] yaw in global coordinate system: either scalar or array
        :param frame_position: translation (shift) of coordinate system to project on: (x,y,z) array of size [3,]
        :param frame_orientation: orientation of the coordinate system to project on: yaw scalar
        :return: The point(s) and yaw relative to coordinate system specified by <frame_position> and
        <frame_orientation>. The shapes of global_pos and yaw are kept.
        """
        quaternion = CartesianFrame.convert_yaw_to_quaternion(frame_orientation)  # Orientation contains yaw

        # operator that projects from global coordinate system to relative coordinate system
        H_r_g = np.linalg.inv(CartesianFrame.homo_matrix_3d_from_quaternion(quaternion, frame_position))

        # add a trailing [1] element to the position vector, for proper multiplication with the 4x4 projection operator
        # then throw it (the result from multiplication is [x, y, z, 1])
        if len(global_pos.shape) == 1:  # relative_pos is a 1D vector
            ones = [1]
            remove_ones = lambda x: x[:3]
        elif len(global_pos.shape) == 2:
            ones = np.ones([global_pos.shape[0], 1])
            remove_ones = lambda x: x[:, :3]
        else:
            raise ValueError("relative_pos cardinality (" + str(global_pos.shape) +
                             ")is not supported in convert_relative_to_global_frame")

        relative_yaw = global_yaw - frame_orientation

        return remove_ones(np.dot(np.hstack((global_pos, ones)), H_r_g.transpose())), relative_yaw

    @staticmethod
    def convert_relative_to_global_frame(relative_pos, relative_yaw, frame_position, frame_orientation):
        # type: (np.array, Union[float, np.array],np.array,float) -> Tuple[np.ndarray, Union[float, np.array]]
        """
        Convert point in relative coordinate-system to a global coordinate-system
        :param relative_pos: (x,y,z) point(s) in relative coordinate system. Either an array array of size [3,] or a
        matrix of size [?, 3]
        :param relative_yaw: [rad] yaw in relative coordinates: either scalar or array
        :param frame_position: translation (shift) of coordinate system to project from: (x,y,z) array of size [3,]
        :param frame_orientation: orientation of the coordinate system to project from: yaw scalar
        :return: the point(s) in the 3D global coordinate system numpy Nx3 and global yaw [rad].
                The shapes of global_pos and yaw are kept.
        """
        quaternion = CartesianFrame.convert_yaw_to_quaternion(frame_orientation)  # Orientation contains yaw

        # operator that projects from relative coordinate system to global coordinate system
        H_g_r = CartesianFrame.homo_matrix_3d_from_quaternion(quaternion, frame_position)

        # add a trailing [1] element to the position vector, for proper multiplication with the 4x4 projection operator
        # then throw it (the result from multiplication is [x, y, z, 1])
        if len(relative_pos.shape) == 1:  # relative_pos is a 1D vector
            ones = [1]
            remove_ones = lambda x: x[:3]
        elif len(relative_pos.shape) == 2:
            ones = np.ones([relative_pos.shape[0], 1])
            remove_ones = lambda x: x[:, :3]
        else:
            raise ValueError("relative_pos cardinality (" + str(relative_pos.shape) +
                             ")is not supported in convert_relative_to_global_frame")

        global_yaw = relative_yaw + frame_orientation

        return remove_ones(np.dot(np.hstack((relative_pos, ones)), H_g_r.transpose())), global_yaw

    @staticmethod
    def convert_angles_to_quaternion(pitch, roll, yaw):
        # type: (float, float, float)->np.ndarray
        """
        :param yaw: angle in [rad]
        :return: quaternion
        """
        return tf_transformations.quaternion_from_euler(roll, pitch, yaw)

    @staticmethod
    def convert_yaw_to_quaternion(yaw):
        # type: (float)->np.ndarray
        """
        :param yaw: angle in [rad]
        :return: quaternion
        """
        return CartesianFrame.convert_angles_to_quaternion(0, 0, yaw)


# TODO: change to matrix operations, use numpy arrays instead of individual values or tuples
class Dynamics:
    """
    predicting location & velocity for moving objects
    """

    @staticmethod
    def predict_dynamics(x, y, yaw, v_x, v_y, accel_lon, turn_radius, dt):
        # type: (float, float, float, float, float, float, float,float) -> Tuple[float, float, float, float, float]
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

        return goal_x, goal_y, goal_yaw, goal_v_x, goal_v_y
