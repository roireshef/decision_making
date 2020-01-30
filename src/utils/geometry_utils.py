from decision_making.src.global_constants import SPLINE_POINT_DEVIATION
from typing import Union, Tuple

import numpy as np
from scipy import interpolate as interp
from scipy.interpolate.fitpack2 import UnivariateSpline

from decision_making.src.exceptions import OutOfSegmentBack, OutOfSegmentFront
from decision_making.src.planning.utils.math_utils import Math


class Euclidean:
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
        points_matrix = points.reshape(np.prod(points.shape[:-1]).astype(np.int), points.shape[-1]).astype(np.float64)
        # matrix that holds the progress of projection of each point [rows] on each segment [columns]

        # Instead of doing the following code, we improve its time by doing some math and avoiding the apply_along_axis()
        """
        # progress_matrix = np.apply_along_axis(
        #     lambda point_i: np.sum((point_i - segments_start) * segments_vec, axis=1) / segments_length ** 2, -1, points_matrix)
        """
        progress_matrix = (np.dot(points_matrix, segments_vec.T) - np.sum(segments_start * segments_vec, axis=1)) / (
                    segments_length * segments_length)

        # clips projections to the range [0,1]
        clipped_progress = np.clip(progress_matrix, a_min=0, a_max=1)


        # Instead of doing the following code, we apply, again, some math and avoid using 3D tensors
        """
        # 3D tensor of [points, segments, clipped projection coordinates]
        # clipped_projections = segments_start[np.newaxis, :, :] + np.einsum('ij,jk->ijk', clipped_progress, segments_vec)

        # euclidean distance from each point [rows] to the corresponding clipped-projection on each segment [columns]
        # distances_to_clipped_projections = np.linalg.norm(points_matrix[..., np.newaxis, :] - clipped_projections, axis=-1)
        """
        distances_to_clipped_projections = np.sum(points_matrix * points_matrix, axis=-1)[:, None] + \
                                           np.sum(segments_start * segments_start, axis=-1)[None, :] + \
                                           clipped_progress * clipped_progress * np.sum(segments_vec * segments_vec, axis=-1)[None, :] - \
                                           2 * np.dot(points_matrix, segments_start.T) + \
                                           2 * clipped_progress * np.sum(segments_start * segments_vec, axis=-1)[None, :] - \
                                           2 * clipped_progress * np.dot(points_matrix, segments_vec.T)

        # assert np.all(np.abs(distances_to_clipped_projections_ - distances_to_clipped_projections ** 2) < 1e-3)
        #
        # 1D for each point, hold the index of the closest segment
        closest_segment_idxs = np.argmin(distances_to_clipped_projections, axis=-1)

        # 1D for each point, hold the clipped-progress (in the range [0, 1]) on the closest segment
        closest_clipped_progress = clipped_progress[np.arange(len(points_matrix)), closest_segment_idxs]

        is_point_in_front_of_curve = (closest_segment_idxs == num_segments - 1) * progress_matrix[:, -1] > 1
        if np.any(is_point_in_front_of_curve, axis=-1):
            raise OutOfSegmentFront("Can't project point(s) %s on curve [%s, ..., %s]" % (
                str(points_matrix[[is_point_in_front_of_curve]]).replace('\n', ', '),
                str(path_points[0]), str(path_points[-1])))

        is_point_in_back_of_curve = (closest_segment_idxs == 0) * progress_matrix[:, 0] < 0
        if np.any(is_point_in_back_of_curve, axis=-1):
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
    def homo_matrix_2d_rotation_translation(rotation_angle: float, translation: np.array) -> np.array:
        """
        Generates a 2D homogeneous matrix for cartesian frame projections, when translation is operated after rotation:
        (y = T*R*x)
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
    def homo_matrix_2d_translation_rotation(rotation_angle: float, translation: np.array) -> np.array:
        """
        Generates a 2D homogeneous matrix for cartesian frame projections, when translation is operated before rotation:
        (y = R*T*x)
        :param rotation_angle: yaw in radians
        :param translation: a [x, y] translation vector
        :return: a 3x3 numpy matrix for projection (translation+rotation)
        """
        cos, sin = np.cos(rotation_angle), np.sin(rotation_angle)
        return np.array([
            [cos, -sin, translation[0]*cos - translation[1]*sin],
            [sin,  cos, translation[0]*sin + translation[1]*cos],
            [0, 0, 1]
        ])

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
    def convert_global_to_relative_frame(global_pos: np.array, global_yaw: Union[float, np.array],
                                         frame_position: np.array, frame_orientation: float) -> \
            Tuple[np.ndarray, Union[float, np.array]]:
        """
        Convert point in global-frame to a point in relative-frame
        :param global_pos: (x,y) point(s) in global coordinate system. Either an array array of size [2,] or a matrix
         of size [?, 2]
        :param global_yaw: [rad] yaw in global coordinate system: either scalar or array
        :param frame_position: translation (shift) of coordinate system to project on: (x,y) array of size [2,]
        :param frame_orientation: orientation of the coordinate system to project on: yaw scalar
        :return: The point(s) and yaw relative to coordinate system specified by <frame_position> and
        <frame_orientation>. The shapes of global_pos and yaw are kept.
        """
        transformation_matrix = CartesianFrame.homo_matrix_2d_translation_rotation(-frame_orientation, -frame_position)[:-1]
        ones = np.ones((global_pos.shape[0], 1)) if global_pos.ndim > 1 else np.array([1])
        relative_pos = np.dot(transformation_matrix, np.concatenate((global_pos, ones), axis=-1).T)
        relative_yaw = global_yaw - frame_orientation
        return relative_pos, relative_yaw

    @staticmethod
    def convert_relative_to_global_frame(relative_pos: np.array, relative_yaw: Union[float, np.array],
                                         frame_position: np.array, frame_orientation: float) -> \
            Tuple[np.ndarray, Union[float, np.array]]:
        """
        Convert point in relative coordinate-system to a global coordinate-system
        :param relative_pos: (x,y) point(s) in relative coordinate system. Either an array array of size [2,] or a
        matrix of size [?, 2]
        :param relative_yaw: [rad] yaw in relative coordinates: either scalar or array
        :param frame_position: translation (shift) of coordinate system to project from: (x,y) array of size [2,]
        :param frame_orientation: orientation of the coordinate system to project from: yaw scalar
        :return: the point(s) in the 2D global coordinate system numpy Nx3 and global yaw [rad].
                The shapes of global_pos and yaw are kept.
        """
        transformation_matrix = CartesianFrame.homo_matrix_2d_rotation_translation(frame_orientation, frame_position)[:-1]
        ones = np.ones((relative_pos.shape[0], 1)) if relative_pos.ndim > 1 else np.array([1])
        relative_pos = np.dot(transformation_matrix, np.concatenate((relative_pos, ones), axis=-1).T)
        relative_yaw = relative_yaw + frame_orientation
        return relative_pos, relative_yaw
