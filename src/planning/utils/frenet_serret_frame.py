from typing import Tuple

import numpy as np

from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures import LcmFrenetSerret2DFrame
from common_data.interface.py.utils.serialization_utils import SerializationUtils
from scipy.interpolate.fitpack2 import UnivariateSpline

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.global_constants import TRAJECTORY_ARCLEN_RESOLUTION, TRAJECTORY_CURVE_SPLINE_FIT_ORDER, \
    TINY_CURVATURE
from decision_making.src.planning.types import FP_SX, FP_DX, CartesianPoint2D, \
    FrenetTrajectory2D, CartesianPath2D, FrenetTrajectories2D, CartesianExtendedTrajectories, FS_SX, \
    FS_SV, FS_SA, FS_DX, FS_DV, FS_DA, C_Y, C_X, CartesianExtendedTrajectory, FrenetPoint, C_YAW, C_K, C_V, C_A, \
    CartesianVectorsTensor2D, CartesianPointsTensor2D, FrenetState2D, CartesianExtendedState
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from mapping.src.transformations.geometry_utils import CartesianFrame, Euclidean


class FrenetSerret2DFrame(PUBSUB_MSG_IMPL):
    def __init__(self, points: CartesianPath2D, T: np.ndarray, N: np.ndarray, k: np.ndarray, k_tag: np.ndarray,
                 ds: float):
        """
        This is an object used for parametrizing a curve given discrete set of points in some "global" cartesian frame,
        and then for transforming from the "global" frame to the curve's frenet frame and back.
        :param points: 2D numpy array of points sampled from a smooth curve (x,y axes; ideally a spline of high order)
        :param T: 2D numpy array of tangent unit vectors (x,y axes) of <points>
        :param N: 2D numpy array of normal unit vectors (x,y axes) of <points>
        :param k: 1D numpy array of curvature values at each point in <points>
        :param k_tag: 1D numpy array of values of 1st derivative of curvature at each point in <points>
        :param ds: the resolution of longitudinal distance along the curve (progress diff between points in <points>)
        """
        self.O = points     # origins (x,y values of points along the curve)
        self.T = T
        self.N = N
        self.k = k
        self.k_tag = k_tag
        self._ds = ds

    @property
    def ds(self):
        return self._ds

    @classmethod
    def fit(cls, spline_points: CartesianPath2D, ds: float = TRAJECTORY_ARCLEN_RESOLUTION,
            spline_order=TRAJECTORY_CURVE_SPLINE_FIT_ORDER):
        """
        Given a set of <x,y> points (in Cartesian frame) that represent a curve, this class method will fit a spline to
        them, then will sample a equi-distant discrete set of points along this spline, and extract all relevant
        statistics on them, and finally will instantiate a FrenetSerret2DFrame from those statistics.
        !!Note!! actual resolution of the sampled points on the curve won't necessarily be equal to the asked resolution
        :param spline_points: a set of points in some "global" cartesian frame
        :param ds: a resolution parameter - the desired distance between each two consecutive points after re-sampling
        :param spline_order: spline order for fitting and re-sampling the original points
        """
        splines, spline_points, effective_ds = CartesianFrame.resample_curve(curve=spline_points, step_size=ds,
                                                                             preserve_step_size=False,
                                                                             spline_order=spline_order)

        s_max = effective_ds * len(spline_points)
        T, N, k, k_tag = FrenetSerret2DFrame._fit_frenet_from_splines(0.0, s_max, effective_ds, splines)
        return cls(spline_points, T, N, k, k_tag, effective_ds)

    @property
    def s_limits(self):
        return np.array([0, self.s_max])

    @property
    def s_max(self):
        return self.ds * (len(self.O) - 1)

    @property
    def points(self):
        return self.O

    def get_curvature(self, s: np.ndarray):
        """
        Computes curvature for a certain point on the curve given by s (progress from the curve's start)
        :param s: progress on the curve from its beginning in meters (any tensor shape)
        :return: curvature in [1/m]
        """
        _, _, _, k_r, _ = self._taylor_interp(s)
        return k_r

    def get_yaw(self, s: np.ndarray):
        """
        Computes yaw (in radians, relative to the origin in which the curve points (self.O) are given
        :param s: progress on the curve from its beginning in meters (any tensor shape)
        :return: yaw in radians (tensor shape is the same as <s>)
        """
        _, T_r, _, _, _ = self._taylor_interp(s)
        return np.arctan2(T_r[..., C_Y], T_r[..., C_X])

    ## FRENET => CARTESIAN

    def fpoint_to_cpoint(self, fpoints: FrenetPoint) -> CartesianPoint2D:
        """Transforms a frenet-frame point to a cartesian-frame point (see self.fpoints_to_cpoints for more details)"""
        return self.fpoints_to_cpoints(fpoints[np.newaxis, :])[0]

    def fpoints_to_cpoints(self, fpoints: FrenetTrajectory2D) -> CartesianPath2D:
        """
        Transforms frenet-frame points to cartesian-frame points (using self.curve)
        :param fpoint: Frenet-frame trajectory (matrix)
        :return: Cartesian-frame trajectory (matrix)
        """
        a_s, _, N_s, _, _ = self._taylor_interp(fpoints[..., FP_SX])
        return a_s + N_s * fpoints[..., [FP_DX]]

    def fstate_to_cstate(self, fstate: FrenetState2D) -> CartesianExtendedState:
        """
        Transforms Frenet-frame state to cartesian-frame state
        :param ftrajectory: a frenet-frame state
        :return: a cartesian-frame state (given in the coordinate frame of self.points)
        """
        return self.ftrajectory_to_ctrajectory(np.array([fstate]))[0]

    def ftrajectory_to_ctrajectory(self, ftrajectory: FrenetTrajectory2D) -> CartesianExtendedTrajectory:
        """
        Transforms Frenet-frame trajectory to cartesian-frame trajectory, using tensor operations
        :param ftrajectory: a frenet-frame trajectory
        :return: a cartesian-frame trajectory (given in the coordinate frame of self.points)
        """
        return self.ftrajectories_to_ctrajectories(np.array([ftrajectory]))[0]

    def ftrajectories_to_ctrajectories(self, ftrajectories: FrenetTrajectories2D) -> CartesianExtendedTrajectories:
        """
        Transforms Frenet-frame trajectories to cartesian-frame trajectories, using tensor operations.
        For formulas derivations please refer to: http://ieeexplore.ieee.org/document/5509799/
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

        # Prevent division by zero velocity (s_v=0):
        #       When vehicle's velocity is zero, we assume that the vehicle is parallel to the road.
        #       Calculate d_tag & d_tagtag as 1st and 2nd derivatives of d_x by distance
        # 1st derivative of d_x by distance: d_tag = d_v / s_v
        d_tag = np.divide(d_v, s_v, where=s_v!=0)
        # 2nd derivative of d_x by distance: d_tagtag = (d_a - d_tag * s_a) / (s_v ** 2)
        d_tagtag = np.divide(d_a - d_tag * s_a, s_v ** 2, where=s_v!=0)

        tan_delta_theta = d_tag / radius_ratio
        delta_theta = np.arctan2(d_tag, radius_ratio)
        cos_delta_theta = np.cos(delta_theta)

        # compute v_x (velocity in the heading direction)
        v_x = np.divide(s_v * radius_ratio, cos_delta_theta)

        # compute k_x (curvature)
        k_x = ((d_tagtag + (k_r_tag * d_x + k_r * d_tag) * np.tan(delta_theta)) * np.cos(
            delta_theta) ** 2 / radius_ratio + k_r) * np.cos(delta_theta) / radius_ratio

        # compute a_x (curvature)
        delta_theta_tag = radius_ratio / np.cos(
            delta_theta) * k_x - k_r  # derivative of delta_theta (via chain rule: d(sx)->d(t)->d(s))
        a_x = s_v ** 2 / cos_delta_theta * (radius_ratio * tan_delta_theta * delta_theta_tag - (
        k_r_tag * d_x + k_r * d_tag)) + s_a * radius_ratio / cos_delta_theta

        # compute position (cartesian)
        pos_x = a_r + N_r * d_x[..., np.newaxis]

        # compute theta_x
        theta_x = theta_r + delta_theta

        return np.concatenate((pos_x.reshape([num_t, num_p, 2]), theta_x.reshape([num_t, num_p, 1]),
                               v_x.reshape([num_t, num_p, 1]), a_x.reshape([num_t, num_p, 1]),
                               k_x.reshape([num_t, num_p, 1])), axis=2)

    ## CARTESIAN => FRENET

    def cpoint_to_fpoint(self, cpoints: CartesianPoint2D) -> FrenetPoint:
        """Transforms a cartesian-frame point to a frenet-frame point (see self.fpoints_to_cpoints for more details)"""
        return self.cpoints_to_fpoints(cpoints[np.newaxis, :])[0]

    def cpoints_to_fpoints(self, cpoints: CartesianPath2D) -> FrenetTrajectory2D:
        """
        Transforms cartesian-frame points to frenet-frame points (using self.curve)
        :param cpoints: Cartesian-frame trajectory (matrix)
        :return: Frenet-frame trajectory (matrix)
        """
        s, a_s, _, N_s, _, _ = self._project_cartesian_points(cpoints)

        # project cpoints on the normals at a_s
        d = np.einsum('ij,ij->i', cpoints - a_s, N_s)

        return np.c_[s, d]

    def cstate_to_fstate(self, cstate: CartesianExtendedState) -> FrenetState2D:
        """
        Transforms Cartesian-frame state to Frenet-frame state
        :param cstate: a cartesian-frame state (in the coordinate frame of self.points)
        :return: a frenet-frame state
        """
        return self.ctrajectory_to_ftrajectory(np.array([cstate]))[0]

    def ctrajectory_to_ftrajectory(self, ctrajectory: CartesianExtendedTrajectory) -> FrenetTrajectory2D:
        """
        Transforms Cartesian-frame trajectory to Frenet-frame trajectory, using tensor operations
        :param ctrajectory: a cartesian-frame trajectory (in the coordinate frame of self.points)
        :return: a frenet-frame trajectory
        """
        return self.ctrajectories_to_ftrajectories(np.array([ctrajectory]))[0]

    def ctrajectories_to_ftrajectories(self, ctrajectories: CartesianExtendedTrajectories) -> FrenetTrajectories2D:
        """
        Transforms Cartesian-frame trajectories to Frenet-frame trajectories, using tensor operations
        For formulas derivations please refer to: http://ieeexplore.ieee.org/document/5509799/
        :param ctrajectories: Cartesian-frame trajectories (tensor)
        :return: Frenet-frame trajectories (tensor)
        """
        pos_x = ctrajectories[:, :, [C_X, C_Y]]
        theta_x = ctrajectories[:, :, C_YAW]
        k_x = ctrajectories[:, :, C_K]
        v_x = ctrajectories[:, :, C_V]
        a_x = ctrajectories[:, :, C_A]

        s_x, a_r, T_r, N_r, k_r, k_r_tag = self._project_cartesian_points(pos_x)

        d_x = np.einsum('tpi,tpi->tp', pos_x - a_r, N_r)

        radius_ratio = 1 - k_r * d_x  # pre-compute terms to use below

        theta_r = np.arctan2(T_r[..., C_Y], T_r[..., C_X])
        delta_theta = theta_x - theta_r

        s_v = v_x * np.cos(delta_theta) / radius_ratio
        d_v = v_x * np.sin(delta_theta)

        # derivative of delta_theta (via chain rule: d(sx)->d(t)->d(s))
        delta_theta_tag = radius_ratio / np.cos(delta_theta) * k_x - k_r

        d_tag = radius_ratio * np.tan(delta_theta)  # invalid: (radius_ratio * np.sin(delta_theta)) ** 2
        d_tag_tag = -(k_r_tag * d_x + k_r * d_tag) * np.tan(delta_theta) + radius_ratio / np.cos(delta_theta) ** 2 * (
        k_x * radius_ratio / np.cos(delta_theta) - k_r)

        s_a = (a_x - s_v ** 2 / np.cos(delta_theta) *
               (radius_ratio * np.tan(delta_theta) * delta_theta_tag - (k_r_tag * d_x + k_r * d_tag))) * np.cos(
            delta_theta) / radius_ratio
        d_a = d_tag_tag * s_v ** 2 + d_tag * s_a

        return np.dstack((s_x, s_v, s_a, d_x, d_v, d_a))

    ## UTILITIES ##

    def _approximate_s_from_points(self, points: np.ndarray) -> np.ndarray:
        """
        Given cartesian points, this method approximates the s longitudinal progress of these points on
        the frenet frame.
        :param points: a tensor (any shape) of 2D points in cartesian frame (same origin as self.O)
        :return: approximate s value on the frame that will be created using self.O
        """
        # perform gradient decent to find s_approx
        O_idx, delta_s = Euclidean.project_on_piecewise_linear_curve(points, self.O)
        s_approx = np.add(O_idx, delta_s) * self.ds
        return s_approx

    def _get_closest_index_on_frame(self, s: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        from s, a vector of longitudinal progress on the frame, return the index of the closest point on the frame and
        a value in the range [0, ds] representing the projection on this closest point.
        The returned values, if summed, represent a "fractional index" on the curve.
        :param s: a vector of longitudinal progress on the frame
        :return: a tuple of: (indices of closest points, (signed) distance on s axis between the given s coordinate and
                             the closest point chosen (can be negative))
        """
        progress_ds = s / self.ds
        O_idx = np.round(progress_ds).astype(np.int)
        delta_s = np.expand_dims((progress_ds - O_idx) * self.ds, axis=len(s.shape))
        return O_idx, delta_s

    def _project_cartesian_points(self, points: np.ndarray) -> \
            (np.ndarray, CartesianPointsTensor2D, CartesianVectorsTensor2D, CartesianVectorsTensor2D, np.ndarray, np.ndarray):
        """Given a tensor (any shape) of 2D points in cartesian frame (same origin as self.O),
        this function uses taylor approximation to return
        s*, a(s*), T(s*), N(s*), k(s*), k'(s*), where:
        s* is the progress along the curve where the point is projected
        a(s*) is the Cartesian-coordinates (x,y) of the projections on the curve,
        T(s*) is the tangent unit vector (dx,dy) of the projections on the curve
        N(s*) is the normal unit vector (dx,dy) of the projections on the curve
        k(s*) is the curvatures (scalars) - assumed to be constant in the neighborhood of the points in self.O and thus
        taken from the nearest point in self.O
        k'(s*) is the derivatives of the curvatures (by distance d(s))
        """
        # perform gradient decent to find s_approx

        s_approx = self._approximate_s_from_points(points)

        a_s, T_s, N_s, k_s, _ = self._taylor_interp(s_approx)

        is_curvature_big_enough = np.greater(np.abs(k_s), TINY_CURVATURE)

        # don't enable zero curvature to prevent numerical problems with infinite radius
        k_s[np.logical_not(is_curvature_big_enough)] = TINY_CURVATURE

        # signed circle radius according to the curvature
        signed_radius = np.divide(1, k_s)

        # vector from the circle center to the input point
        center_to_point = points - a_s - N_s * signed_radius[..., np.newaxis]

        # sign of the step (sign of the inner product between the position error and the tangent of all samples)
        step_sign = np.sign(np.einsum('...ik,...ik->...i', points - a_s, T_s))

        # cos(angle between N_s and this vector)
        cos = np.abs(np.einsum('...ik,...ik->...i', N_s, center_to_point) / np.linalg.norm(center_to_point, axis=-1))

        # prevent illegal (greater than 1) argument for arccos()
        # don't enable zero curvature to prevent numerical problems with infinite radius
        cos[np.logical_or(np.logical_not(is_curvature_big_enough), cos > 1.0)] = 1.0

        # arc length from a_s to the new guess point
        step = step_sign * np.arccos(cos) * np.abs(signed_radius)
        s_approx[is_curvature_big_enough] += step[is_curvature_big_enough]  # next s_approx of the current point

        a_s, T_s, N_s, k_s, k_s_tag = self._taylor_interp(s_approx)

        return s_approx, a_s, T_s, N_s, k_s, k_s_tag

    def _taylor_interp(self, s: np.ndarray) -> \
            (CartesianPointsTensor2D, CartesianVectorsTensor2D, CartesianVectorsTensor2D, np.ndarray, np.ndarray):
        """Given arbitrary s tensor (of shape D) of progresses along the curve (in the range [0, self.s_max]),
        this function uses taylor approximation to return curve parameters at each progress. For derivations of
        formulas, see: http://www.cnbc.cmu.edu/~samondjm/papers/Zucker2005.pdf (page 4). Curve parameters are:
        a(s) is the map to Cartesian-frame (a point on the curve. will have shape of Dx2),
        T(s) is the tangent unit vector (will have shape of Dx2)
        N(s) is the normal unit vector (will have shape of Dx2)
        k(s) is the curvature (scalar) - assumed to be constant in the neighborhood of the points in self.O and thus
        taken from the nearest point in self.O (will have shape of D)
        k'(s) is the derivative of the curvature (by distance d(s))
        """
        assert np.all(np.bitwise_and(0 <= s, s <= self.s_max)), \
            "Cannot extrapolate, desired progress (%s) is out of the curve (s_max = %s)." % (s, self.s_max)

        O_idx, delta_s = self._get_closest_index_on_frame(s)

        a_s = self.O[O_idx] + \
              delta_s * self.T[O_idx] + \
              delta_s ** 2 / 2 * self.k[O_idx] * self.N[O_idx] - \
              delta_s ** 3 / 6 * self.k[O_idx] ** 2 * self.T[O_idx]

        T_s = self.T[O_idx] + \
              delta_s * self.k[O_idx] * self.N[O_idx] - \
              delta_s ** 2 / 2 * self.k[O_idx] ** 2 * self.T[O_idx]
        T_s /= np.linalg.norm(T_s, axis=-1, keepdims=True)

        N_s = self.N[O_idx] - \
              delta_s * self.k[O_idx] * self.T[O_idx] - \
              delta_s ** 2 / 2 * self.k[O_idx] ** 2 * self.N[O_idx]
        N_s /= np.linalg.norm(N_s, axis=-1, keepdims=True)

        k_s = self.k[O_idx] + \
              delta_s * self.k_tag[O_idx]

        k_s_tag = self.k_tag[O_idx]

        return a_s, T_s, N_s, k_s[..., 0], k_s_tag[..., 0]

    @staticmethod
    def _fit_frenet_from_splines(start: float, stop: float, step: float,
                                 xy_splines: Tuple[UnivariateSpline, UnivariateSpline]) -> \
            (CartesianVectorsTensor2D, CartesianVectorsTensor2D, np.ndarray, np.ndarray):
        """
        Utility for the construction of the Frenet-Serret frame. Given a set of 2D points in cartesian-frame, it fits
        a curve and returns its parameters at the given points (Tangent, Normal, curvature, etc.).
        Formulas are similar to: dipy.tracking.metrics.frenet_serret() but modified for 2D (rather than 3D), for
        signed-curvature and for continuity of the Normal vector regardless of the curvature-sign.
        :param start: [m] start of progress on the curve. The natural value to use here is 0.0
        :param stop: [m] the end of the curve to use
        :param step: [m] the constant step-size in meters
        :param xy_splines: a tuple of the splines objects used for fitting x, y
        :return:
        """

        x = xy_splines[0]
        x_dot = x.derivative(1)
        x_dotdot = x.derivative(2)
        y = xy_splines[1]
        y_dot = y.derivative(1)
        y_dotdot = y.derivative(2)

        # parametrization of progress on the curve (in meters)
        s = np.arange(start, stop, step)

        dxy = np.c_[x_dot(s), y_dot(s)]
        ddxy = np.c_[x_dotdot(s), y_dotdot(s)]

        dxy_norm = np.linalg.norm(dxy, axis=1)

        # Tangent
        T = np.divide(dxy, np.c_[dxy_norm])

        # Normal - robust to zero-curvature
        N = NumpyUtils.row_wise_normal(T)

        # SIGNED (!) Curvature
        cross_norm = np.sum(NumpyUtils.row_wise_normal(dxy) * ddxy, axis=1)
        k = cross_norm / dxy_norm ** 3

        # derivative of curvature (by ds)
        k_tag = np.divide(np.gradient(k), step)

        return T, N, np.c_[k], np.c_[k_tag]

    def serialize(self):
        # type: () -> LcmFrenetSerret2DFrame
        lcm_msg = LcmFrenetSerret2DFrame()

        lcm_msg.points = SerializationUtils.serialize_non_typed_array(self.O)
        lcm_msg.T = SerializationUtils.serialize_non_typed_array(self.T)
        lcm_msg.N = SerializationUtils.serialize_non_typed_array(self.N)
        lcm_msg.k = SerializationUtils.serialize_non_typed_array(self.k)
        lcm_msg.k_tag = SerializationUtils.serialize_non_typed_array(self.k_tag)

        lcm_msg.ds = self.ds

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmFrenetSerret2DFrame)->FrenetSerret2DFrame
        return cls(SerializationUtils.deserialize_any_array(lcmMsg.points),
                   SerializationUtils.deserialize_any_array(lcmMsg.T),
                   SerializationUtils.deserialize_any_array(lcmMsg.N),
                   SerializationUtils.deserialize_any_array(lcmMsg.k),
                   SerializationUtils.deserialize_any_array(lcmMsg.k_tag),
                   lcmMsg.ds)
