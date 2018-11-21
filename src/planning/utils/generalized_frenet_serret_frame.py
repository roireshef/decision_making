from typing import List

import numpy as np
import numpy_indexed as npi

from decision_making.src.global_constants import TINY_CURVATURE
from decision_making.src.planning.types import CartesianPath2D, FrenetState2D, FrenetStates2D, NumpyIndicesArray, FS_SX, \
    CartesianPointsTensor2D, CartesianVectorsTensor2D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from mapping.src.transformations.geometry_utils import Euclidean


class FrenetSubSegment:
    def __init__(self, segment_id: int, s_start: float, s_end: float, ds: float):
        """
        An object containing information on a partial lane segment, used for concatenating or splitting of frenet frames
        :param segment_id:usually lane_id, indicating which lanes are taken to build the generalized frenet frames.
        :param s_start: starting longitudinal s to be taken into account when segmenting frenet frames.
        :param s_end: ending longitudinal s to be taken into account when segmenting frenet frames.
        :param ds: sampling interval on curve
        """
        self.segment_id = segment_id
        self.s_start = s_start
        self.s_end = s_end
        self.ds = ds
        self.num_points_so_far = None


class GeneralizedFrenetSerretFrame(FrenetSerret2DFrame):
    def __init__(self, points: CartesianPath2D, T: np.ndarray, N: np.ndarray, k: np.ndarray, k_tag: np.ndarray,
                 ds: float, sub_segments: List[FrenetSubSegment]):
        super().__init__(points, T, N, k, k_tag, ds)
        self.sub_segments = sub_segments

    @property
    def segments_s_offsets(self):
        return np.insert(np.cumsum([seg.s_end - seg.s_start for seg in self.sub_segments]), 0, 0., axis=0)

    @property
    def segments_points_offset(self):
        # TODO: Explain this +1/+0
        return np.insert(np.array([seg.num_points_so_far + 0 for seg in self.sub_segments]), 0, 0., axis=0)

    @classmethod
    def build(cls, frenet_frames: List[FrenetSerret2DFrame], sub_segments: List[FrenetSubSegment]):
        """
        Create a generalized frenet frame, which is a concatenation of some frenet frames or a part of them.
        A special case might be a sub segment of a single frenet frame.
        :param frenet_frames: a list of all frenet frames involved in creating the new generalized frame.
        :param sub_segments: a list of FrenetSubSegment objects, used for segmenting the respective elements from
        the frenet_frames parameter.
        :return: A new GeneralizedFrenetSerretFrame built out of different other frenet frames.
        """

        exact_sub_segments = sub_segments

        points = []
        T = []
        N = []
        k = []
        k_tag = []

        for i in range(len(frenet_frames)):
            frame = frenet_frames[i]
            if exact_sub_segments[i].s_start == 0:
                start_ind = 0
            else:
                start_ind = int(np.floor(exact_sub_segments[i].s_start / exact_sub_segments[i].ds))
            # if this is not the last frame then the next already has this frame's last point as its first - omit it.
            if exact_sub_segments[i].s_end == frame.s_max and i < len(frenet_frames) - 1:
                end_ind = frame.points.shape[0] - 1
            else:
                end_ind = int(np.ceil(exact_sub_segments[i].s_end / exact_sub_segments[i].ds))

            points.append(frame.points[start_ind:end_ind, :])
            T.append(frame.T[start_ind:end_ind, :])
            N.append(frame.N[start_ind:end_ind, :])
            k.append(frame.k[start_ind:end_ind, :])
            k_tag.append(frame.k_tag[start_ind:end_ind, :])

            exact_sub_segments[i].num_points_so_far = np.concatenate(points).shape[0]

        return cls(np.concatenate(points), np.concatenate(T), np.concatenate(N), np.concatenate(k),
                   np.concatenate(k_tag), None, exact_sub_segments)

    # TODO: not validated to work
    def convert_from_segment_states(self, frenet_states: FrenetStates2D, segment_ids: List[int]) -> FrenetStates2D:
        # return np.array([self.convert_from_segment_state(frenet_state, segment_id)
        #                  for frenet_state, segment_id in zip(frenet_states, segment_ids)])
        segment_idxs = self._get_segment_idxs_from_ids(segment_ids)
        s_offset = self.segments_s_offsets[segment_idxs]
        new_frenet_states = frenet_states.copy()
        new_frenet_states[..., FS_SX] += s_offset
        return new_frenet_states

    # TODO: not validated to work
    def convert_from_segment_state(self, frenet_state: FrenetState2D, segment_id: int) -> FrenetState2D:
        """
        Converts a frenet_state on a frenet_frame to a frenet_state on the generalized frenet frame.
        :param frenet_state: a frenet_state on another frenet_frame which was part in building the generalized frenet frame.
        :param segment_id: a segment_id, usually lane_id, of one of the frenet frames which were used in building the generalized frenet frame.
        :return: a frenet state on the generalized frenet frame.
        """
        return self.convert_from_segment_states(frenet_state[np.newaxis, ...], [segment_id])[0]

    # TODO: not validated to work
    def convert_to_segment_states(self, frenet_states: FrenetStates2D) -> (List[int], FrenetStates2D):
        # segment_ids, states = zip(*[self.convert_to_segment_state(state) for state in frenet_states])
        # return list(segment_ids), np.array(states)
        # Find the closest greater segment offset for each frenet state longitudinal
        s_offset = self.segments_s_offsets[self._get_segment_idxs_from_s(frenet_states[:, FS_SX])]
        new_frenet_states = frenet_states.copy()
        new_frenet_states[..., FS_SX] -= s_offset
        return new_frenet_states

    def convert_to_segment_state(self, frenet_state: FrenetState2D) -> (int, FrenetState2D):
        """
        Converts a frenet_state on the generalized frenet frame to a frenet_state on a frenet_frame it's built from.
        :param frenet_state: a frenet_state on the generalized frenet frame.
        :return: a tuple: (the segment_id, usually lane_id, this frenet_state will land on after the conversion, the resulted frenet state)
        """
        pass

    # TODO: not validated to work
    def _get_segment_idxs_from_ids(self, segment_ids: NumpyIndicesArray):
        all_ids = np.array([seg.segment_id for seg in self.sub_segments], dtype=np.int)
        return npi.indices(all_ids, segment_ids)

    def _get_segment_idxs_from_s(self, s_values: np.ndarray):
        segments_idxs = np.searchsorted(self.segments_s_offsets, s_values) - 1
        segments_idxs[s_values == 0] = 0
        return segments_idxs

    @property
    def s_max(self):
        return self.segments_s_offsets[-1]

    def _project_cartesian_points(self, points: np.ndarray) -> \
            (np.ndarray, CartesianPointsTensor2D, CartesianVectorsTensor2D, CartesianVectorsTensor2D, np.ndarray,
             np.ndarray):
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
        O_idx, delta_s = Euclidean.project_on_piecewise_linear_curve(points, self.O)

        # s_approx = np.add(O_idx, delta_s) * self.sub_segments[0].ds

        subsegments_ds = np.array([segment.ds for segment in self.sub_segments])

        segment_idx_per_point = np.searchsorted(self.segments_points_offset, np.add(O_idx, delta_s)) - 1
        ds = subsegments_ds[segment_idx_per_point]
        s_approx = self.segments_s_offsets[segment_idx_per_point] + \
                   (((O_idx - self.segments_points_offset[segment_idx_per_point]) + delta_s) * ds)

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
            "Cannot extrapolate, desired progress (%s) is out of the curve." % s

        segment_idxs = self._get_segment_idxs_from_s(s)
        subsegments_ds = np.array([segment.ds for segment in self.sub_segments])
        s_in_segment = s - self.segments_s_offsets[segment_idxs]
        points_offset = self.segments_points_offset[segment_idxs]
        # progress_ds = s / self.sub_segments[0].ds
        progress_ds = np.divide(s_in_segment, subsegments_ds[segment_idxs]) + points_offset

        O_idx = np.round(progress_ds).astype(np.int)
        # delta_s = np.expand_dims((progress_ds - O_idx) * self.sub_segments[0].ds, axis=len(s.shape))
        delta_s = np.expand_dims((progress_ds - O_idx) * subsegments_ds[segment_idxs], axis=len(s.shape))

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
              delta_s * self.k_tag[O_idx]
        # delta_s ** 2 / 2 * np.gradient(np.gradient(self.k, axis=0), axis=0)[O_idx]

        k_s_tag = self.k_tag[O_idx]  # + delta_s * np.gradient(np.gradient(self.k, axis=0), axis=0)[O_idx]

        return a_s, T_s, N_s, k_s[..., 0], k_s_tag[..., 0]
