from typing import List

import numpy as np
import numpy_indexed as npi

from decision_making.src.planning.types import CartesianPath2D, FrenetState2D, FrenetStates2D, NumpyIndicesArray, FS_SX
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
        return np.insert(np.array([seg.num_points_so_far for seg in self.sub_segments]), 0, 0., axis=0)

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

        points = np.empty(shape=[0, 2])
        T = np.empty(shape=[0, 2])
        N = np.empty(shape=[0, 2])
        k = np.empty(shape=[0, 1])
        k_tag = np.empty(shape=[0, 1])

        for i in range(len(frenet_frames)):
            frame = frenet_frames[i]
            start_ind = int(np.floor(exact_sub_segments[i].s_start / exact_sub_segments[i].ds))
            # if this is not the last frame then the next already has this frame's last point as its first - omit it.
            if i < len(frenet_frames) - 1:
                # if this frame is not the last frame, it must end in s_max
                assert exact_sub_segments[i].s_end == frame.s_max
                end_ind = frame.points.shape[0] - 1
            else:
                end_ind = int(np.ceil(exact_sub_segments[i].s_end / exact_sub_segments[i].ds))

            points = np.vstack((points, frame.points[start_ind:end_ind, :]))
            T = np.vstack((T, frame.T[start_ind:end_ind, :]))
            N = np.vstack((N, frame.N[start_ind:end_ind, :]))
            k = np.vstack((k, frame.k[start_ind:end_ind, :]))
            k_tag = np.vstack((k_tag, frame.k_tag[start_ind:end_ind, :]))

            exact_sub_segments[i].num_points_so_far = points.shape[0]

        return cls(points, T, N, k, k_tag, None, exact_sub_segments)

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

    def _approximate_s_from_points_idxs(self, points: np.ndarray):
        """
        given cartesian points, this method approximates the s longitudinal values of these points.
        :param points: a tensor (any shape) of 2D points in cartesian frame (same origin as self.O)
        :return: approximate s value on the frame that will be created using self.O
        """
        O_idx, delta_s = Euclidean.project_on_piecewise_linear_curve(points, self.O)
        segment_idx_per_point = np.searchsorted(self.segments_points_offset, np.add(O_idx, delta_s)) - 1
        subsegments_ds = np.array([segment.ds for segment in self.sub_segments])
        ds = subsegments_ds[segment_idx_per_point]
        s_approx = self.segments_s_offsets[segment_idx_per_point] + \
                   (((O_idx - self.segments_points_offset[segment_idx_per_point]) + delta_s) * ds)

        return s_approx

    def _get_closest_index_on_frame(self, s: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        from s, a vector of longitudinal progress on the frame, return the closest index on the frame and
        the residue fractional value.
        :param s: a vector of longitudinal progress on the frame
        :return: a tuple of: a vector of closest indices, a vector of fractional residues
        """
        segment_idxs = self._get_segment_idxs_from_s(s)
        subsegments_ds = np.array([segment.ds for segment in self.sub_segments])
        s_in_segment = s - self.segments_s_offsets[segment_idxs]
        points_offset = self.segments_points_offset[segment_idxs]
        progress_ds = np.divide(s_in_segment, subsegments_ds[segment_idxs]) + points_offset
        O_idx = np.round(progress_ds).astype(np.int)
        delta_s = np.expand_dims((progress_ds - O_idx) * subsegments_ds[segment_idxs], axis=len(s.shape))
        return O_idx, delta_s
