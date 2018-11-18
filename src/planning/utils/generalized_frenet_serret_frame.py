from typing import List

import numpy as np
import numpy_indexed as npi

from decision_making.src.planning.types import CartesianPath2D, FrenetState2D, FrenetStates2D, NumpyIndicesArray, FS_SX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame


class FrenetSubSegment:
    def __init__(self, segment_id: int, s_start: float, s_end: float):
        """
        An object containing information on a partial lane segment, used for concatenating or splitting of frenet frames
        :param segment_id:usually lane_id, indicating which lanes are taken to build the generalized frenet frames.
        :param s_start: starting longitudinal s to be taken into account when segmenting frenet frames.
        :param s_end: ending longitudinal s to be taken into account when segmenting frenet frames.
        """
        self.segment_id = segment_id
        self.s_start = s_start
        self.s_end = s_end


class GeneralizedFrenetSerretFrame(FrenetSerret2DFrame):
    def __init__(self, points: CartesianPath2D, T: np.ndarray, N: np.ndarray, k: np.ndarray, k_tag: np.ndarray,
                 ds: float, sub_segments: List[FrenetSubSegment]):
        super().__init__(points, T, N, k, k_tag, ds)
        self.sub_segments = sub_segments

    @property
    def segments_offsets(self):
        return np.insert(np.cumsum([seg.s_end - seg.s_start for seg in self.sub_segments]), 0, 0., axis=0)

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
        # TODO: Handle concatenation of frenet frames with different ds?
        # assert len(set([frame.ds for frame in frenet_frames])) == 1

        min_ds = min([frame.ds for frame in frenet_frames])
        for i in range(len(frenet_frames)):
            if frenet_frames[i].ds != min_ds:
                frenet_frames[i] = FrenetSerret2DFrame.fit(frenet_frames[i].points, min_ds)

        # TODO: Check that all sub_segments limits are multiples of ds and raise a warning otherwise
        # np.all(np.isclose(np.fmod(np.array( [frame.ds for frame in frenet_frames]), 0.5), 0,atol=1e-4))

        points = []
        T = []
        N = []
        k = []
        k_tag = []

        for i in range(len(frenet_frames)):
            frame = frenet_frames[i]
            if sub_segments[i].s_start == 0:
                start_ind = 0
            else:
                start_ind = GeneralizedFrenetSerretFrame.get_point_index_closest_to_lon(frame, sub_segments[i].s_start)
            # if this is not the last frame then the next already has this frame's last point as its first - omit it.
            if sub_segments[i].s_end == frame.s_max and i < len(frenet_frames)-1:
                end_ind = frame.points.shape[0] - 2
            else:
                end_ind = GeneralizedFrenetSerretFrame.get_point_index_closest_to_lon(frame, sub_segments[i].s_end)

            points.append(frame.points[start_ind:end_ind, :])
            T.append(frame.T[start_ind:end_ind, :])
            N.append(frame.N[start_ind:end_ind, :])
            k.append(frame.k[start_ind:end_ind, :])
            k_tag.append(frame.k_tag[start_ind:end_ind, :])

        return cls(np.concatenate(points), np.concatenate(T), np.concatenate(N), np.concatenate(k),
                   np.concatenate(k_tag), min_ds, sub_segments)

    @staticmethod
    def get_point_index_closest_to_lon(frame: FrenetSerret2DFrame, longitude: float) -> int:
        """
        given a longitude, get the point index in frame.points which is the closest to this longitude.
        :param frame: frenet frame
        :param longitude: longitude in frenet
        :return:the index of the closest point in frame.points
        """
        cartesian_point = frame.fpoint_to_cpoint(np.array([longitude, 0]))
        closest_ind = np.argmin(np.linalg.norm(frame.points - cartesian_point, axis=1))
        return closest_ind

    # TODO: not validated to work
    def convert_from_segment_states(self, frenet_states: FrenetStates2D, segment_ids: List[int]) -> FrenetStates2D:
        # return np.array([self.convert_from_segment_state(frenet_state, segment_id)
        #                  for frenet_state, segment_id in zip(frenet_states, segment_ids)])
        segment_idxs = self._get_segment_idxs_from_ids(segment_ids)
        s_offset = self.segments_offsets[segment_idxs]
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
        s_offset = self.segments_offsets[np.searchsorted(self.segments_offsets, frenet_states[:, FS_SX])-1]
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

