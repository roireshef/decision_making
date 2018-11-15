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
        pass

    # TODO: not validated to work
    def convert_from_segment_states(self, frenet_states: FrenetStates2D, segment_ids: List[int]) -> FrenetStates2D:
        # return np.array([self.convert_from_segment_state(frenet_state, segment_id)
        #                  for frenet_state, segment_id in zip(frenet_states, segment_ids)])
        segment_idxs = self._get_segment_idxs_from_ids(segment_ids)
        s_offset = self.segments_offsets[segment_idxs]
        new_frenet_states = frenet_states.copy()
        new_frenet_states[..., FS_SX] = s_offset
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

    # TODO: replace with vectorized operations
    def convert_to_segment_states(self, frenet_states: FrenetStates2D) -> (List[int], FrenetStates2D):
        segment_ids, states = zip(*[self.convert_to_segment_state(state) for state in frenet_states])
        return list(segment_ids), np.array(states)

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

