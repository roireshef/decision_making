from typing import List

import numpy as np

from decision_making.src.planning.types import CartesianPath2D, FrenetState2D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame


class FrenetSubSegment:
    def __init__(self, segment_id: int, lon_start: float, lon_end: float):
        """
        An object containing information on a partial lane segment, used for concatenating or splitting of frenet frames
        :param segment_id:usually lane_id, indicating which lanes are taken to build the generalized frenet frames.
        :param lon_start: starting longitudinal s to be taken into account when segmenting frenet frames.
        :param lon_end: ending longitudinal s to be taken into account when segmenting frenet frames.
        """
        self.segment_id = segment_id
        self.lon_start = lon_start
        self.lon_end = lon_end


class GeneralizedFrenetSerretFrame(FrenetSerret2DFrame):

    def __init__(self, points: CartesianPath2D, T: np.ndarray, N: np.ndarray, k: np.ndarray, k_tag: np.ndarray,
                 ds: float, sub_segments: List[FrenetSubSegment]):
        super().__init__(points, T, N, k, k_tag, ds)
        self.sub_segments = sub_segments

    @classmethod
    def create_generalized_frenet_frame(cls, frenet_frames: List[FrenetSerret2DFrame], sub_segments: List[FrenetSubSegment]):
        """
        Create a generalized frenet frame, which is a concatenation of some frenet frames or a part of them.
        A special case might be a sub segment of a single frenet frame.
        :param frenet_frames: a list of all frenet frames involved in creating the new generalized frame.
        :param sub_segments: a list of FrenetSubSegment objects, used for segmenting the respective elements from
        the frenet_frames parameter.
        :return: A new GeneralizedFrenetSerretFrame built out of different other frenet frames.
        """
        pass

    def convert_from_fstate(self, frenet_state: FrenetState2D, segment_id: int) -> FrenetState2D:
        """
        Converts a frenet_state on a frenet_frame to a frenet_state on the generalized frenet frame.
        :param frenet_state: a frenet_state on another frenet_frame which was part in building the generalized frenet frame.
        :param segment_id: a segment_id, usually lane_id, of one of the frenet frames which were used in building the generalized frenet frame.
        :return: a frenet state on the generalized frenet frame.
        """
        pass

    def convert_to_fstate(self, frenet_state: FrenetState2D) -> (int, FrenetState2D):
        """
        Converts a frenet_state on the generalized frenet frame to a frenet_state on a frenet_frame it's built from.
        :param frenet_state: a frenet_state on the generalized frenet frame.
        :return: a tuple: (the segment_id, usually lane_id, this frenet_state will land on after the conversion, the resulted frenet state)
        """
        pass
