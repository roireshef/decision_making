from typing import List, Optional, Dict, Tuple, Set

import numpy as np
from decision_making.src.planning.types import FrenetState2D, FS_SX
from decision_making.src.rl_agent.utils.class_utils import Representable


class LaneSegmentKey(Representable):
    def __init__(self, edge: str, lane_id: int):
        """
        Data-structure to hold the identifier key for a LaneSegment object
        :param edge: SUMO edge id / UC road id
        :param lane_id: SUMO lane id. 0 is rightmost, increasing towards left
        """
        self.edge = edge
        self.lane_id = lane_id
        self._hash = hash(str(self))

    def __eq__(self, other) -> bool:
        assert isinstance(other, LaneSegmentKey), 'trying to compare %s to LaneSegmentKey' % type(other)
        return self.__hash__() == other.__hash__()

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return self._hash

    def __lt__(self, other):
        return self.__hash__() < other.__hash__()


class LaneSegmentAttributes(Representable):
    def __init__(self, length: float, speed_limit: float, is_intersection: bool):
        """
        A data-structure that holds attributes of a LaneSegment object
        :param length: lane segment's length in meters (longitudinal)
        :param speed_limit: speed limit for driving on this lane segment in m/s
        :param is_intersection: does this lane segment geographically intersect with another lane segment
        """
        self.speed_limit = speed_limit
        self.length = length
        self.is_intersection = is_intersection


class LaneSegment(Representable):
    def __init__(self, key: LaneSegmentKey, attributes: LaneSegmentAttributes):
        """ key-value pair of LaneSegmentKey and LaneSegmentAttributes to represent full LaneSegment object.
         Note that this object is hashed and compared according to key only """
        self.attributes = attributes
        self.key = key

    def __eq__(self, other) -> bool:
        assert isinstance(other, LaneSegment), 'trying to compare %s to LaneSegment' % type(other)
        return self.__hash__() == other.__hash__()

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.__dict__)

    def __hash__(self):
        return hash(self.key)

    def __lt__(self, other):
        return self.__hash__() < other.__hash__()


class GeneralizedFrenetFrame(Representable):
    def __init__(self, lane_segments: List[LaneSegment]):
        """
        This class is mimicking the GeneralizedFrenetSerretFrame in decision_making repository, but is currently lacking
        the frame points layer, so it doesn't actually store the curve the moving frame is moving on. This can be
        abstracted away since we currently only care about longitudinal movements, and lane changes (that are discrete
        transitions between GFFs)
        :param lane_segments: a list of LaneSegment elements, ordered by longitudinal order on the road, assumed to
        intersect in transitions between any two consecutive elements (no longitudinal gaps or overlaps allowed)
        """
        assert len(set(lane_segments)) == len(lane_segments), "Duplicates found in %s" % lane_segments

        # original list of LaneSegment objects
        self.lane_segments = lane_segments

        # dictionary that maps LaneSegmentKeys to LaneSegmentAttributes for fast access by LaneSegmentKey
        self.lane_segments_dict = {ls.key: ls.attributes for ls in lane_segments}

        # cache of lane segment's longitudinal offset of its start point (from the beginning of the GFF) and attributes
        self.segments_speed_limits = np.array([lane_segment.attributes.speed_limit for lane_segment in lane_segments])
        self.segments_lengths = np.array([lane_segment.attributes.length for lane_segment in lane_segments])
        self.segments_s_offsets = np.insert(np.cumsum(self.segments_lengths), 0, 0., axis=0)

        # cache a boolean vector indicates if lane segment is an intersection
        self.segments_is_intersection = np.array([ls.attributes.is_intersection for ls in self.lane_segments],
                                                 dtype=np.bool)

        # cache indices of lane-segments for fast query
        self.lane_segment_index_by_key = {seg.key: idx for idx, seg in enumerate(self.lane_segments)}
        self.lane_segment_index_by_edge_id = {seg.key.edge: idx for idx, seg in enumerate(self.lane_segments)}
        self.lane_segment_index_by_lane_id = {seg.key.lane_id: idx for idx, seg in enumerate(self.lane_segments)}

    @property
    def lane_segment_keys(self) -> Set[LaneSegmentKey]:
        return set(ls.key for ls in self.lane_segments)

    @property
    def edge_ids(self) -> Set[str]:
        return set(self.lane_segment_index_by_edge_id.keys())

    @property
    def lane_ids(self) -> Set[int]:
        return set(self.lane_segment_index_by_lane_id.keys())

    @property
    def s_max(self):
        """
        :return: the largest longitudinal value of the generalized frenet frame (end of curve).
        """
        return self.segments_s_offsets[-1]

    @property
    def seg_to_edge_ahead_offsets(self) -> Dict[Tuple[LaneSegmentKey, str], float]:
        """ returns a dictionary that maps ((edge, lane_id), edge_ahead) tuples to the longitudinal offset difference
        between them, iterating all couples of edges in this gff """
        return {(seg.key, seg_ahead.key.edge): self.segments_s_offsets[j+i+1] - self.segments_s_offsets[i]
                for i, seg in enumerate(self.lane_segments)
                for j, seg_ahead in enumerate(self.lane_segments[(i + 1):])}

    def convert_from_segment_state(self, fstate: FrenetState2D, lane_segment_key: LaneSegmentKey) -> FrenetState2D:
        """
        Converts a frenet_state that is relative to a lane_segment to a frenet_state on this GFF.
        :param fstate: coordinates to convert (station should be relative to lane_segment start)
        :param lane_segment_key: the lane segment <fstate> is projected on
        :return: 1d numpy array that represents <fstate> projected onto the GFF
        """
        new_fstate = fstate.copy()
        new_fstate[..., FS_SX] += self.get_offset_for_lane_segment(lane_segment_key)

        return new_fstate

    def convert_to_segment_state(self, fstate: FrenetState2D) -> Tuple[LaneSegmentKey, FrenetState2D]:
        """
        Converts a frenet_state that is relative to this GFF to a frenet_state on the relevant LaneSegment (from
        self.lane_segments).
        :param fstate: 1d numpy array that represents <fstate> relative to this GFF
        :return: tuple (relevant LaneSegment object, frenet coordinates relative to it)
        """
        assert fstate[..., FS_SX] <= self.s_max, "%s doesn't have s coordinate %f (s_max is %f)" % \
                                                 (self, fstate[..., FS_SX], self.s_max)
        lane_segment_idx = self._get_segment_idxs_from_s(np.array([fstate[..., FS_SX]]))[0]
        s_offset = self.segments_s_offsets[lane_segment_idx]
        new_fstate = fstate.copy()
        new_fstate[..., FS_SX] -= s_offset

        return self.lane_segments[lane_segment_idx].key, new_fstate

    def get_offset_for_lane_segment(self, lane_segment_key: LaneSegmentKey) -> float:
        """
        Given a lane segment object, it queries for it from self.lane_segments and retrieves its s_offset
        (see constructor for more details)
        :param lane_segment_key: LaneSegment object to query for
        :return: the <lane_segment>'s longitudinal offset from beginning of GFF
        """
        assert lane_segment_key in self.lane_segment_keys, "%s doesn't have %s" % (self, lane_segment_key)
        lane_segment_idx = self._get_segment_idx(lane_segment_key)
        return self.segments_s_offsets[lane_segment_idx]

    def get_offset_for_edge(self, edge_id: str) -> float:
        """
        Given an edge id, it finds the relevant LaneSegment and retrieves its s_offset
        :param edge_id: edge id
        :return: the <lane_segment>'s longitudinal offset from beginning of GFF
        """
        assert edge_id in self.edge_ids, "%s doesn't have edge %s" % (self, edge_id)
        lane_segment_key = self.get_lane_segment_key_by_edge(edge_id)
        return self.get_offset_for_lane_segment(lane_segment_key)

    def get_length_for_lane_segment(self, lane_segment_key: LaneSegmentKey) -> float:
        """
        Given a lane segment object, it queries for it from self.lane_segments and retrieves its s_offset
        (see constructor for more details)
        :param lane_segment_key: LaneSegment object to query for
        :return: the <lane_segment>'s longitudinal offset from beginning of GFF
        """
        assert lane_segment_key in self.lane_segment_keys, "%s doesn't have %s" % (self, lane_segment_key)
        lane_segment_idx = self._get_segment_idx(lane_segment_key)
        return self.segments_lengths[lane_segment_idx]

    def get_length_for_edge(self, edge_id: str) -> float:
        """
        Given an edge id, it finds the relevant LaneSegment and retrieves its length
        :param edge_id: edge id
        :return: the legnth of the edge
        """
        assert edge_id in self.edge_ids, "%s doesn't have edge %s" % (self, edge_id)
        lane_segment_key = self.get_lane_segment_key_by_edge(edge_id)
        return self.get_length_for_lane_segment(lane_segment_key)

    def get_sum_lengths_for_edges(self, edge_ids: List[str]) -> float:
        return sum([self.get_length_for_edge(edge_id) for edge_id in edge_ids])

    def get_speed_limit_for_lane_segment(self, lane_segment_key: LaneSegmentKey) -> float:
        assert lane_segment_key in self.lane_segment_keys, "%s doesn't have %s" % (self, lane_segment_key)
        lane_segment_idx = self._get_segment_idx(lane_segment_key)
        return self.segments_speed_limits[lane_segment_idx]

    def get_speed_limit_for_edge(self, edge_id: str) -> float:
        assert edge_id in self.edge_ids, "%s doesn't have edge %s" % (self, edge_id)
        lane_segment_key = self.get_lane_segment_key_by_edge(edge_id)
        return self.get_speed_limit_for_lane_segment(lane_segment_key)

    def get_lane_segment_key_by_edge(self, edge_id: str) -> LaneSegmentKey:
        """ Given an edge ID (string), get its corresponding LaneSegment object from self.lane_segments """
        return self.lane_segments[self.lane_segment_index_by_edge_id[edge_id]].key

    def next_intersection_offset(self, sx: float) -> float:
        """
        returns the offset (station) of the next intersection segment (relative to sx)
        :param sx: start looking from that station on this GFF
        :return: station of the beginning of the next intersection lane-segment
        """
        next_intersection_idxs = self._next_intersections_segment_idxs(sx)
        return self.segments_s_offsets[next_intersection_idxs[0].item()] if len(next_intersection_idxs) > 0 else np.nan

    def next_intersection_segment(self, sx: float) -> Optional[LaneSegmentKey]:
        next_intersection_idxs = self._next_intersections_segment_idxs(sx)
        return self.lane_segments[next_intersection_idxs[0].item()].key if len(next_intersection_idxs) > 0 else None

    def next_intersection_segments_on_gff(self, sx: float) -> List[LaneSegmentKey]:
        next_intersection_idxs = self._next_intersections_segment_idxs(sx)
        return [lane_segment.key for idx, lane_segment in enumerate(self.lane_segments)
                if idx in next_intersection_idxs]

    # Protected methods

    def _next_intersections_segment_idxs(self, sx: float) -> np.ndarray:
        """
        returns an array with indices of the lane segments ahead of longitudinal position <sx> that are intersections
        :param sx: start looking from that station on this GFF
        :return: indices of the lane segments ahead of longitudinal position <sx> that are intersections
        """
        is_intersection_ahead = np.logical_and(self.segments_is_intersection, (self.segments_s_offsets[:-1] > sx))
        next_intersection_idxs = np.argwhere(is_intersection_ahead)
        return next_intersection_idxs

    def _get_segment_idx(self, lane_segment: LaneSegmentKey):
        """
        Given an array of segment_ids, this method returns the indices of these segment_ids in self.sub_segments
        :param segment_ids:
        :return:
        """
        return self.lane_segment_index_by_key[lane_segment]

    def _get_segment_idxs_from_s(self, s_values: np.ndarray):
        """
        for each longitudinal progress on the curve, s, return the index of the segment it belonged to
        from self.sub_segments (the index of the frame it was on before the concatenation into a generalized frenet frame)
        :param s_values: an np.array object containing longitudinal progresses on the generalized frenet curve.
        :return: the indices of the respective frames in self.sub_segments
        """
        segments_idxs = np.searchsorted(self.segments_s_offsets, s_values) - 1
        segments_idxs[s_values == 0] = 0

        assert np.all(segments_idxs >= 0), \
            "indices out of bounds (<0) for s_values: %s" % s_values[segments_idxs < 0]
        assert np.all(segments_idxs < len(self.segments_s_offsets)), \
            "indices out of bounds (>=len) for s_values: %s" % s_values[segments_idxs >= len(self.segments_s_offsets)]

        return segments_idxs

    def _get_segment_attributes(self, lane_segment_key: LaneSegmentKey) -> LaneSegmentAttributes:
        return self.lane_segments_dict[lane_segment_key]
