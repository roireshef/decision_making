from typing import List, Optional
from enum import Enum

import numpy as np
import numpy_indexed as npi

from interface.Rte_Types.python.sub_structures.TsSYS_FrenetSubsegment import TsSYSFrenetSubsegment
from interface.Rte_Types.python.sub_structures.TsSYS_GeneralizedFrenetSerretFrame import TsSYSGeneralizedFrenetSerretFrame
from decision_making.src.utils.serialization_utils import SerializationUtils

from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import CartesianPath2D, FrenetState2D, FrenetStates2D, NumpyIndicesArray, FS_SX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.exceptions import OutOfSegmentFront, OutOfSegmentBack
from decision_making.src.utils.geometry_utils import Euclidean
import rte.python.profiler as prof


class FrenetSubSegment(PUBSUB_MSG_IMPL):
    def __init__(self, segment_id: int, s_start: float, s_end: float):
        """
        An object containing information on a partial lane segment, used for concatenating or splitting of frenet frames
        :param segment_id:usually lane_id, indicating which lanes are taken to build the generalized frenet frames.
        :param s_start: starting longitudinal s to be taken into account when segmenting frenet frames.
        :param s_end: ending longitudinal s to be taken into account when segmenting frenet frames.
        """
        self.e_i_SegmentID = segment_id
        self.e_i_SStart = s_start
        self.e_i_SEnd = s_end

    def serialize(self) -> TsSYSFrenetSubsegment:
        pubsub_msg = TsSYSFrenetSubsegment()
        pubsub_msg.e_i_SegmentID = self.e_i_SegmentID
        pubsub_msg.e_i_SStart = self.e_i_SStart
        pubsub_msg.e_i_SEnd = self.e_i_SEnd
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSFrenetSubsegment):
        return cls(pubsubMsg.e_i_SegmentID, pubsubMsg.e_i_SStart, pubsubMsg.e_i_SEnd)

    def __eq__(self, other: 'FrenetSubSegment'):
        if self is other:
            return True
        return self.e_i_SegmentID == other.e_i_SegmentID and \
               self.e_i_SStart == other.e_i_SStart and \
               self.e_i_SEnd == other.e_i_SEnd

    def __hash__(self):
        return hash((self.e_i_SegmentID, self.e_i_SStart, self.e_i_SEnd))


class GFFType(Enum):
    Normal = 0
    Augmented = 1
    Partial = 2
    AugmentedPartial = 3

    @staticmethod
    def get(is_partial, is_augmented):
        if is_partial and is_augmented:
            return GFFType.AugmentedPartial
        elif is_partial:
            return GFFType.Partial
        elif is_augmented:
            return GFFType.Augmented
        else:
            return GFFType.Normal


class GeneralizedFrenetSerretFrame(FrenetSerret2DFrame, PUBSUB_MSG_IMPL):
    def __init__(self, points: CartesianPath2D, T: np.ndarray, N: np.ndarray, k: np.ndarray, k_tag: np.ndarray,
                 segment_ids: np.ndarray, segments_s_start: np.ndarray, segments_s_offsets: np.ndarray,
                 segments_ds: np.ndarray, segments_point_offset: np.ndarray, gff_type: Optional[GFFType] = None):
        """
        A generalized frenet frame, which is a concatenation of some frenet frames or a part of them.
        A special case might be a subsegment of a single frenet frame.
        :param points: 2D numpy array of points sampled from a smooth curve (x,y axes; ideally a spline of high order)
        :param T: 2D numpy array of tangent unit vectors (x,y axes) of <points>
        :param N: 2D numpy array of normal unit vectors (x,y axes) of <points>
        :param k: 1D numpy array of curvature values at each point in <points>
        :param k_tag: 1D numpy array of values of 1st derivative of curvature at each point in <points>
        :param segment_ids: id (usually lane id) of the frenet frames that make up the GFF
        :param segments_s_start: starting longitudinal s to be taken into account for each frenet frame segment
        :param segments_s_offsets: The accumulated longitudinal progress on the generalized frenet frame for each segment,
                                    e.g. segments_s_offsets[2] contains the length of (subsegment #0 + subsegment #1)
        :param segments_ds: the resolution of longitudinal distance along the curve (progress diff between points in <points>), per segment
        :param segments_point_offset:The accumulated number of points participating in the generation of the generalized frenet frame
                                     for each segment, segments_points_offset[2] contains the number of points taken from subsegment #0
                                     plus the number of points taken from subsegment #1.
        :param gff_type: type of GFF (Normal, Partial, Augmented, AugmentedPartial). This field is only used within the planning module,
                         and will be lost when the object is published.
        """
        FrenetSerret2DFrame.__init__(self, points, T, N, k, k_tag, None)
        self._segment_ids = segment_ids
        self._segments_s_start = segments_s_start
        self._segments_s_offsets = segments_s_offsets
        self._segments_ds = segments_ds
        self._segments_point_offset = segments_point_offset
        self._gff_type = gff_type

    def serialize(self) -> TsSYSGeneralizedFrenetSerretFrame:
        # GFF Type information will be lost when serialized!
        pubsub_msg = TsSYSGeneralizedFrenetSerretFrame()
        pubsub_msg.s_Points = SerializationUtils.serialize_non_typed_array(self.O)
        pubsub_msg.s_T = SerializationUtils.serialize_non_typed_array(self.T)
        pubsub_msg.s_N = SerializationUtils.serialize_non_typed_array(self.N)
        pubsub_msg.s_K = SerializationUtils.serialize_non_typed_array(self.k)
        pubsub_msg.s_KTag = SerializationUtils.serialize_non_typed_array(self.k_tag)
        pubsub_msg.s_SegmentsID = SerializationUtils.serialize_non_typed_int_array(self._segment_ids)
        pubsub_msg.s_SegmentsSStart = SerializationUtils.serialize_non_typed_array(self._segments_s_start)
        pubsub_msg.s_SegmentsSOffsets = SerializationUtils.serialize_non_typed_array(self._segments_s_offsets)
        pubsub_msg.s_SegmentsDS = SerializationUtils.serialize_non_typed_array(self._segments_ds)
        pubsub_msg.s_SegmentsPointOffset = SerializationUtils.serialize_non_typed_int_array(self._segments_point_offset)

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSGeneralizedFrenetSerretFrame):
        # GFF type will not be set when the message is deserialized.
        # This must be manually set if the GFF is to be used. 
        return cls(SerializationUtils.deserialize_any_array(pubsubMsg.s_Points),
                   SerializationUtils.deserialize_any_array(pubsubMsg.s_T),
                   SerializationUtils.deserialize_any_array(pubsubMsg.s_N),
                   SerializationUtils.deserialize_any_array(pubsubMsg.s_K),
                   SerializationUtils.deserialize_any_array(pubsubMsg.s_KTag),
                   SerializationUtils.deserialize_any_array(pubsubMsg.s_SegmentsID),
                   SerializationUtils.deserialize_any_array(pubsubMsg.s_SegmentsSStart),
                   SerializationUtils.deserialize_any_array(pubsubMsg.s_SegmentsSOffsets),
                   SerializationUtils.deserialize_any_array(pubsubMsg.s_SegmentsDS),
                   SerializationUtils.deserialize_any_array(pubsubMsg.s_SegmentsPointOffset),
                   None)

    @property
    def segments(self) -> List[FrenetSubSegment]:
        """
        :return: List of frenet sub segments with the starting and ending s in the GFF that correspond to that segment
        """
        # self._segments_s_offsets is of length len(self._segment_ids) + 1, the first len(self._segment_ids) correspond
        # to the initial s of every segment and the last len(self._segments_id) correspond to the final s of every
        # segment
        segments_start = self._segments_s_offsets[:-1]
        segments_end = self._segments_s_offsets[1:]
        return [FrenetSubSegment(segment_id=self._segment_ids[idx],
                                 s_start=segments_start[idx],
                                 s_end=segments_end[idx])
                for idx in range(len(self._segment_ids))]

    @property
    def ds(self):
        raise NotImplementedError('GeneralizedFrenetSerretFrame doesn\'t have a single ds value.')

    @property
    def s_max(self):
        """
        :return: the largest longitudinal value of the generalized frenet frame (end of curve).
        """
        return self._segments_s_offsets[-1]

    @classmethod
    @prof.ProfileFunction()
    def build(cls, frenet_frames: List[FrenetSerret2DFrame], sub_segments: List[FrenetSubSegment], gff_type: Optional[GFFType] = None):
        """
        Create a generalized frenet frame, which is a concatenation of some frenet frames or a part of them.
        A special case might be a sub segment of a single frenet frame.
        :param frenet_frames: a list of all frenet frames involved in creating the new generalized frame.
        :param sub_segments: a list of FrenetSubSegment objects, used for segmenting the respective elements from
        the frenet_frames parameter.
        :param gff_type: type of GFF (Normal, Partial, Augmented, AugmentedPartial)
        :return: A new GeneralizedFrenetSerretFrame built out of different other frenet frames.
        """
        # If the GFF type is not provided, default to Normal
        gff_type = gff_type or GFFType.Normal

        segments_id = np.array([sub_seg.e_i_SegmentID for sub_seg in sub_segments])
        segments_s_start = np.array([sub_seg.e_i_SStart for sub_seg in sub_segments])
        segments_s_end = np.array([sub_seg.e_i_SEnd for sub_seg in sub_segments])
        segments_ds = np.array([frame.ds for frame in frenet_frames])
        segments_num_points_so_far = np.zeros(shape=[len(sub_segments)], dtype=int)

        # The accumulated longitudinal progress on the generalized frenet frame for each segment,
        # e.g.  segments_s_offsets[2] contains the length of (subsegment #0 + subsegment #1)
        segments_s_offsets = np.insert(np.cumsum(segments_s_end - segments_s_start), 0, 0., axis=0)

        points = np.empty(shape=[0, 2])
        T = np.empty(shape=[0, 2])
        N = np.empty(shape=[0, 2])
        k = np.empty(shape=[0, 1])
        k_tag = np.empty(shape=[0, 1])

        for i in range(len(frenet_frames)):
            frame = frenet_frames[i]
            start_ind = int(np.floor(segments_s_start[i] / segments_ds[i]))
            # if this is not the last frame then the next already has this frame's last point as its first - omit it.
            if i < len(frenet_frames) - 1:
                # if this frame is not the last frame, it must end in s_max
                # TODO: figure out how to solve it better!!
                assert segments_s_end[i] - frame.s_max < 0.01, 'frenet frame of segment %s has problems with s_max' % \
                                                                sub_segments[i].e_i_SegmentID
                end_ind = frame.points.shape[0] - 1
            else:
                end_ind = int(np.ceil(segments_s_end[i] / segments_ds[i])) + 1

            points = np.vstack((points, frame.points[start_ind:end_ind, :]))
            T = np.vstack((T, frame.T[start_ind:end_ind, :]))
            N = np.vstack((N, frame.N[start_ind:end_ind, :]))
            k = np.vstack((k, frame.k[start_ind:end_ind, :]))
            k_tag = np.vstack((k_tag, frame.k_tag[start_ind:end_ind, :]))

            segments_num_points_so_far[i] = points.shape[0]

        # The accumulated number of points participating in the generation of the generalized frenet frame
        # for each segment, segments_points_offset[2] contains the number of points taken from subsegment #0
        # plus the number of points taken from subsegment #1.
        segments_point_offset = np.insert(segments_num_points_so_far, 0, 0., axis=0)

        return cls(points, T, N, k, k_tag, segments_id, segments_s_start, segments_s_offsets, segments_ds,
                   segments_point_offset, gff_type)

    def has_segment_id(self, segment_id: int) -> bool:
        """see has_segment_ids"""
        return self.has_segment_ids(np.array([segment_id]))[0]

    @property
    def gff_type(self):
        return self._gff_type

    @property
    def segment_ids(self):
        return self._segment_ids

    def has_segment_ids(self, segment_ids: np.array) -> np.array:
        """
        returns boolean value indicating if segment id(s) is part of this generalized frame.
        :param segment_ids:
        :return: boolean multi-dimensional array of the same size of <segment_ids> that has True whenever segment_ids[.]
        exists in self._segment_ids
        """
        if len(segment_ids) == 0:
            return np.array([], dtype=bool)
        assert np.issubdtype(segment_ids.dtype, np.integer), 'Array of indices should have int type'
        return np.isin(segment_ids, self._segment_ids)

    def convert_from_segment_states(self, frenet_states: FrenetStates2D, segment_ids: NumpyIndicesArray) -> FrenetStates2D:
        """
        Converts frenet_states on a frenet_frame to frenet_states on the generalized frenet frame.
        :param frenet_states: frenet_states on another frenet_frame which was part in building the generalized frenet frame.
                TODO: It may NOT have more than 2 dimensions.
        :param segment_ids: segment_ids, usually lane_ids, of one of the frenet frames which were used in building the generalized frenet frame.
                TODO: It may NOT have more than 1 dimension.
        :return: frenet states on the generalized frenet frame.
        """
        segment_idxs = self._get_segment_idxs_from_ids(segment_ids)
        s_offset = self._segments_s_offsets[segment_idxs]
        s_start = self._segments_s_start[segment_idxs]
        new_frenet_states = frenet_states.copy()
        new_frenet_states[..., FS_SX] += s_offset
        # For points that belong to the first subsegment, the frame bias (initial s) have to be added
        new_frenet_states[..., FS_SX] -= s_start
        return new_frenet_states

    def convert_from_segment_state(self, frenet_state: FrenetState2D, segment_id: int) -> FrenetState2D:
        """
        Converts a frenet_state on a frenet_frame to a frenet_state on the generalized frenet frame.
        :param frenet_state: a frenet_state on another frenet_frame which was part in building the generalized frenet frame.
        :param segment_id: a segment_id, usually lane_id, of one of the frenet frames which were used in building the
        generalized frenet frame.
        :return: a frenet state on the generalized frenet frame.
        """
        return self.convert_from_segment_states(frenet_state[np.newaxis, ...], np.array([segment_id]))[0]

    def convert_to_segment_states(self, frenet_states: FrenetStates2D) -> (List[int], FrenetStates2D):
        """
        Converts frenet_states on the generalized frenet frame to frenet_states on a frenet_frame it's built from.
        :param frenet_states: frenet_states on the generalized frenet frame. It may have more than 2 dimensions.
        :return: a tuple: ((segment_ids, usually lane_ids, these frenet_states will land on after the conversion),
        (the resulted frenet states))
        """
        # Find the closest greater segment offset for each frenet state longitudinal
        if np.max(frenet_states[..., FS_SX]) > self.s_max:
            raise OutOfSegmentFront("frenet_states[%s, FS_SX] = %s exceeds the frame length %f" %
                                    (np.argmax(frenet_states[..., FS_SX]), np.max(frenet_states[..., FS_SX]), self.s_max))
        segment_idxs = self._get_segment_idxs_from_s(frenet_states[..., FS_SX])
        s_offset = self._segments_s_offsets[segment_idxs]
        s_start = self._segments_s_start[segment_idxs]
        new_frenet_states = frenet_states.copy()
        new_frenet_states[..., FS_SX] -= s_offset
        # For points that belong to the first subsegment, the frame bias (initial s) have to be added
        new_frenet_states[..., FS_SX] += s_start
        return self._segment_ids[segment_idxs], new_frenet_states

    def convert_to_segment_state(self, frenet_state: FrenetState2D) -> (int, FrenetState2D):
        """
        Converts a frenet_state on the generalized frenet frame to a frenet_state on a frenet_frame it's built from.
        :param frenet_state: a frenet_state on the generalized frenet frame.
        :return: a tuple: (the segment_id, usually lane_id, this frenet_state will land on after the conversion, the resulted frenet state)
        """
        ids, fstates = self.convert_to_segment_states(frenet_state[np.newaxis, ...])
        return ids[0], fstates[0]

    def _get_segment_idxs_from_ids(self, segment_ids: NumpyIndicesArray):
        """
        Given an array of segment_ids, this method returns the indices of these segment_ids in self.sub_segments
        :param segment_ids:
        :return:
        """
        return npi.indices(self._segment_ids, segment_ids)

    def get_segment_idxs_from_ids(self, segment_ids: NumpyIndicesArray) -> NumpyIndicesArray:
        """
        Given an array of segment_ids, this method returns the indices of these segment_ids in self.segments
        :param segment_ids: Ids of segment to find
        :return: For each segment id return the index of the corresponding FrenetSubSegment in self.segments
        If no corresponding subsegment was found return -1
        """
        if len(segment_ids) == 0:
            return np.array([], dtype=np.int)

        return npi.indices(self._segment_ids, segment_ids, missing=-1)

    def _get_segment_idxs_from_s(self, s_values: np.ndarray):
        """
        for each longitudinal progress on the curve, s, return the index of the segment it belonged to
        from self.sub_segments (the index of the frame it was on before the concatenation into a generalized frenet frame)
        :param s_values: an np.array object containing longitudinal progresses on the generalized frenet curve.
        :return: the indices of the respective frames in self.sub_segments
        """
        segments_idxs = np.searchsorted(self._segments_s_offsets, s_values) - 1
        segments_idxs[s_values == 0] = 0
        return segments_idxs

    def _approximate_s_from_points(self, points: np.ndarray, raise_on_points_out_of_frame: bool = True):
        """
        Given cartesian points, this method approximates the s longitudinal progress of these points on
        the frenet frame.
        If raise_on_points_out_of_frame=False, out-of-frame points filled by nans.
        :param points: a tensor (any shape) of 2D points in cartesian frame (same origin as self.O)
        :param raise_on_points_out_of_frame: if False, don't raise exception if there are input points out of GFF
        :return: approximate s value on the frame that will be created using self.O
        """
        # Find the index of closest point on curve, and a fractional value representing the projection on this point.
        O_idx, delta_s = Euclidean.project_on_piecewise_linear_curve(points, self.O, raise_on_points_out_of_frame)
        # given the fractional index of the point (O_idx+delta_s), find which segment it belongs to based
        # on the points offset of each segment
        return self.get_s_from_index_on_frame(O_idx, delta_s, raise_on_points_out_of_frame)

    def get_s_from_index_on_frame(self, O_idx: np.ndarray, delta_s: np.ndarray, raise_on_points_out_of_frame: bool = True):
        """
        given fractional index of the point (O_idx+delta_s), find which segment it belongs to based on
        the points offset of each segment
        If raise_on_points_out_of_frame=False, out-of-frame points filled by nans.
        :param O_idx: tensor of segment index per point in <points>,
        :param delta_s: tensor of progress of projection of each point in <points> on its relevant segment
        :param raise_on_points_out_of_frame: if False, don't raise exception if there are input points out of FrenetFrame
        :return: approximate s value on the frame that will be created using self.O
        """
        is_valid = (O_idx >= 0)
        # if raise_on_points_out_of_frame=False, calculate s_approx only for valid points (non-negative indices)
        if not raise_on_points_out_of_frame and not np.all(is_valid):
            O_idx = O_idx[is_valid]
            delta_s = delta_s[is_valid]

        segment_idx_per_point = np.searchsorted(self._segments_point_offset, np.add(O_idx, delta_s)) - 1

        if (segment_idx_per_point >= len(self._segments_ds)).any():
            raise OutOfSegmentFront("Cannot extrapolate, desired progress (%s) is out of the curve (s_max = %s)." %
                                    (np.add(O_idx, delta_s), self.s_max))
        if (segment_idx_per_point < 0).any():
            raise OutOfSegmentBack("Cannot extrapolate, desired progress (%s) is out of the curve" % np.add(O_idx, delta_s))

        # get ds of every point based on the ds of the segment
        ds = self._segments_ds[segment_idx_per_point]

        # calculate the offset of the first segment starting relative to the first GFF point
        # subtract this offset from s_approx for the points on the first segment (in other segments this offset is 0)
        intra_point_offsets = np.zeros_like(ds)
        intra_point_offsets[segment_idx_per_point == 0] = self._segments_s_start[0] % self._segments_ds[0]
        # The approximate longitudinal progress is the longitudinal offset of the segment plus the in-segment-index
        # times the segment ds.
        valid_s_approx = self._segments_s_offsets[segment_idx_per_point] - intra_point_offsets + \
                         (O_idx + delta_s - self._segments_point_offset[segment_idx_per_point]) * ds

        # if raise_on_points_out_of_frame=False, set negative s_approx for invalid points
        if valid_s_approx.size != is_valid.size:
            s_approx = np.full(is_valid.shape, -1.)
            s_approx[is_valid] = valid_s_approx
        else:
            s_approx = valid_s_approx

        return s_approx

    def get_closest_index_on_frame(self, s: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        from s, a vector of longitudinal progress on the frame, return the index of the closest point on the frame and
        a value in the range [0, ds] representing the projection on this closest point.
        The returned values, if summed, represent a "fractional index" on the curve.
        :param s: a vector of longitudinal progress on the frame
        :return: a tuple of: indices of closest points, a vector of projections on these points.
        """
        # get the index of the segment that contains each s value
        segment_idxs = self._get_segment_idxs_from_s(s)
        # get ds of every point based on the ds of the segment
        ds = self._segments_ds[segment_idxs]
        # subtracting the s offset, we get the s value in the associated segment.
        s_in_segment = s - self._segments_s_offsets[segment_idxs]
        # get the points offset of the segment that each longitudinal value resides in.
        segment_points_offset = self._segments_point_offset[segment_idxs]

        # generally, the first GFF point lays outside the GFF, since GFF's origin does not coincide with any map point
        # for all points on the first segment get offset of the GFF origin from the first GFF point
        intra_point_offsets = np.zeros_like(s)
        intra_point_offsets[segment_idxs == 0] = self._segments_s_start[0] % self._segments_ds[0]

        # get the progress in index units (progress_in_points is a "floating-point index")
        progress_in_points = np.divide(s_in_segment + intra_point_offsets, ds) + segment_points_offset

        # calculate and return the integer and fractional parts of the index
        O_idx = np.round(progress_in_points).astype(np.int)

        delta_s = np.expand_dims((progress_in_points - O_idx) * ds, axis=len(s.shape))

        return O_idx, delta_s
