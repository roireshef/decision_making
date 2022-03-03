from abc import abstractmethod
from collections import defaultdict
from enum import Enum
from itertools import product
from typing import Optional, List, Dict

import numpy as np
from decision_making.src.planning.types import FrenetState2D
from decision_making.src.rl_agent.global_types import LateralDirection
from decision_making.src.rl_agent.utils.generalized_frenet_frame import GeneralizedFrenetFrame, LaneSegmentKey
from decision_making.src.rl_agent.utils.primitive_utils import VectorizedDict


class RelativeLane(Enum):
    """"
    The lane associated with a certain Recipe, relative to ego
    """
    RIGHT_LANE = -1
    SAME_LANE = 0
    LEFT_LANE = 1

    def __invert__(self):
        return RelativeLane(-self.value)

    def __sub__(self, other):
        assert isinstance(other, RelativeLane), "can't subtract RelativeLane and %s" % type(other)
        return LateralDirection(self.value.__sub__(other.value))

    def __add__(self, other):
        assert isinstance(other, RelativeLane), "can't add RelativeLane and %s" % type(other)
        return LateralDirection(self.value.__add__(other.value))


class MapAnchor(Enum):
    YIELD_LINE = 0
    MERGE_BEGINNING = 1
    MERGE_DONE = 2
    END_OF_SEGMENT = 3


class UCRLMap:
    def __init__(self, gffs: Dict[str, GeneralizedFrenetFrame], gff_ordinal: Dict[str, int],
                 anchors: Optional[Dict[str, Dict[MapAnchor, float]]] = None):
        """
        Endpoint for working with map elements (indluing cache and utility functions)
        :param gffs: a dictionary containing all GeneralizedFrenetFrame objects (scene's lane centers) and their keys
        :param gff_ordinal: (temporary) representation of geographical order of GFFs to reconstruct connectivity
        between their lane segments. This should eventually be replaced by a more accurate connectivity repr.
        :param anchors: dictionary to map a GFF to its longitudinal anchors (map between MapAnchor and its longitudinal
        position on the GFF).
        """
        self.anchors = anchors or {}
        self.gffs = gffs
        self.gff_ordinal = gff_ordinal

        # Cache mapping between lane_segments and gff_id
        self.lane_to_gff_ids = defaultdict(lambda: [])
        for key, gff in self.gffs.items():
            for seg in gff.lane_segments:
                self.lane_to_gff_ids[seg.key].append(key)

        self.lane_to_first_gff_id = VectorizedDict({lane: gffs[0] for lane, gffs in self.lane_to_gff_ids.items()})
        self.gff_offset_for_lane_segment = VectorizedDict(
            {(gff_id, lane): self.gffs[gff_id].get_offset_for_lane_segment(lane)
             for lane, gff_id in self.lane_to_first_gff_id.items()})

        # Cache longitudinal differences between edges in the map
        self.seg_to_edge_lon_offset = {k: v for d in [gff.seg_to_edge_ahead_offsets for gff in self.gffs.values()]
                                       for k, v in d.items()}

        # Cache lateral offsets between GFFs (leftwards positive, rightwards negative)
        self.gff_lateral_offsets = VectorizedDict({(a_key, b_key): gff_ordinal[b_key] - gff_ordinal[a_key]
                                                   for (a_key, a_gff), (b_key, b_gff) in
                                                   product(self.gffs.items(), self.gffs.items())})

        # Cache GFF-ID triplets (left adj., middle, right adj.) for each of the GFFs in the map. This can be used
        # further to process actions towards adjacent lanes
        adjacency_values = [rl.value for rl in RelativeLane]
        self.gff_adjacency = {middle_gff_id: {RelativeLane(lat_diff): other_gff
                                              for (self_gff, other_gff), lat_diff in self.gff_lateral_offsets.items()
                                              if self_gff == middle_gff_id and lat_diff in adjacency_values}
                              for middle_gff_id in self.gffs.keys()}

        # this dictionary maps between each lane segment key of a merge and the relativity (left/right) to its
        # counterpart on another gff (this is based on ordinal_gff)
        self.merges = {
            self._first_merging_lane_segment(a_gff, b_gff): RelativeLane(np.sign(gff_ordinal[b_key] - gff_ordinal[a_key]).item())
            for (a_key, a_gff), (b_key, b_gff)
            in product(self.gffs.items(), self.gffs.items())
            if a_key != b_key and self._first_merging_lane_segment(a_gff, b_gff) is not None}

    @abstractmethod
    def project_fstate_to_gff(self, fstate: FrenetState2D, source_gff: str, target_gff: str) -> FrenetState2D:
        """
        Naive implementation for GFF->GFF projection of Frenet states assuming all GFFs are parallel to each other
        and their width is constant
        :param fstate: state relative to source_gff to project onto target_gff
        :param source_gff: source GFF ID
        :param target_gff: target GFF ID
        :return: a FrenetState2D vector of fstate projected onto target_gff
        """
        pass

    @abstractmethod
    def project_stations_to_gff(self, sx: np.ndarray, source_gffs: List[str], target_gff: str) -> np.ndarray:
        """
        batch-project an array of station coordinates (sx) from their list of correponding source-GFFs onto a single
        target GFF
        :param sx: 1d numpy array of station coordinates (sx)
        :param source_gffs: a list of source GFF IDs correponding to the station coordinates
        :param target_gff: a GFF ID of the target GFF to project to
        :return: 1d numpy array of the projections of <sx> on <target_gff>
        """
        pass

    @abstractmethod
    def get_gff_width_at(self, gff_id: str, sx: float) -> float:
        """
        Queries for the width of a GFF at a specific station coordinate along it
        :param gff_id: the ID of the GFF to query
        :param sx: the station from the GFF beginning in meters
        :return: the width of the GFF at station <sx>
        """
        pass

    def project_fstate_to_adjacent_gffs(self, fstate: FrenetState2D, source_gff: str) -> Dict[str, FrenetState2D]:
        """
        Takes a Frenet state and a correponding GFF id and projects the state on all adjacent GFFs
        :param fstate: the Frenet state to project
        :param source_gff: source GFF on which the state lies
        :return: dictionary that maps GFF-id to projected fstate on that GFF
        """
        return {
            target_gff: self.project_fstate_to_gff(fstate, source_gff, target_gff)
            for rel_lane, target_gff in self.gff_adjacency[source_gff].items()
        }

    def next_intersection_offset_on_gff(self, gff: str, sx: float) -> float:
        return self.gffs[gff].next_intersection_offset(sx)

    def next_intersection_segment_on_gff(self, gff: str, sx: float) -> Optional[LaneSegmentKey]:
        return self.gffs[gff].next_intersection_segment(sx)

    def next_intersection_segments_on_gff(self, gff: str, sx: float) -> List[LaneSegmentKey]:
        return self.gffs[gff].next_intersection_segments_on_gff(sx)

    @staticmethod
    def _get_gff_pair_offset(from_gff: GeneralizedFrenetFrame, to_gff: GeneralizedFrenetFrame) -> float:
        """
        For a pair of given GFF object, this function looks for an edge they share, and based on it, it computes the
        longitudinal offset between the two GFFs, for future projection between the two
        :return: offset in [m] between from_gff to to_gff
        """
        intersecting_edges = list(from_gff.edge_ids.intersection(to_gff.edge_ids))

        if len(intersecting_edges) == 0:
            return np.nan

        from_offset = from_gff.get_offset_for_edge(intersecting_edges[0])
        to_offset = to_gff.get_offset_for_edge(intersecting_edges[0])

        return to_offset - from_offset

    @staticmethod
    def _first_merging_lane_segment(gff: GeneralizedFrenetFrame, other: GeneralizedFrenetFrame) \
            -> Optional[LaneSegmentKey]:
        """ This utility method finds the first lane segment in <gff> that is merging with another lane segment in
        <other>. This is done by searching for the first shared lane segment of both <gff> and <other> and returning
        the predecessor to that lane segment in <gff> """
        intersecting_segment_keys = gff.lane_segment_keys.intersection(other.lane_segment_keys)

        if not intersecting_segment_keys:
            return None

        first_intersection = min([i for i, ls in enumerate(gff.lane_segments) if ls.key in intersecting_segment_keys])

        # if the first intersection is a first GFF's lane segment - this is not a merge, those are splitting GFFs
        return gff.lane_segments[first_intersection - 1].key if first_intersection > 0 else None


class UCRLMapWithConstWidthLanes(UCRLMap):
    def __init__(self, gffs: Dict[str, GeneralizedFrenetFrame], gff_ordinal: Dict[str, int],
                 anchors: Optional[Dict[str, Dict[MapAnchor, float]]] = None, lane_width: float = 3.2):
        super().__init__(gffs=gffs, gff_ordinal=gff_ordinal, anchors=anchors)

        # stores lane-widths for indexed by their GFFs
        self._gff_widths = {gff_id: lane_width for gff_id in self.gffs.keys()}

        # Create mapping between every pair of GFFs <from_gff, to_gff> and the offset needs to be applied if one would
        # like to project coordinates from from_gff to to_gff. self.gff_offsets is a Dict[(from_gff, to_gff): offset]
        # where offset is in [m] unit.
        # NOTE: this assumes GFFs are parallel and have no curves. This assumption breaks in the real world quite often,
        # which is why is is only implemented in this class and not UCRLMap parent
        self.gff_offsets = VectorizedDict({(a_key, b_key): self._get_gff_pair_offset(a_gff, b_gff)
                                           for (a_key, a_gff), (b_key, b_gff) in
                                           product(self.gffs.items(), self.gffs.items())})

    def project_fstate_to_gff(self, fstate: FrenetState2D, source_gff: str, target_gff: str) -> FrenetState2D:
        """
        Naive implementation for GFF->GFF projection of Frenet states assuming all GFFs are parallel to each other
        and their width is constant
        :param fstate: state relative to source_gff to project onto target_gff
        :param source_gff: source GFF ID
        :param target_gff: target GFF ID
        :return: a FrenetState2D vector of fstate projected onto target_gff
        """
        dx_diff = -self.gff_lateral_offsets[(source_gff, target_gff)] * self._gff_widths[target_gff]
        sx_diff = self.gff_offsets[(source_gff, target_gff)]
        return fstate + np.array([sx_diff, 0, 0, dx_diff, 0, 0])

    def project_stations_to_gff(self, sx: np.ndarray, source_gffs: List[str], target_gff: str) -> FrenetState2D:
        """
        Naive implementation for GFF->GFF projection of Frenet states assuming all GFFs are parallel to each other
        and their width is constant
        """
        gff_pair_offsets = self.gff_offsets[list(zip(source_gffs, [target_gff] * len(source_gffs)))]
        return sx + gff_pair_offsets

    def get_gff_width_at(self, gff_id: str, sx: float) -> float:
        """ Given that GFF width is constant, we ignore sx and return cached GFF width """
        return self._gff_widths[gff_id]
