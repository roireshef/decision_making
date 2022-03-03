import heapq
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set
from typing import NamedTuple

import numpy as np
from decision_making.src.planning.types import FrenetState2D, FS_SX, FS_DX
from decision_making.src.rl_agent.environments.uc_rl_map import UCRLMap, RelativeLane
from decision_making.src.rl_agent.global_types import LongitudinalDirection
from decision_making.src.rl_agent.utils.class_utils import Representable, Cloneable, CloneableNamedTuple
from decision_making.src.rl_agent.utils.generalized_frenet_frame import LaneSegmentKey


class VehicleSize(NamedTuple, CloneableNamedTuple):
    length: float
    width: float


class LaneChangeState(Representable, Cloneable):
    """ Ego vehicle's state of changing-lane """
    def __init__(self, target_relative_lane: RelativeLane, target_offset: float, time_to_target: float,
                 start_time: Optional[float] = None, neg_offset_arrival_time: Optional[float] = None,
                 abort_time: Optional[float] = None, commit_time: Optional[float] = None,
                 source_gff: Optional[str] = None):
        """
        :param target_relative_lane: the RelativeLane currently targeting (relative to ego gff)
        :param target_offset: the lateral offset (dx) currently targeting
        :param source_gff: The original GFF ID from which the lane change started
        """
        assert commit_time is None or start_time is not None
        assert abort_time is None or start_time is not None
        assert neg_offset_arrival_time is None or start_time is not None

        assert abort_time is None or target_relative_lane == RelativeLane.SAME_LANE

        assert time_to_target == 0 or start_time    # if time to target > 0, lane change is active

        self.time_to_target = time_to_target
        self.source_gff = source_gff
        self.target_relative_lane = target_relative_lane
        self.target_offset = target_offset
        self.start_time = start_time
        self.neg_offset_arrival_time = neg_offset_arrival_time
        self.abort_time = abort_time
        self.commit_time = commit_time

    @classmethod
    def get_inactive(cls):
        """ constructor for an inactive lane change state (target is same and time from start = 0, any one of them
        can independently indicate there is no lane change active """
        return cls(target_relative_lane=RelativeLane.SAME_LANE, target_offset=.0, start_time=None, time_to_target=.0,
                   neg_offset_arrival_time=None, abort_time=None, commit_time=None, source_gff=None)

    @property
    def is_active(self):
        """ if lane change is active, time_from_start is already > 0"""
        return not self.is_inactive

    @property
    def is_inactive(self):
        return self.start_time is None

    @property
    def is_negotiating(self):
        """ Is host vehicle targeting (on the way or there already) the negotiation offset?"""
        return self.is_active and not self.is_committing and not self.is_aborting

    @property
    def is_moving_to_negotiation_offset(self):
        """ Is host vehicle on the way to the negotiation offset?"""
        return self.is_negotiating and self.neg_offset_arrival_time is None

    @property
    def is_committing(self):
        """ Is host vehicle targeting the adjacent lane? (full lane change) """
        return self.commit_time is not None

    @property
    def is_aborting(self):
        """ Is host vehicle aborting from negotiation? (targeting original lane's center) """
        return self.abort_time is not None

    @property
    def status_vector(self):
        return np.array([self.is_active, self.is_negotiating, self.is_committing, self.is_aborting])

    def time_since_start(self, now: float) -> Optional[float]:
        return now - self.start_time if self.start_time else None

    def time_since_negotiation_offset_arrival(self, now: float) -> Optional[float]:
        return now - self.neg_offset_arrival_time if self.neg_offset_arrival_time else None

    def time_since_negotiation_offset_deprature(self, now: float) -> Optional[float]:
        left_time = self.abort_time or self.commit_time
        return now - left_time if self.neg_offset_arrival_time and left_time else None

    def time_since_abort(self, now: float) -> Optional[float]:
        return now - self.abort_time if self.abort_time else None

    def time_since_commit(self, now: float) -> Optional[float]:
        return now - self.commit_time if self.commit_time else None

    def time_since_vector(self, now: float) -> np.ndarray:
        return np.array([self.time_since_start(now) if self.is_moving_to_negotiation_offset else 0,
                         self.time_since_commit(now) or 0,
                         self.time_since_abort(now) or 0])


class EgoCentricActorState(NamedTuple, CloneableNamedTuple):
    """
    Actor's state on the main road
        :s_relative_to_ego: [m] s relative to ego (considering the distance from the actor and from ego to the
        merge point)
        :velocity: [m/sec] actor's velocity
        :acceleration: [m/sec^2] actor's acceleration
        :length: [m] actor's length
        :lane_difference: [int] actor's lane relative to host (right is negative, left is positive)
    """
    actor_id: str
    s_relative_to_ego: float
    velocity: float
    acceleration: float
    size: VehicleSize
    lane_difference: int

    @property
    def length(self):
        return self.size.length

    def __lt__(self, other):
        # implements comparison between actors based on proximity to ego vehicle, for efficient clipping of far actors
        return abs(self.s_relative_to_ego) < abs(other.s_relative_to_ego)


class EgoState(NamedTuple, CloneableNamedTuple):
    """
    State of ego
    Note that fstate, lane_segment and gff_id should agree, so putting an assert when using them can be helpful
        :fstate: Frenet coordinates of ego vehicle relative to the GFF in <gff_id>
        :lane_segment: The lane segment the ego is on (edge + lane_id)
        :gff_id: the id of the GeneralizedFrenetFrame the ego vehicle is localized on
        :length: length of the ego vehicle
        :lane_change_state: the state of lane change for ego agent, stores time from start and target lane
    """
    fstate: FrenetState2D
    lane_segment: LaneSegmentKey
    gff_id: str
    size: VehicleSize
    lane_change_state: Optional[LaneChangeState]

    @property
    def is_on_source_gff(self) -> bool:
        """ is true as long as lane change is inactive or ego vehicle is yet to move into target lane """
        return self.lane_change_state.is_inactive or self.gff_id == self.lane_change_state.source_gff

    @property
    def length(self):
        return self.size.length


class EgoCentricState(Representable, Cloneable):
    def __init__(self, roads_map: UCRLMap, ego_state: EgoState, actors_state: List[EgoCentricActorState],
                 existing_lanes: List[int], valid_target_lanes: List[RelativeLane], timestamp_in_sec: float,
                 ego_state_projections: Optional[Dict[str, FrenetState2D]] = None,
                 goal_relative_lane: Optional[int] = None, ego_jerk: Optional[float] = None):
        """
        Full state of ego and surroundings in an ego-centric representation
            :roads_map: map object of the whole scene
            :ego_state: state of ego vehicle
            :actors_state: list of actors states
            :map_anchors: (optional) dict(GFF -> dict(map anchor -> longitudinal position))
            :existing_lanes: list of relative lane offsets (int type) representing all lanes existing in the scene, relative
            to host lane (right is negative, left positive), even if not host is not able to lane change onto them. In a
            2-lane scene with host vehicle on the rightmost, this list would be [0, 1]. Host vehicle on the leftmost lane
            would result with [-1, 0].
            :valid_target_lanes: list indicates whether <left,same,right> lanes valid to lane change towards
            :timestamp_in_sec: simulation timestamp of when this state was acquired - populated with Env.sim_time_in_sec!
            :ego_projections: (optional) in case projections of ego fstate on other GFFs are already pre-computed, the
            mapping can be specified here to avoid recomputing projections
            :goal_relative_lane: lane difference between goal lane and current ego lane (rightwards negative)
            :ego_jerk: the immediate longitudinal jerk of the ego vehicle the time of the state (extension to ego_state)
        """
        self.roads_map = roads_map
        self.ego_state = ego_state
        self.actors_state = actors_state
        self.existing_lanes = existing_lanes
        self.valid_target_lanes = valid_target_lanes
        self.timestamp_in_sec = timestamp_in_sec
        self.goal_relative_lane = goal_relative_lane
        self.ego_jerk = ego_jerk

        # projections of ego frenet state on adjacent lanes (two dicts with keys: RelativeLane and GFF-ID)
        self.ego_fstate_on_gffs = ego_state_projections or roads_map.project_fstate_to_adjacent_gffs(
            self.ego_state.fstate, self.ego_state.gff_id)
        self.ego_fstate_on_adj_lanes = {
            rel_lane: self.ego_fstate_on_gffs[target_gff]
            for rel_lane, target_gff in self.roads_map.gff_adjacency[self.ego_state.gff_id].items()
        }

        # behavioral grid maps (RelativeLane, LongitudinalDirection) spatial slice to a min-heap of actors (based on
        # abs(distance to ego) so that closest actor can be queried in o(1)
        self.behavioral_grid = self._generate_behavioral_grid()
        self.lanes_ego_on = self._get_lanes_ego_on()

    @property
    def map_anchors(self):
        return self.roads_map.anchors

    @property
    def self_gff_anchors(self):
        """ returns the dictionary of anchors for the GFF the ego is currently assigned to """
        return self.map_anchors[self.ego_state.gff_id]

    @property
    def self_gff(self):
        return self.roads_map.gffs[self.ego_state.gff_id]

    def gff_relativity_to_ego(self, gff_id: str) -> RelativeLane:
        """ given a target GFF ID, it returns the lateral offset of this GFF relative to the GFF ego is on """
        return RelativeLane(self.roads_map.gff_lateral_offsets[(self.ego_state.gff_id, gff_id)])

    # UTILITY FUNCTIONS USED BY CONSTRUCTOR #

    def _generate_behavioral_grid(self) -> Dict[Tuple[RelativeLane, LongitudinalDirection], List[EgoCentricActorState]]:
        """ 2 x 2 behavioral grid state with actor min-heaps (of absolute distance to ego)"""
        # TODO: this lists also relative lanes the agent can't currently change-lane into
        valid_target_lane_ids = set(self.existing_lanes).intersection({-1, 0, 1})

        grid = defaultdict(list)
        for actor_state in self.actors_state:
            if actor_state.lane_difference in valid_target_lane_ids:
                # TODO: hack! parallel vehicles!
                lon_assignment = LongitudinalDirection(np.sign(actor_state.s_relative_to_ego) or 1)
                lat_assignment = RelativeLane(actor_state.lane_difference)
                grid[(lat_assignment, lon_assignment)].append(actor_state)

        # heapify every actor list so that min-distance actor can be queried in o(1)
        for cell_key, cell_actors in grid.items():
            heapq.heapify(cell_actors)

        return grid

    def _get_lanes_ego_on(self) -> Set[RelativeLane]:
        lanes = {RelativeLane.SAME_LANE}

        same_lane_width = self.roads_map.get_gff_width_at(self.ego_state.gff_id, self.ego_state.fstate[FS_SX])
        if self.ego_state.fstate[FS_DX] + self.ego_state.size.width / 2 > same_lane_width / 2:
            lanes.add(RelativeLane.LEFT_LANE)
        elif self.ego_state.fstate[FS_DX] - self.ego_state.size.width / 2 < -same_lane_width / 2:
            lanes.add(RelativeLane.RIGHT_LANE)
        elif self.ego_state.lane_segment in self.roads_map.merges:
            lanes.add(self.roads_map.merges[self.ego_state.lane_segment])

        return lanes
