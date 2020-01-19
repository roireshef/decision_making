from logging import Logger

import numpy as np
from decision_making.src.exceptions import MappingException
from decision_making.src.global_constants import LANE_MERGE_STATE_FAR_AWAY_DISTANCE, MAX_FORWARD_HORIZON, \
    MAX_BACKWARD_HORIZON, LANE_MERGE_ACTORS_HORIZON, LANE_MERGE_RED_LINE_EXTENTION
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.scene_static_enums import ManeuverType
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState, \
    RoadSemanticOccupancyGrid, DynamicObjectWithRoadSemantics
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeState
from decision_making.src.planning.behavioral.state.lane_merge_actor_state import LaneMergeActorState
from decision_making.src.planning.types import FS_DX, FS_SX, FS_2D_LEN, FrenetState1D, FS_1D_LEN, FrenetState2D, FS_SV
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.state.state import State, EgoState, DynamicObject, ObjectSize
from decision_making.src.utils.map_utils import MapUtils
from typing import List, Dict

from rte.python.logger.AV_logger import AV_Logger


class LaneMergeState(BehavioralGridState):
    def __init__(self, road_occupancy_grid: RoadSemanticOccupancyGrid, ego_state: EgoState,
                 extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame],
                 projected_ego_fstates: Dict[RelativeLane, FrenetState2D],
                 merge_from_s_on_ego_gff: float, red_line_s_on_ego_gff: float, target_rel_lane: RelativeLane,
                 lane_change_state: LaneChangeState, logger: Logger):
        """
        lane merge state
        :param red_line_s_on_ego_gff: s of the red line on SAME_LANE GFF
        Red line is s coordinate, from which host starts to interference laterally with the main road actors.
        We assume that there is a host's road segment starting from the red line and ending at the merge point.
        If initial_lane_id == segment.e_i_SegmentID, then we already crossed the red line.
        :param target_rel_lane: RelativeLane of the merge target lane
        """
        super().__init__(road_occupancy_grid, ego_state, extended_lane_frames, projected_ego_fstates, {},
                         lane_change_state, {}, logger)
        self.merge_from_s_on_ego_gff = merge_from_s_on_ego_gff
        self.red_line_s_on_ego_gff = red_line_s_on_ego_gff
        self.target_rel_lane = target_rel_lane

    @property
    def ego_fstate_1d(self) -> FrenetState1D:
        """
        :return: longitudinal (1D) part of ego Frenet state projected on GFF
        """
        return self.projected_ego_fstates[RelativeLane.SAME_LANE][:FS_DX]

    @ego_fstate_1d.setter
    def ego_fstate_1d(self, s_state):
        """
        :set longitudinal (1D) part of ego Frenet state projected on GFF
        """
        self.projected_ego_fstates[RelativeLane.SAME_LANE] = np.concatenate((s_state, np.zeros(FS_1D_LEN)))

    @property
    def actors_states(self) -> List[LaneMergeActorState]:
        return [LaneMergeActorState(obj.longitudinal_distance, obj.dynamic_object.velocity, obj.dynamic_object.size.length)
                for lon_pos in RelativeLongitudinalPosition if (self.target_rel_lane, lon_pos) in self.road_occupancy_grid
                for obj in self.road_occupancy_grid[(self.target_rel_lane, lon_pos)]]

    def __str__(self) -> str:
        return f'EGO: {self.ego_fstate_1d} + \r\nACTORS:{[[actor.s_relative_to_ego, actor.velocity, actor.length] for actor in self.actors_states]}'

    @classmethod
    def create_from_behavioral_state(cls, behaviral_state: BehavioralGridState, target_lane: RelativeLane):
        return cls(behaviral_state.road_occupancy_grid, behaviral_state.ego_state, behaviral_state.extended_lane_frames,
                   behaviral_state.projected_ego_fstates, 0, np.inf, target_lane, behaviral_state.lane_change_state,
                   behaviral_state.logger)

    @classmethod
    def create_from_state(cls, state: State, route_plan: RoutePlan, lane_change_state: LaneChangeState, logger: Logger):
        """
        Create LaneMergeState from a given State.
        The output state has two GFFs: for same lane and for the target lane.
        Actors's longitudinal distances from ego are aligned to the merge point that is common to both GFFs.
        :param state: current state from scene_dynamic
        :param route_plan: route plan
        :param lane_change_state: not in use
        :param logger:
        :return: LaneMergeState
        """
        ego_state = state.ego_state
        ego_lane_id, ego_lane_fstate = ego_state.map_state.lane_id, ego_state.map_state.lane_fstate

        # find merge lane_id of ego_gff, merge side and the first common lane_id
        merge_lane_id = MapUtils.get_merge_lane_id(
            ego_lane_id, ego_lane_fstate[FS_SX], LANE_MERGE_STATE_FAR_AWAY_DISTANCE, route_plan, logger)
        downstream_connectivity = MapUtils.get_lane(merge_lane_id).as_downstream_lanes[0]
        maneuver_type, common_lane_id = downstream_connectivity.e_e_maneuver_type, downstream_connectivity.e_i_lane_segment_id

        target_rel_lane = RelativeLane.LEFT_LANE if maneuver_type == ManeuverType.LEFT_MERGE_CONNECTION else RelativeLane.RIGHT_LANE

        try:
            # create GFF for the host's lane
            ego_gff = BehavioralGridState._get_generalized_frenet_frames(
                lane_id=ego_lane_id, station=ego_lane_fstate[FS_SX], route_plan=route_plan, logger=logger)[RelativeLane.SAME_LANE]

            # project ego on its GFF
            ego_on_same_gff = ego_gff.convert_from_segment_state(ego_lane_fstate, ego_lane_id)

            # merge_lane_origin_s is the origin of merge_lane_id (the last lane segment before the merge point)
            merge_lane_origin_s = ego_gff.convert_from_segment_state(np.zeros(FS_2D_LEN), merge_lane_id)[FS_SX]

            # calculate merge point s relative to ego
            merge_point_on_ego_gff = ego_gff.convert_from_segment_state(np.zeros(FS_2D_LEN), common_lane_id)[FS_SX]
            merge_dist = merge_point_on_ego_gff - ego_on_same_gff[FS_SX]

            # Create target GFF for the merge with the backward & forward horizons.
            # Since both horizons are relative to ego while common_lane_id starts at the merge-point,
            # then merge_dist (merge-point relative to ego) is added to MAX_BACKWARD_HORIZON and subtracted
            # from MAX_FORWARD_HORIZON.
            target_gff = BehavioralGridState._get_generalized_frenet_frames(
                lane_id=common_lane_id, station=0, route_plan=route_plan, logger=logger,
                forward_horizon=MAX_FORWARD_HORIZON - merge_dist,
                backward_horizon=MAX_BACKWARD_HORIZON + merge_dist)[RelativeLane.SAME_LANE]

            all_gffs = {RelativeLane.SAME_LANE: ego_gff, target_rel_lane: target_gff}

            # Project ego on target_GFF, such that the projection has the same distance to the merge point as ego
            # itself. The lateral parameters of the projection are zeros.
            ego_on_target_gff = np.concatenate((ego_on_same_gff[:FS_DX], np.zeros(FS_1D_LEN)))
            merge_point_on_target_gff = target_gff.convert_from_segment_state(np.zeros(FS_2D_LEN), common_lane_id)[FS_SX]
            ego_on_target_gff[FS_SX] += merge_point_on_target_gff - merge_point_on_ego_gff
            projected_ego = {RelativeLane.SAME_LANE: ego_on_same_gff, target_rel_lane: ego_on_target_gff}

            # create road_occupancy_grid by using the appropriate BehavioralGridState methods
            actors_with_road_semantics = \
                sorted(BehavioralGridState._add_road_semantics(state.dynamic_objects, all_gffs, projected_ego, logger),
                       key=lambda rel_obj: abs(rel_obj.longitudinal_distance))
            road_occupancy_grid = BehavioralGridState._project_objects_on_grid(actors_with_road_semantics, ego_state)

            return cls(road_occupancy_grid, ego_state, all_gffs, projected_ego,
                       merge_from_s_on_ego_gff=merge_lane_origin_s,  # TODO: change it when the map will be fixed
                       red_line_s_on_ego_gff=merge_lane_origin_s + LANE_MERGE_RED_LINE_EXTENTION,
                       target_rel_lane=target_rel_lane, lane_change_state=lane_change_state, logger=logger)

        except MappingException as e:
            # in case of failure to build GFF for SAME_LANE or target lane GFF, stop processing this BP frame
            raise AssertionError("Trying to fetch data for %s, but data is unavailable. %s" % (RelativeLane.SAME_LANE, str(e)))

    @classmethod
    def create_thin_state(cls, ego_length: float, ego_fstate: FrenetState1D,
                          actors_lane_merge_state: List[LaneMergeActorState], front_actor: LaneMergeActorState,
                          merge_from_s: float, red_line_s: float):
        """
        This function may be used by RL training procedure, where ego & actors are created in a fast simple simulator,
        like SUMO, where scene_static (with Frenet frames) does not exist.
        Create LaneMergeState without GFFs and without Cartesian & Frenet coordinates of ego & actors.
        In using s coordinates, we assume some virtual GFF along the host's lane.
        :param ego_length: [m] ego length
        :param ego_fstate: 1D ego Frenet state (only s dimension, s is relative to the virtual GFF's origin)
        :param actors_lane_merge_state: list of actors states, each of type LaneMergeActorState
        :param merge_from_s: s of the point from which the merge is allowed (dashed line)
        :param red_line_s: [m] s of the red line (relative to the virtual GFF's origin)
        :return: LaneMergeState
        """
        ego_state = EgoState.create_from_cartesian_state(
            obj_id=0, timestamp=0, cartesian_state=np.array([0, 0, 0, ego_fstate[FS_SV], 0, 0]),
            size=ObjectSize(ego_length, 0, 0), confidence=1, off_map=False)

        target_rel_lane = RelativeLane.LEFT_LANE  # does not matter LEFT or RIGHT, since the merge problem is symmetric
        road_occupancy_grid = {(target_rel_lane, RelativeLongitudinalPosition.PARALLEL):
                               [DynamicObjectWithRoadSemantics(
                                   DynamicObject.create_from_cartesian_state(
                                       obj_id=i+1, timestamp=0, cartesian_state=np.array([0, 0, 0, actor.velocity, 0, 0]),
                                       size=ObjectSize(actor.length, 0, 0), confidence=1, off_map=False),
                                   longitudinal_distance=actor.s_relative_to_ego, relative_lanes=[target_rel_lane])
                                for i, actor in enumerate(actors_lane_merge_state)],

                               (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT):
                               [DynamicObjectWithRoadSemantics(DynamicObject.create_from_cartesian_state(
                                   obj_id=len(actors_lane_merge_state)+1, timestamp=0,
                                   cartesian_state=np.array([0, 0, 0, front_actor.velocity, 0, 0]),
                                   size=ObjectSize(front_actor.length, 0, 0), confidence=1, off_map=False),
                                   longitudinal_distance=front_actor.s_relative_to_ego, relative_lanes=[RelativeLane.SAME_LANE])]
                               if front_actor is not None else []
                              }

        ego_fstate2D = np.concatenate((ego_fstate, np.zeros(FS_1D_LEN)))
        return cls(road_occupancy_grid, ego_state, {}, {RelativeLane.SAME_LANE: ego_fstate2D},
                   merge_from_s, red_line_s, target_rel_lane, None, AV_Logger.get_logger())
