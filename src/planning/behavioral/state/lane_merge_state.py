from logging import Logger
from decision_making.src.exceptions import MappingException
from decision_making.src.global_constants import MERGE_LOOKAHEAD
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.scene_static_enums import ManeuverType
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState, \
    RoadSemanticOccupancyGrid, DynamicObjectWithRoadSemantics
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
import numpy as np
from decision_making.src.planning.types import FS_DX, FS_SX, FS_2D_LEN, FrenetState1D, FS_1D_LEN, FrenetState2D, FS_SV
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.state.state import State, EgoState, DynamicObject, ObjectSize
from decision_making.src.utils.map_utils import MapUtils
from gym.spaces.tuple_space import Tuple as GymTuple
import torch
from typing import List, Dict

DEFAULT_ADDITIONAL_ENV_PARAMS = {
    "RED_LINE_STOPPING_SPEED": 0.01,  # what is considered  "stopping" at the redline
    "MAX_VELOCITY": 25.0,  # maximum allowed velocity [m/sec]
    "RED_LINE": 240,  # location of the red line (meters)
    "GOAL_LOCATION": 320,  # location of the goal (where host should arrive without collision)
    "MERGE_SAFETY": 0.5,  # safety margin between vehicles, expressed with seconds of headway
    "VEHICLES_PER_HOUR": 1200,  # number of vehicles to generate on target lane
    "D_HORIZON_BACKWARD": 800,  # perception horizon going backward [m]
    "D_HORIZON_FORWARD": 800,  # perception horizon going forward [m]
    "CONSIDERED_NUM_ACTORS": 1,  # desired number of other actors to be represented in state
    "ACTION_MAX_TIME_HORIZON": 40.0,  # longest allowed action
    "FAR_AWAY_DISTANCE": 300.0,  # location defined for dummy vehicles
    "HOST_INITIAL_LOCATION": 0,  # Initial location of host
    'HOST_INITIAL_SPEED': 0,  # Initial velocity of host
    'HOST_LOCATION_PERTURBATION': 0,  # Initial location of host (note this is fixed for initial state)
    'VEHICLES_DEPART_SPEED': 25,  # The initial speed for other vehicles
    'VEHICLES_INFLOW_PROBABILITY': 0.25,
    'OCCUPANCY_GRID_RESOLUTION': 4.5,
    'OCCUPANCY_GRID_ONESIDED_LENGTH': 120,
    'LON_ACC_LIMIT_FACTOR': 1,
    # Relaxation of the longitudinal acceleration filter (test LON_ACC_LIMITS * LON_ACC_LIMIT_FACTOR)
    'MAX_SIMULATION_STEPS': 1000,
    'REWARD_PER_STEP': -1,
    'REWARD_FOR_SUCCESS': 1000,
    'REWARD_FOR_FAILURE': 0,
    "REWARD_CONSTANT_SUBTRACTOR": 0,
    "REWARD_CONSTANT_MULTIPLIER": 1,
    "RED_LINE_PROXIMITY": 15,
    'WARMUP_STEPS': 0,
    'STORE_REPLAY': False,
    'JERK_REWARD_COEFFICIENT': 0.1,
    'ACTION_SPACE': {
        'MIN_VELOCITY': 0.0,
        'MAX_VELOCITY': 25.0,
        'VELOCITY_RESOLUTION': 5
    }
}


class LaneMergeActorState:
    def __init__(self, s_relative_to_ego: float, velocity: float, length: float):
        """
        Actor's state on the main road
        :param s_relative_to_ego: [m] s relative to ego (considering the distance from the actor and from ego to the merge point)
        :param velocity: [m/sec] actor's velocity
        :param length: [m] actor's length
        """
        self.s_relative_to_ego = s_relative_to_ego
        self.velocity = velocity
        self.length = length


class LaneMergeState(BehavioralGridState):
    def __init__(self, road_occupancy_grid: RoadSemanticOccupancyGrid, ego_state: EgoState,
                 extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame],
                 projected_ego_fstates: Dict[RelativeLane, FrenetState2D],
                 red_line_s_on_ego_gff: float, target_rel_lane: RelativeLane):
        """
        lane merge state
        :param red_line_s_on_ego_gff: s of the red line on SAME_LANE GFF
        :param target_rel_lane: RelativeLane of the merge target lane
        """
        super().__init__(road_occupancy_grid, ego_state, extended_lane_frames, projected_ego_fstates)
        self.red_line_s_on_ego_gff = red_line_s_on_ego_gff
        self.target_rel_lane = target_rel_lane

    @property
    def ego_fstate(self) -> FrenetState1D:
        return self.projected_ego_fstates[RelativeLane.SAME_LANE][:FS_DX]

    @property
    def actors_states(self) -> List[LaneMergeActorState]:
        return [LaneMergeActorState(obj.longitudinal_distance, obj.dynamic_object.velocity, obj.dynamic_object.size.length)
                for lon_pos in RelativeLongitudinalPosition if (self.target_rel_lane, lon_pos) in self.road_occupancy_grid
                for obj in self.road_occupancy_grid[(self.target_rel_lane, lon_pos)]]

    @property
    def ego_length(self) -> float:
        return self.ego_state.size.length

    def to_string(self) -> str:
        return f'EGO: {self.ego_fstate} + \r\nACTORS:{self.actors_states}'

    @classmethod
    def create_from_state(cls, state: State, route_plan: RoutePlan, logger: Logger):
        """
        Create LaneMergeState from a given State.
        The output state has two GFFs: for same lane and for the target lane.
        Actors's longitudinal distances from ego are aligned to the merge point that is common to both GFFs.
        :param state: current state
        :param route_plan: route plan
        :param logger:
        :return: LaneMergeState
        """
        ego_state = state.ego_state
        ego_lane_id, ego_lane_fstate = ego_state.map_state.lane_id, ego_state.map_state.lane_fstate

        # find merge lane_id of ego_gff, merge side and the first common lane_id
        merge_lane_id, maneuver_type, common_lane_id = MapUtils.get_closest_lane_merge(
            ego_lane_id, ego_lane_fstate[FS_SX], MERGE_LOOKAHEAD, route_plan)

        target_rel_lane = RelativeLane.LEFT_LANE if maneuver_type == ManeuverType.LEFT_MERGE_CONNECTION else RelativeLane.RIGHT_LANE

        try:
            # create GFF for the host's lane
            ego_gff = BehavioralGridState._get_generalized_frenet_frames(
                lane_id=ego_lane_id, station=ego_lane_fstate[FS_SX], route_plan=route_plan)[RelativeLane.SAME_LANE]

            # project ego on its GFF
            ego_on_same_gff = ego_gff.convert_from_segment_state(ego_lane_fstate, ego_lane_id)

            # set red line s to be the origin of merge_lane_id (the last lane segment before the merge point)
            red_line_s = ego_gff.convert_from_segment_state(np.zeros(FS_2D_LEN), merge_lane_id)[FS_SX]

            # calculate merge point s relative to ego
            merge_point_on_ego_gff = ego_gff.convert_from_segment_state(np.zeros(FS_2D_LEN), common_lane_id)[FS_SX]
            merge_point_from_ego = merge_point_on_ego_gff - ego_on_same_gff[FS_SX]

            # create target GFF for the merge, such that its backward & forward horizons are equal to MERGE_LOOKAHEAD
            # relative to ego
            target_gff = BehavioralGridState._get_generalized_frenet_frames(
                lane_id=common_lane_id, station=0, route_plan=route_plan, forward_horizon=MERGE_LOOKAHEAD - merge_point_from_ego,
                backward_horizon=MERGE_LOOKAHEAD + merge_point_from_ego)[RelativeLane.SAME_LANE]

            all_gffs = {RelativeLane.SAME_LANE: ego_gff, target_rel_lane: target_gff}

            # Project ego on target_GFF, such that the projection has the same distance to the merge point as ego
            # itself. The lateral parameters of the projection are zeros.
            ego_on_target_gff = np.concatenate((ego_on_same_gff[:FS_DX], np.zeros(FS_1D_LEN)))
            merge_point_on_target_gff = target_gff.convert_from_segment_state(np.zeros(FS_2D_LEN), common_lane_id)[FS_SX]
            ego_on_target_gff[FS_SX] += merge_point_on_target_gff - merge_point_on_ego_gff
            projected_ego = {RelativeLane.SAME_LANE: ego_on_same_gff, target_rel_lane: ego_on_target_gff}

            # create road_occupancy_grid by using the appropriate BehavioralGridState methods
            actors_with_road_semantics = \
                sorted(BehavioralGridState._add_road_semantics(state.dynamic_objects, all_gffs, projected_ego),
                       key=lambda rel_obj: abs(rel_obj.longitudinal_distance))
            road_occupancy_grid = BehavioralGridState._project_objects_on_grid(actors_with_road_semantics, ego_state)

            print('ego_lane=', ego_lane_id, 'dist_to_red_line=', red_line_s - ego_on_same_gff[FS_SX], 'vel=', ego_state.velocity)

            return cls(road_occupancy_grid, ego_state, all_gffs, projected_ego, red_line_s, target_rel_lane)

        except MappingException as e:
            # in case of failure to build GFF for SAME_LANE or target lane GFF, stop processing this BP frame
            raise AssertionError("Trying to fetch data for %s, but data is unavailable. %s" % (RelativeLane.SAME_LANE, str(e)))

    @classmethod
    def create_thin_state(cls, ego_len: float, ego_fstate: FrenetState1D, actors: List[LaneMergeActorState], red_line_s: float):
        """
        Create LaneMergeState without GFFs and without Cartesian & Frenet coordinates of ego & actors
        :param ego_len:
        :param ego_fstate:
        :param actors:
        :param red_line_s:
        :return: LaneMergeState
        """
        ego_state = EgoState.create_from_cartesian_state(
            0, 0, np.array([0, 0, 0, ego_fstate[FS_SV], 0, 0]), ObjectSize(ego_len, 0, 0), 1, False)

        target_rel_lane = RelativeLane.LEFT_LANE
        road_occupancy_grid = {(target_rel_lane, RelativeLongitudinalPosition.PARALLEL):
                               [DynamicObjectWithRoadSemantics(
                                   DynamicObject.create_from_cartesian_state(
                                       i+1, 0, np.array([0, 0, 0, actor.velocity, 0, 0]),
                                       ObjectSize(actor.length, 0, 0), 1, False),
                                   longitudinal_distance=actor.s_relative_to_ego)
                                for i, actor in enumerate(actors)]}

        ego_fstate2D = np.concatenate((ego_fstate, np.zeros(FS_1D_LEN)))
        return cls(road_occupancy_grid, ego_state, {}, {RelativeLane.SAME_LANE: ego_fstate2D}, red_line_s, target_rel_lane)

    def encode_state_for_RL(self) -> GymTuple:

        # encode host
        host_state = np.copy(self.ego_fstate)
        # replace the host station coordinate with its distance to red line
        host_state[FS_SX] = self.red_line_s_on_ego_gff - host_state[FS_SX]

        params = DEFAULT_ADDITIONAL_ENV_PARAMS
        grid_res = params["OCCUPANCY_GRID_RESOLUTION"]
        grid_onesided_length = params["OCCUPANCY_GRID_ONESIDED_LENGTH"]

        # actors state is an occupancy grid containing the different vehicles' distance from merge and velocity
        num_of_onesided_grid_cells = np.ceil(grid_onesided_length / grid_res).astype(int)
        num_of_grid_cells = 2 * num_of_onesided_grid_cells

        # init for empty grid cells
        actors_exist_default = np.zeros(shape=(num_of_grid_cells, 1))
        actors_vel_default = -params["MAX_VELOCITY"] * np.ones(shape=(num_of_grid_cells, 1))
        actors_state = np.hstack((actors_exist_default, actors_vel_default))

        for actor in self.actors_states:
            actor_exists = 1
            actor_grid_cell = np.floor(actor.s_relative_to_ego / grid_res).astype(int) + num_of_onesided_grid_cells
            if 0 <= actor_grid_cell <= num_of_grid_cells - 1:
                actors_state[actor_grid_cell] = np.array([actor_exists, actor.velocity])

        return torch.from_numpy(host_state[np.newaxis, np.newaxis, :]).float(), \
               torch.from_numpy(actors_state[np.newaxis, :]).float()
