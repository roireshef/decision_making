from decision_making.src.global_constants import LANE_MERGE_STATE_FAR_AWAY_DISTANCE, \
    LANE_MERGE_ACTION_SPACE_MAX_VELOCITY
from decision_making.src.global_constants import LANE_MERGE_STATE_OCCUPANCY_GRID_ONESIDED_LENGTH as GRID_HORIZON, \
    LANE_MERGE_STATE_OCCUPANCY_GRID_RESOLUTION as GRID_RESOLUTION
from decision_making.src.rl_agent.environments.state_space.actors.actor_grid_encoder import \
    MultiLaneActorPositionalGridEncoderV1, SingleLaneActorPositionalGridEncoderV2, \
    SingleLaneActorPositionalGridEncoderV1, ActorRelationalGridEncoderV1
from decision_making.src.rl_agent.environments.state_space.actors.actor_list_encoder import \
    MultiLaneActorListEncoderV1, MultiLaneActorListEncoderV2, SingleLaneActorListEncoderV2, MultiLaneActorListEncoderV3
from decision_making.src.rl_agent.environments.uc_rl_map import MapAnchor
from decision_making.src.rl_agent.environments.state_space.host_actors_state_encoder import HostActorsStateEncoder
from decision_making.src.rl_agent.environments.state_space.host.host_encoder import MultiLaneLonKinematicsHostEncoderWithGoal, \
    MultiLaneLonKinematicsHostEncoder, TransposedHostEncoderWrapper, SingleLaneLonKinematicsHostEncoderWithMergeLength, \
    MultiLaneFullKinematicsHostEncoderWithGoal, SingleLaneFullKinematicsLFSMHostEncoder, \
    MultiLaneLonKinematicsHostEncoderWithGoalOld, SingleLaneFullKinematicsHostEncoderWithMergeLength, \
    SingleLaneFullKinematicsLCFSMHostEncoderWithMergeLength, \
    MultiLaneHostEncoderFZI, MultiLaneHostEncoderFZIWithAcc, MultiLaneFullKinematicsNoFSMHostEncoderWithGoal, \
    MultiLaneHostEncoderFZIWithAccAngGoal, SingleLaneLonKinematicsLCFSMHostEncoderWithMergeLength, \
    SingleLaneLonKinematicsLCFSMHostEncoderWithMergeLengthAndSpeed
from decision_making.src.rl_agent.environments.state_space.host.host_encoder import SingleLaneLonKinematicsHostEncoder
from decision_making.src.rl_agent.environments.state_space.host.host_encoder import \
    SingleLaneFullKinematicsHostEncoder

ENCODER_BUILDERS = {
    # FOUR LANE LCFTR
    "4lane_lcftr_default": lambda: HostActorsStateEncoder(
        host_encoder=MultiLaneLonKinematicsHostEncoderWithGoal(station_norm_const=600, velocity_norm_const=25,
                                                               acceleration_norm_const=5.5, lane_change_time_norm_const=6,
                                                               absolute_num_lanes=4),
        actors_encoder=MultiLaneActorListEncoderV1(station_norm_const=600, velocity_norm_const=25,
                                                   acceleration_norm_const=5.5, max_actors=30)),

    "4lane_lcftr_lon_kinematics_with_goal_no_fsm": lambda: HostActorsStateEncoder(
        host_encoder=MultiLaneLonKinematicsHostEncoderWithGoalOld(station_norm_const=600, velocity_norm_const=25,
                                                                  acceleration_norm_const=5.5,
                                                                  lane_change_time_norm_const=6,
                                                                  absolute_num_lanes=4),
        actors_encoder=MultiLaneActorListEncoderV2(station_norm_const=600, velocity_norm_const=25,
                                                   acceleration_norm_const=5.5, absolute_num_lanes=4, max_actors=30)),

    "4lane_lcftr_lon_kinematics_with_goal": lambda: HostActorsStateEncoder(
        host_encoder=MultiLaneLonKinematicsHostEncoderWithGoal(station_norm_const=600, velocity_norm_const=25,
                                                               acceleration_norm_const=5.5, lane_change_time_norm_const=6,
                                                               absolute_num_lanes=4),
        actors_encoder=MultiLaneActorListEncoderV2(station_norm_const=600, velocity_norm_const=25,
                                                   acceleration_norm_const=5.5, absolute_num_lanes=4, max_actors=30)),

    "4lane_lcftr_full_kinematics_with_goal": lambda: HostActorsStateEncoder(
        host_encoder=MultiLaneFullKinematicsHostEncoderWithGoal(station_norm_const=600,
                                                                velocity_norm_const=30,
                                                                acceleration_norm_const=5.5,
                                                                lane_change_time_norm_const=6,
                                                                absolute_num_lanes=4,
                                                                end_anchor=MapAnchor.END_OF_SEGMENT),
        actors_encoder=MultiLaneActorListEncoderV2(station_norm_const=600, velocity_norm_const=25,
                                                   acceleration_norm_const=5.5, absolute_num_lanes=4, max_actors=30)),

    # TWO LANE LCFT
    "2lane_lcft_actors_list": lambda: HostActorsStateEncoder(
        host_encoder=MultiLaneLonKinematicsHostEncoder(station_norm_const=500, velocity_norm_const=25,
                                                       acceleration_norm_const=10, lane_change_time_norm_const=6,
                                                       absolute_num_lanes=2),
        actors_encoder=MultiLaneActorListEncoderV1(station_norm_const=500, velocity_norm_const=25,
                                                   acceleration_norm_const=10, max_actors=20)
    ),
    "2lane_lcft_actors_grid": lambda: HostActorsStateEncoder(
        host_encoder=TransposedHostEncoderWrapper(MultiLaneLonKinematicsHostEncoder(
            station_norm_const=500, velocity_norm_const=25, acceleration_norm_const=10,
            lane_change_time_norm_const=6, absolute_num_lanes=2)),
        actors_encoder=MultiLaneActorPositionalGridEncoderV1(station_norm_const=500, velocity_norm_const=25,
                                                             acceleration_norm_const=10, absolute_num_lanes=2,
                                                             longitudinal_resolution=GRID_RESOLUTION,
                                                             longitundinal_horizon=GRID_HORIZON),
    ),

    # NEGOTIATION MERGE
    "negotiation_merge_default": lambda: HostActorsStateEncoder(
        host_encoder=SingleLaneFullKinematicsHostEncoder(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                                         LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                         1, MapAnchor.YIELD_LINE),
        actors_encoder=SingleLaneActorPositionalGridEncoderV2(LANE_MERGE_ACTION_SPACE_MAX_VELOCITY, 1,
                                                              GRID_RESOLUTION, GRID_HORIZON)),

    # NEGOTIATION MERGE with Full (2D) Kinematics
    "negotiation_merge_actors_list": lambda: HostActorsStateEncoder(
        host_encoder=SingleLaneFullKinematicsHostEncoder(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                                         LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                         1, MapAnchor.YIELD_LINE),
        actors_encoder=SingleLaneActorListEncoderV2(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                                    LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                    5.5, max_actors=20)),
    "negotiation_merge_actors_list_w_merge_length": lambda: HostActorsStateEncoder(
        host_encoder=SingleLaneFullKinematicsHostEncoderWithMergeLength(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                                                        LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                                        1, MapAnchor.YIELD_LINE,
                                                                        MapAnchor.MERGE_BEGINNING),
        actors_encoder=SingleLaneActorListEncoderV2(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                                    LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                    5.5, max_actors=20)),
    "negotiation_merge_actors_list_fsm_w_merge_length": lambda: HostActorsStateEncoder(
        host_encoder=SingleLaneLonKinematicsLCFSMHostEncoderWithMergeLength(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                                                            LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                                            1, 6.0, MapAnchor.YIELD_LINE,
                                                                            MapAnchor.MERGE_BEGINNING),
        actors_encoder=SingleLaneActorListEncoderV2(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                                    LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                    5.5, max_actors=20)),


    # FULL LATERAL NEGOTIATION MERGE (LC FSM and 2D Kinematics)
    "full_merge_w_fsm_actors_list": lambda: HostActorsStateEncoder(
        host_encoder=SingleLaneFullKinematicsLFSMHostEncoder(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                                             LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                             1, MapAnchor.YIELD_LINE, 6.0),
        actors_encoder=SingleLaneActorListEncoderV2(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                                    LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                    5.5, max_actors=20)),

    # FULL LATERAL NEGOTIATION MERGE (LC FSM and 1D Kinematics - Longitudinal)
    # TODO: This is actually using full kinematics!!! don't use it! (left for backward compatibility)
    "negotiation_merge_actors_list_fsm_lon_kinematics_w_mzl": lambda: HostActorsStateEncoder(
        host_encoder=SingleLaneFullKinematicsLCFSMHostEncoderWithMergeLength(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                                                             LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                                             1, 6.0, MapAnchor.YIELD_LINE,
                                                                             MapAnchor.MERGE_BEGINNING),
        actors_encoder=SingleLaneActorListEncoderV2(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                                    LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                    5.5, max_actors=20)),

    "negotiation_merge_actors_list_fsm_lon_kinematics_w_mzl_fix": lambda: HostActorsStateEncoder(
        host_encoder=SingleLaneLonKinematicsLCFSMHostEncoderWithMergeLength(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                                                            LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                                            1, 6.0, MapAnchor.YIELD_LINE,
                                                                            MapAnchor.MERGE_BEGINNING),
        actors_encoder=SingleLaneActorListEncoderV2(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                                    LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                    5.5, max_actors=20)),

    "negotiation_merge_actors_list_fsm_lon_kinematics_w_mzl_and_speed": lambda: HostActorsStateEncoder(
        host_encoder=SingleLaneLonKinematicsLCFSMHostEncoderWithMergeLengthAndSpeed(
            LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
            LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
            1, 6.0, MapAnchor.YIELD_LINE,
            MapAnchor.MERGE_BEGINNING),
        actors_encoder=SingleLaneActorListEncoderV2(
            LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
            LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
            5.5, max_actors=20)),

    # LC MERGE
    "lc_merge_default": lambda: HostActorsStateEncoder(
        host_encoder=TransposedHostEncoderWrapper(SingleLaneLonKinematicsHostEncoder(
            LANE_MERGE_STATE_FAR_AWAY_DISTANCE, LANE_MERGE_ACTION_SPACE_MAX_VELOCITY, 1, MapAnchor.YIELD_LINE)),
        actors_encoder=SingleLaneActorPositionalGridEncoderV1(LANE_MERGE_ACTION_SPACE_MAX_VELOCITY, 0,
                                                              GRID_RESOLUTION, GRID_HORIZON)),

    "lc_merge_with_merge_zone_length": lambda: HostActorsStateEncoder(
        host_encoder=TransposedHostEncoderWrapper(SingleLaneLonKinematicsHostEncoderWithMergeLength(
            LANE_MERGE_STATE_FAR_AWAY_DISTANCE, LANE_MERGE_ACTION_SPACE_MAX_VELOCITY, 1,
            MapAnchor.YIELD_LINE, MapAnchor.MERGE_BEGINNING)),
        actors_encoder=SingleLaneActorPositionalGridEncoderV1(LANE_MERGE_ACTION_SPACE_MAX_VELOCITY, 0,
                                                              GRID_RESOLUTION, GRID_HORIZON)),

    # SIMPLE MERGE
    "simple_merge_default": lambda: HostActorsStateEncoder(
        host_encoder=TransposedHostEncoderWrapper(
            SingleLaneLonKinematicsHostEncoder(LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
                                               LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                               1, MapAnchor.YIELD_LINE)),
        actors_encoder=SingleLaneActorPositionalGridEncoderV1(LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                              -LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                                                              GRID_RESOLUTION, GRID_HORIZON)),

    # FZI Paper Encoding
    "fzi_research_encoder": lambda: HostActorsStateEncoder(
        host_encoder=MultiLaneHostEncoderFZI(
            station_norm_const=LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
            velocity_norm_const=LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
            absolute_num_lanes=3, end_anchor=MapAnchor.YIELD_LINE),

        actors_encoder=ActorRelationalGridEncoderV1(station_norm_const=500, velocity_norm_const=25,
                                                    acceleration_norm_const=10, lambda_lateral=2,
                                                    lambda_ahead=2, lambda_behind=1,
                                                    longitudinal_horizon=GRID_HORIZON),
    ),

    "fzi_research_encoder_with_acceleration": lambda: HostActorsStateEncoder(
        host_encoder=MultiLaneHostEncoderFZIWithAcc(
            station_norm_const=LANE_MERGE_STATE_FAR_AWAY_DISTANCE,
            velocity_norm_const=LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
            acceleration_norm_const=1,
            absolute_num_lanes=3, end_anchor=MapAnchor.YIELD_LINE),

        actors_encoder=ActorRelationalGridEncoderV1(station_norm_const=500, velocity_norm_const=25,
                                                    acceleration_norm_const=10, lambda_lateral=2,
                                                    lambda_ahead=2, lambda_behind=1,
                                                    longitudinal_horizon=GRID_HORIZON),
    ),

    "fzi_research_encoder_with_acceleration_lcftr": lambda: HostActorsStateEncoder(
        host_encoder=MultiLaneHostEncoderFZIWithAccAngGoal(
            station_norm_const=600,
            velocity_norm_const=30,
            acceleration_norm_const=5.5,
            absolute_num_lanes=4,
            end_anchor=MapAnchor.END_OF_SEGMENT),

        actors_encoder=ActorRelationalGridEncoderV1(station_norm_const=150, velocity_norm_const=30,
                                                    acceleration_norm_const=5.5, lambda_lateral=2,
                                                    lambda_ahead=2, lambda_behind=1,
                                                    longitudinal_horizon=GRID_HORIZON),
    ),

    # Volvo Paper Encoding
    "4lane_lcftr_full_kinematics_with_goal_no_fsm": lambda: HostActorsStateEncoder(
        host_encoder=MultiLaneFullKinematicsNoFSMHostEncoderWithGoal(station_norm_const=600,
                                                                     velocity_norm_const=30,
                                                                     acceleration_norm_const=5.5,
                                                                     absolute_num_lanes=4,
                                                                     end_anchor=MapAnchor.END_OF_SEGMENT),
        actors_encoder=MultiLaneActorListEncoderV3(station_norm_const=600, velocity_norm_const=30,
                                                   acceleration_norm_const=5.5, absolute_num_lanes=4, max_actors=30)),

    # Our Paper Encoding ( Highway case ) - This is with full LFSM encoding!!
    "4lane_lcftr_full_kinematics_with_goal_paper": lambda: HostActorsStateEncoder(
        host_encoder=MultiLaneFullKinematicsHostEncoderWithGoal(station_norm_const=600,
                                                                velocity_norm_const=30,
                                                                acceleration_norm_const=5.5,
                                                                lane_change_time_norm_const=6,
                                                                absolute_num_lanes=4,
                                                                end_anchor=MapAnchor.END_OF_SEGMENT),
        actors_encoder=MultiLaneActorListEncoderV3(station_norm_const=600, velocity_norm_const=30,
                                                   acceleration_norm_const=5.5, absolute_num_lanes=4, max_actors=30)),
}
