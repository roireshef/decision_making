from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
from decision_making.src.planning.types import FS_SV, FS_SA, FS_SX, FS_DX, FS_DV, FS_DA
from gym import Space
from gym.spaces import Box
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState
from decision_making.src.rl_agent.environments.state_space.common.state_encoder import StateEncoder
from decision_making.src.rl_agent.environments.state_space.common.state_mixins import HasLonKinematicInformation, \
    HasMultipleLanesInformation, HasLaneChangeCommitOnlyInformation, HasLaneChangeFSMInformation, \
    HasFullKinematicInformation, HasLaneChangeCommitSignOnlyInformation
from decision_making.src.rl_agent.environments.state_space.common.state_mixins import ScaleNormalization
from decision_making.src.rl_agent.environments.uc_rl_map import MapAnchor


class HostEncoder(StateEncoder, metaclass=ABCMeta):
    """ Abstract class for encoding host agent information
        NOTE: Do not change public methods, implement only protected/private methodes """
    def encode(self, state: EgoCentricState, normalize: bool = True) -> np.ndarray:
        """
        The external API for calling the encoder to encode the ego information from an EgoCentricState
        :param state: the input to the encoding process
        :param normalize: a flag determining either to normalize the state output or not. default: True.
        :return: a (possibly normalized) state numpy array
        """
        encoded = self._encode(state)
        return self._normalize(encoded) if normalize else encoded

    @property
    def observation_space(self) -> Box:
        return Box(low=-np.inf, high=np.inf, shape=(self._num_host_features, 1), dtype=np.float32)

    @property
    @abstractmethod
    def _num_host_features(self):
        pass

    @abstractmethod
    def _encode(self, state: EgoCentricState) -> np.ndarray:
        """
        :param state: state to encode
        :return: encoded state array (numpy) and its mask shape (size of valid entries for all dimensions)
        """
        pass

    @abstractmethod
    def _normalize(self, encoded_state: np.ndarray) -> np.ndarray:
        pass


class TransposedHostEncoderWrapper(HostEncoder):
    """ A utility wrapper class used to wrap any host encoder and transpose its shape (for backward compatibility) """
    def __init__(self, raw_encoder: HostEncoder):
        self.raw_encoder = raw_encoder

    def encode(self, state: EgoCentricState, normalize: bool = True) -> Any:
        encoded = self.raw_encoder._encode(state)
        return self.raw_encoder._normalize(encoded).transpose() if normalize else encoded.transpose()

    @property
    def observation_space(self) -> Space:
        return Box(low=-np.inf, high=np.inf, shape=(1, self.raw_encoder._num_host_features), dtype=np.float32)

    @property
    def _num_host_features(self):
        return self.raw_encoder._num_host_features

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        raise NotImplementedError("TransposedHostEncoderWrapper._encode shouldn't be called")

    def _normalize(self, encoded_state: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TransposedHostEncoderWrapper._normalize shouldn't be called")


class SingleLaneLonKinematicsHostEncoder(ScaleNormalization, HasLonKinematicInformation, HostEncoder):
    """
    Single-lane, 1D longitudinal frenet state with normalization of station coordinate by anchor.
    has the following channels:
        1) 3 DOF longitudinal frenet coordinate [SX (relative to anchor), SV, SA]

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 anchor: MapAnchor):
        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        ScaleNormalization.__init__(self, scaling_array=self.kinematics_scale_array[:, np.newaxis])
        self.anchor = anchor

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to the anchor
        host_s = state.self_gff_anchors[self.anchor] - state.ego_state.fstate[FS_SX]
        host_v = state.ego_state.fstate[FS_SV]
        host_a = state.ego_state.fstate[FS_SA]
        return np.array([host_s, host_v, host_a], dtype=np.float32)[:, np.newaxis]


class SingleLaneFullKinematicsHostEncoder(ScaleNormalization, HasFullKinematicInformation, HostEncoder):
    """
    Single-lane, 2D full frenet state with normalization of station coordinate by anchor.
    has the following channels:
        1) 6 DOF frenet coordinate [SX (relative to anchor), SV, SA, DX, DV, DA]

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 anchor: MapAnchor):
        HasFullKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        ScaleNormalization.__init__(
            self, scaling_array=self.kinematics_scale_array.astype(np.float32)[:, np.newaxis])

        self.anchor = anchor

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to red line
        host_s = state.self_gff_anchors[self.anchor] - state.ego_state.fstate[FS_SX]
        host_v = state.ego_state.fstate[FS_SV]
        host_a = state.ego_state.fstate[FS_SA]
        host_d = state.ego_state.fstate[FS_DX]
        host_dv = state.ego_state.fstate[FS_DV]
        host_da = state.ego_state.fstate[FS_DA]
        return np.array([host_s, host_v, host_a, host_d, host_dv, host_da], dtype=np.float32)[:, np.newaxis]


class SingleLaneLonKinematicsHostEncoderWithMergeLength(ScaleNormalization, HasLonKinematicInformation, HostEncoder):
    """
    Single-lane, 1D longitudinal frenet state with normalization of station coordinate by anchor.
    has the following channels:
        1) 3 DOF longitudinal frenet coordinate [SX (relative to anchor), SV, SA]
        2) Merge zone length (a map feature)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 end_anchor: MapAnchor, start_anchor: MapAnchor):
        """ 1D kinematics with merge_zone_length """
        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        ScaleNormalization.__init__(
            self, scaling_array=self.kinematics_scale_array[[0, 1, 2, 0]][:, np.newaxis].astype(np.float32))

        self.start_anchor = start_anchor
        self.end_anchor = end_anchor

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to the anchor
        host_s = state.self_gff_anchors[self.end_anchor] - state.ego_state.fstate[FS_SX]
        host_v = state.ego_state.fstate[FS_SV]
        host_a = state.ego_state.fstate[FS_SA]
        merge_zone_length = state.self_gff_anchors[self.end_anchor] - state.self_gff_anchors[self.start_anchor]
        return np.array([host_s, host_v, host_a, merge_zone_length], dtype=np.float32)[:, np.newaxis]


class SingleLaneFullKinematicsHostEncoderWithMergeLength(ScaleNormalization, HasLonKinematicInformation, HostEncoder):
    """
    Single-lane, longitudinal frenet state (position and velocity only) with normalization of station coordinate by
    anchor. has the following channels:
        1) 2 DOF longitudinal frenet coordinate [SX (relative to anchor), SV]
        2) Merge zone length (a map feature)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 end_anchor: MapAnchor, start_anchor: MapAnchor):
        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        ScaleNormalization.__init__(self, scaling_array=np.concatenate((
                self.kinematics_scale_array,
                [station_norm_const]
        )).astype(np.float32)[:, np.newaxis])

        self.start_anchor = start_anchor
        self.end_anchor = end_anchor

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to the anchor
        host_s = state.self_gff_anchors[self.end_anchor] - state.ego_state.fstate[FS_SX]
        merge_zone_length = state.self_gff_anchors[self.end_anchor] - state.self_gff_anchors[self.start_anchor]

        return np.concatenate(([host_s],
                               state.ego_state.fstate[FS_SV:],
                               [merge_zone_length])).astype(np.float32)[:, np.newaxis]


class SingleLaneFullKinematicsLFSMHostEncoder(ScaleNormalization, HasFullKinematicInformation,
                                              HasLaneChangeFSMInformation, HostEncoder):
    """
    Single-lane, 2D full frenet state (station normalized by anchor) with Lateral Finite State Machine (LFSM) info.
    has the following channels:
        1) 6 DOF frenet coordinate [SX (relative to anchor), SV, SA, DX, DV, DA]
        2) Lateral FSM status vector (4 values; binary)
        3) Lateral FSM times vector (3 values; sec)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 anchor: MapAnchor, time_norm_const: float):

        self.anchor = anchor
        self.time_norm_const = time_norm_const

        HasLaneChangeFSMInformation.__init__(self, time_norm_const)
        HasFullKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        ScaleNormalization.__init__(
            self, scaling_array=np.concatenate((
                self.kinematics_scale_array,            # 6DOF Frenet Coordinates
                self.lane_change_scale_array            # LFSM (status+times)
            )).astype(np.float32)[:, np.newaxis])

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to red line
        relative_sx = state.self_gff_anchors[self.anchor] - state.ego_state.fstate[FS_SX]
        host_frenet = np.concatenate(([relative_sx], state.ego_state.fstate[FS_SV:]))
        lc_fsm_vector = self.get_lane_change_vector(state)

        return np.concatenate((host_frenet, lc_fsm_vector)).astype(np.float32)[:, np.newaxis]


class SingleLaneFullKinematicsLCFSMHostEncoderWithMergeLength(ScaleNormalization, HasFullKinematicInformation,
                                                              HasLaneChangeFSMInformation, HostEncoder):
    """
    Single-lane, 2D full frenet state (station normalized by anchor) with Lateral Finite State Machine (LFSM) info, and
    merge zone length. has the following channels:
        1) 6 DOF frenet coordinate [SX (relative to anchor), SV, SA, DX, DV, DA]
        2) Lateral FSM status vector (4 values; binary)
        3) Lateral FSM times vector (3 values; sec)
        4) Merge zone length (a map feature)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 time_norm_const: float, end_anchor: MapAnchor, start_anchor: MapAnchor):

        self.time_norm_const = time_norm_const
        self.start_anchor = start_anchor
        self.end_anchor = end_anchor

        HasLaneChangeFSMInformation.__init__(self, time_norm_const)
        HasFullKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        ScaleNormalization.__init__(
            self, scaling_array=np.concatenate((
                self.kinematics_scale_array,            # full kinematics
                self.lane_change_scale_array,           # LC FSM
                [station_norm_const]
            )).astype(np.float32)[:, np.newaxis])

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to red line
        relative_sx = state.self_gff_anchors[self.end_anchor] - state.ego_state.fstate[FS_SX]
        host_frenet = np.concatenate(([relative_sx], state.ego_state.fstate[FS_SV:]))

        merge_zone_length = state.self_gff_anchors[self.end_anchor] - state.self_gff_anchors[self.start_anchor]

        return np.concatenate((host_frenet,
                               self.get_lane_change_vector(state),
                               [merge_zone_length])).astype(np.float32)[:, np.newaxis]


class SingleLaneLonKinematicsLCFSMHostEncoderWithMergeLength(ScaleNormalization, HasLonKinematicInformation,
                                                             HasLaneChangeFSMInformation, HostEncoder):
    """
    Single-lane, 1D longitudinal-only frenet state (station normalized by anchor) with Lateral Finite State Machine
    (LFSM) info, and merge zone length. has the following channels:
        1) 3 DOF frenet coordinate [SX (relative to anchor), SV, SA]
        2) Lateral FSM status vector (4 values; binary)
        3) Lateral FSM times vector (3 values; sec)
        4) Merge zone length (a map feature)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 time_norm_const: float, end_anchor: MapAnchor, start_anchor: MapAnchor):

        self.time_norm_const = time_norm_const
        self.start_anchor = start_anchor
        self.end_anchor = end_anchor

        HasLaneChangeFSMInformation.__init__(self, time_norm_const)
        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        ScaleNormalization.__init__(
            self, scaling_array=np.concatenate((
                self.kinematics_scale_array,            # Lon kinematics
                self.lane_change_scale_array,           # LC FSM - Lat
                [station_norm_const]
            )).astype(np.float32)[:, np.newaxis])

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to the yield line anchor
        relative_sx = state.self_gff_anchors[self.end_anchor] - state.ego_state.fstate[FS_SX]
        host_frenet = np.concatenate(([relative_sx], state.ego_state.fstate[[FS_SV, FS_SA]]))

        merge_zone_length = state.self_gff_anchors[self.end_anchor] - state.self_gff_anchors[self.start_anchor]

        return np.concatenate((host_frenet,
                               self.get_lane_change_vector(state),
                               [merge_zone_length])).astype(np.float32)[:, np.newaxis]


class MultiLaneLonKinematicsHostEncoder(ScaleNormalization, HasLonKinematicInformation, HasMultipleLanesInformation,
                                        HasLaneChangeFSMInformation, HostEncoder):
    """
    Multi-lane, 1D longitudinal state with lane existence vector and Lateral Finite State Machine (LFSM) info.
    has the following channels:
        1) 6 DOF frenet coordinate [SX (relative to anchor), SV, SA]
        2) Adjacent Lane Existence (binary)
        3) Lateral FSM status vector (4 values; binary)
        4) Lateral FSM times vector (3 values; sec)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 lane_change_time_norm_const: float, absolute_num_lanes: int):

        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        HasMultipleLanesInformation.__init__(self, absolute_num_lanes)
        HasLaneChangeFSMInformation.__init__(self, lane_change_time_norm_const)

        # Initialize scaling array for normalization
        host_scaling_array = np.concatenate([
            # frenet coordinates
            self.kinematics_scale_array,
            # lane existence
            self.binary_lanes_scale_array,
            # lane change state
            self.lane_change_scale_array
        ]).astype(np.float32)[:, np.newaxis]
        ScaleNormalization.__init__(self, scaling_array=host_scaling_array)

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to red line
        host_frenet = state.ego_state.fstate[[FS_SX, FS_SV, FS_SA]]

        # binary vector for lane-existence to both sides, centered according to ego lane
        lane_existence = self.binarize_lane_indices(np.array(state.existing_lanes))

        # hosts lane change information
        lane_change_vector = self.get_lane_change_vector(state)

        return np.concatenate([host_frenet, lane_existence, lane_change_vector]).astype(np.float32)[:, np.newaxis]


class MultiLaneHostEncoderFZI(ScaleNormalization, HasLonKinematicInformation, HasMultipleLanesInformation, HostEncoder):
    """
    Multi-lane, 1D longitudinal state with lane existence vector.
    has the following channels:
        1) 2DOF frenet coordinate [SX (absolute), SV, SX (relative to anchor)]
        2) frenet coordinate [SX (relative to anchor)]
        3) Adjacent Lane Existence (binary)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, absolute_num_lanes: int,
                 end_anchor: MapAnchor):

        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, 0)
        HasMultipleLanesInformation.__init__(self, absolute_num_lanes)

        # Initialize scaling array for normalization
        host_scaling_array = np.concatenate([
            # frenet coordinates
            self.kinematics_scale_array[:FS_SA],
            # relative sx
            [self.station_norm_const],
            # lane existence
            self.binary_lanes_scale_array,
        ]).astype(np.float32)[:, np.newaxis]
        ScaleNormalization.__init__(self, scaling_array=host_scaling_array)

        self.end_anchor = end_anchor

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        host_frenet = state.ego_state.fstate[[FS_SX, FS_SV]]
        relative_sx = state.self_gff_anchors[self.end_anchor] - state.ego_state.fstate[FS_SX]

        # binary vector for lane-existence to both sides, centered according to ego lane
        lane_existence = self.binarize_lane_indices(np.array(state.existing_lanes))

        return np.concatenate([host_frenet, [relative_sx], lane_existence]).astype(np.float32)[:, np.newaxis]


class MultiLaneHostEncoderFZIWithAcc(ScaleNormalization, HasFullKinematicInformation, HasMultipleLanesInformation,
                                     HostEncoder):
    """
    Multi-lane, 2D full frenet state with lane existence vector.
    has the following channels:
        1) 6DOF frenet coordinate [SX (absolute), SV, SA, DX, DV, DA]
        2) frenet coordinate [SX (relative to anchor)]
        3) Adjacent Lane Existence (binary)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 absolute_num_lanes: int, end_anchor: MapAnchor):
        HasFullKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        HasMultipleLanesInformation.__init__(self, absolute_num_lanes)

        # Initialize scaling array for normalization
        host_scaling_array = np.concatenate([
            # frenet coordinates
            self.kinematics_scale_array,
            # relative sx
            [self.station_norm_const],
            # lane existence
            self.binary_lanes_scale_array,
        ]).astype(np.float32)[:, np.newaxis]
        ScaleNormalization.__init__(self, scaling_array=host_scaling_array)
        self.end_anchor = end_anchor

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to red line
        host_frenet = state.ego_state.fstate
        relative_sx = state.self_gff_anchors[self.end_anchor] - state.ego_state.fstate[FS_SX]

        # binary vector for lane-existence to both sides, centered according to ego lane
        lane_existence = self.binarize_lane_indices(np.array(state.existing_lanes))

        return np.concatenate([host_frenet, [relative_sx], lane_existence]).astype(np.float32)[:, np.newaxis]


class MultiLaneHostEncoderFZIWithAccAngGoal(ScaleNormalization, HasFullKinematicInformation, 
                                            HasMultipleLanesInformation, HostEncoder):
    """
    Multi-lane, 2D full frenet state with lane existence vector and goal context vector.
    has the following channels:
        1) 6DOF frenet coordinate [SX (absolute), SV, SA, DX, DV, DA]
        2) frenet coordinate [SX (relative to anchor)]
        3) Adjacent Lane Existence (binary)
        4) Goal context - one-hot vector indicating the goal lane, relative to current ego lane (binary)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 absolute_num_lanes: int, end_anchor: MapAnchor):
        HasFullKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        HasMultipleLanesInformation.__init__(self, absolute_num_lanes)

        # Initialize scaling array for normalization
        host_scaling_array = np.concatenate([
            # frenet coordinates
            self.kinematics_scale_array,
            # relative sx
            [self.station_norm_const],
            # lane existence
            self.binary_lanes_scale_array,
            # goal vector
            self.binary_lanes_scale_array,
        ]).astype(np.float32)[:, np.newaxis]
        ScaleNormalization.__init__(self, scaling_array=host_scaling_array)
        self.end_anchor = end_anchor

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to red line
        host_frenet = state.ego_state.fstate
        relative_sx = state.self_gff_anchors[self.end_anchor] - state.ego_state.fstate[FS_SX]

        # binary vector for lane-existence to both sides, centered according to ego lane
        lane_existence = self.binarize_lane_indices(np.array(state.existing_lanes))

        # goal one-hot vector
        binary_goal_vector = self.binarize_lane_indices(np.array(state.goal_relative_lane))

        return np.concatenate([
            host_frenet, [relative_sx], lane_existence, binary_goal_vector]).astype(np.float32)[:, np.newaxis]


class MultiLaneLonKinematicsHostEncoderWithGoalOld(ScaleNormalization, HasLonKinematicInformation,
                                                   HasMultipleLanesInformation, HasLaneChangeCommitOnlyInformation,
                                                   HostEncoder):
    """
    Multi-lane, 1D longitudinal frenet state with lane-existence vector, lane change (commit-only) information,
     and goal context vector. has the following channels:
        1) 3DOF frenet coordinate [SX (absolute), SV, SA]
        2) Adjacent Lane Existence (binary)
        3) Lane Change information (target lane, time from start)
        4) Goal context - one-hot vector indicating the goal lane, relative to current ego lane (binary)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 lane_change_time_norm_const: float, absolute_num_lanes: int):
        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        HasMultipleLanesInformation.__init__(self, absolute_num_lanes)
        HasLaneChangeCommitOnlyInformation.__init__(self, lane_change_time_norm_const)

        # Initialize scaling array for normalization
        host_scaling_array = np.concatenate([
            # frenet coordinates
            self.kinematics_scale_array,
            # lane existence
            self.binary_lanes_scale_array,
            # lane change state
            self.lane_change_scale_array,
            # goal vector
            self.binary_lanes_scale_array
        ]).astype(np.float32)[:, np.newaxis]
        ScaleNormalization.__init__(self, scaling_array=host_scaling_array)

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to red line
        host_frenet = state.ego_state.fstate[[FS_SX, FS_SV, FS_SA]]

        # binary vector for lane-existence to both sides, centered according to ego lane
        lane_existence = self.binarize_lane_indices(np.array(state.existing_lanes))

        # hosts lane change information
        lane_change_vector = self.get_lane_change_vector(state)

        # goal one-hot vector
        binary_goal_vector = self.binarize_lane_indices(np.array(state.goal_relative_lane))

        return np.concatenate([
            host_frenet, lane_existence, lane_change_vector, binary_goal_vector]).astype(np.float32)[:, np.newaxis]


class MultiLaneLonKinematicsHostEncoderWithGoal(ScaleNormalization, HasLonKinematicInformation,
                                                HasMultipleLanesInformation, HasLaneChangeFSMInformation, HostEncoder):
    """
    Multi-lane, 1D longitudinal frenet state with lane-existence vector, lateral finite state machine information,
    and goal context vector. has the following channels:
        1) 3DOF frenet coordinate [SX (absolute), SV, SA]
        2) Adjacent Lane Existence (binary)
        3) Lateral FSM status vector (4 values; binary)
        4) Lateral FSM times vector (3 values; sec)
        5) Goal context - one-hot vector indicating the goal lane, relative to current ego lane (binary)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 lane_change_time_norm_const: float, absolute_num_lanes: int):
        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        HasMultipleLanesInformation.__init__(self, absolute_num_lanes)
        HasLaneChangeFSMInformation.__init__(self, lane_change_time_norm_const)

        # Initialize scaling array for normalization
        host_scaling_array = np.concatenate([
            # frenet coordinates
            self.kinematics_scale_array,
            # lane existence
            self.binary_lanes_scale_array,
            # lane change state
            self.lane_change_scale_array,
            # goal vector
            self.binary_lanes_scale_array
        ]).astype(np.float32)[:, np.newaxis]
        ScaleNormalization.__init__(self, scaling_array=host_scaling_array)

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to red line
        host_frenet = state.ego_state.fstate[[FS_SX, FS_SV, FS_SA]]

        # binary vector for lane-existence to both sides, centered according to ego lane
        lane_existence = self.binarize_lane_indices(np.array(state.existing_lanes))

        # hosts lane change information
        lane_change_vector = self.get_lane_change_vector(state)

        # goal one-hot vector
        binary_goal_vector = self.binarize_lane_indices(np.array(state.goal_relative_lane))

        return np.concatenate([host_frenet, lane_existence, lane_change_vector,
                               binary_goal_vector]).astype(np.float32)[:, np.newaxis]


class MultiLaneFullKinematicsHostEncoderWithGoal(ScaleNormalization, HasFullKinematicInformation,
                                                 HasMultipleLanesInformation, HasLaneChangeFSMInformation, HostEncoder):
    """
    Multi-lane, 2D full frenet state with lane-existence vector, lateral finite state machine information,
    and goal context vector. has the following channels:
        1) 6DOF frenet coordinate [SX (relative to anchor), SV, SA, DX, DV, DA]
        2) Adjacent Lane Existence (binary)
        3) Lateral FSM status vector (4 values; binary)
        4) Lateral FSM times vector (3 values; sec)
        5) Goal context - one-hot vector indicating the goal lane, relative to current ego lane (binary)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 lane_change_time_norm_const: float, absolute_num_lanes: int, end_anchor: MapAnchor):
        HasFullKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        HasMultipleLanesInformation.__init__(self, absolute_num_lanes)
        HasLaneChangeFSMInformation.__init__(self, lane_change_time_norm_const)

        self.end_anchor = end_anchor

        # Initialize scaling array for normalization
        host_scaling_array = np.concatenate([
            # frenet coordinates
            self.kinematics_scale_array,
            # lane existence
            self.binary_lanes_scale_array,
            # lane change state
            self.lane_change_scale_array,
            # goal vector
            self.binary_lanes_scale_array
        ]).astype(np.float32)[:, np.newaxis]
        ScaleNormalization.__init__(self, scaling_array=host_scaling_array)

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to red line
        host_frenet = np.concatenate(([state.self_gff_anchors[self.end_anchor] - state.ego_state.fstate[FS_SX]],
                                      state.ego_state.fstate[FS_SV:]))

        # binary vector for lane-existence to both sides, centered according to ego lane
        lane_existence = self.binarize_lane_indices(np.array(state.existing_lanes))

        # hosts lane change information
        lane_change_vector = self.get_lane_change_vector(state)

        # goal one-hot vector
        binary_goal_vector = self.binarize_lane_indices(np.array(state.goal_relative_lane))

        return np.concatenate([host_frenet, lane_existence, lane_change_vector,
                               binary_goal_vector]).astype(np.float32)[:, np.newaxis]


class MultiLaneFullKinematicsNoFSMHostEncoderWithGoal(ScaleNormalization, HasFullKinematicInformation,
                                                      HasMultipleLanesInformation, 
                                                      HasLaneChangeCommitSignOnlyInformation, HostEncoder):
    """
    Multi-lane, 2D full frenet state with lane-existence vector, lateral finite state machine information,
    and goal context vector. has the following channels:
        1) 6DOF frenet coordinate [SX (relative to anchor), SV, SA, DX, DV, DA]
        2) Adjacent Lane Existence (binary)
        3) Lane Change information (target lane, time from start)
        4) Goal context - one-hot vector indicating the goal lane, relative to current ego lane (binary)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 absolute_num_lanes: int, end_anchor: MapAnchor):
        HasFullKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        HasMultipleLanesInformation.__init__(self, absolute_num_lanes)
        HasLaneChangeCommitSignOnlyInformation.__init__(self)

        self.end_anchor = end_anchor

        # Initialize scaling array for normalization
        host_scaling_array = np.concatenate([
            # frenet coordinates
            self.kinematics_scale_array,
            # lane existence
            self.binary_lanes_scale_array,
            # lane change state
            self.lane_change_scale_array,
            # goal vector
            self.binary_lanes_scale_array
        ]).astype(np.float32)[:, np.newaxis]
        ScaleNormalization.__init__(self, scaling_array=host_scaling_array)

    @property
    def _num_host_features(self):
        return len(self.scaling_array)

    def _encode(self, state: EgoCentricState) -> np.ndarray:
        # encode host: replace the host station coordinate with its distance to red line
        host_frenet = np.concatenate(([state.self_gff_anchors[self.end_anchor] - state.ego_state.fstate[FS_SX]],
                                      state.ego_state.fstate[FS_SV:]))

        # binary vector for lane-existence to both sides, centered according to ego lane
        lane_existence = self.binarize_lane_indices(np.array(state.existing_lanes))

        # hosts lane change information
        lane_change_vector = self.get_lane_change_vector(state)

        # goal one-hot vector
        binary_goal_vector = self.binarize_lane_indices(np.array(state.goal_relative_lane))

        return np.concatenate([host_frenet, lane_existence, lane_change_vector,
                               binary_goal_vector]).astype(np.float32)[:, np.newaxis]