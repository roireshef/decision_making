import heapq
from abc import ABCMeta
from typing import Tuple

import numpy as np
from decision_making.src.planning.types import FS_SV, FS_SX, FS_SA
from gym.spaces import Box
from decision_making.src.rl_agent.environments.state_space.common.state_mixins import ScaleNormalization
from decision_making.src.rl_agent.environments.state_space.actors.actor_encoder import ActorEncoder
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState, \
    EgoCentricActorState
from decision_making.src.rl_agent.environments.state_space.common.state_mixins import HasLonKinematicInformation, \
    HasMultipleLanesInformation


class ActorListEncoder(ActorEncoder, metaclass=ABCMeta):
    def __init__(self, max_actors: int):
        """
        Abstract class for encoder that encodes actors state as an unordered list (2D)
        :param max_actors: maximum number of actors in the 2D list. If there are more than this number of actors in the
        input state, they are clipped based on longitudinal proximity to the ego agent
        """
        self.max_actors = max_actors

    @property
    def _actors_state_space(self) -> Box:
        return Box(low=-np.inf, high=np.inf, shape=(self._num_actor_channels, self.max_actors), dtype=np.float32)

    def _encode(self, state: EgoCentricState) -> Tuple[np.ndarray, np.ndarray]:
        # Initialize list (tensor) of "empty" values
        encoded_actors_states = np.full((self._num_actor_channels, self.max_actors),
                                        self._empty_channels_template(state)[:, np.newaxis])

        # Clip top <num_vehicles> closest vehicles (possibly with a heap)
        num_actors = min(len(state.actors_state), self.max_actors)
        if len(state.actors_state) <= self.max_actors:
            clipped_actors_list = state.actors_state
        else:
            # This assumes comparison between EgoCentricActorState instances is based on <s_relative_to_ego> field
            clipped_actors_list = heapq.nsmallest(self.max_actors, state.actors_state)

        # extract actor channels from actor_states and override top of list, leaving the rest of it with "empty" values
        actors_channels = np.array([self._actor_channels(actor_state, state)
                                    for actor_state in clipped_actors_list], dtype=np.float32)
        encoded_actors_states[:, :len(actors_channels)] = actors_channels.transpose()

        return encoded_actors_states, np.array([self._num_actor_channels, num_actors])


class MultiLaneActorListEncoderV1(ScaleNormalization, HasLonKinematicInformation, ActorListEncoder):
    """
    A ActorListEncoder with the following channels:
        1) Actor Longitudinal Road Coordinate (normalized)
        2) Actor Velocity (normalized and relative to ego velocity)
        3) Actor Lateral index (lane adjacency to ego lane, in road coordinate frame)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 max_actors: int = 20):
        ActorListEncoder.__init__(self, max_actors)
        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)

        # Setup normalization (scaling) vector for kinematics (no normalization for lane difference!)
        ScaleNormalization.__init__(self, scaling_array=np.concatenate([
            self.kinematics_scale_array[[FS_SX, FS_SV]], [1]])[:, np.newaxis].astype(np.float32))

    @property
    def _num_actor_channels(self) -> int:
        """ Actor channels: d(sx), d(sv), un-normalized relative lane (rightward-negative) """
        return 3

    def _empty_channels_template(self, state: EgoCentricState) -> np.ndarray:
        return np.array([-self.station_norm_const, 0, 0], dtype=np.float32)

    def _actor_channels(self, actor_state: EgoCentricActorState, state: EgoCentricState) -> np.ndarray:
        return np.array([
            actor_state.s_relative_to_ego,
            actor_state.velocity - state.ego_state.fstate[FS_SV],
            actor_state.lane_difference
        ], dtype=np.float32)


class MultiLaneActorListEncoderV2(ScaleNormalization, HasLonKinematicInformation, HasMultipleLanesInformation,
                                  ActorListEncoder):
    """
    A ActorListEncoder with the following channels:
        1) Actor Longitudinal Road Coordinate (normalized)
        2) Actor Velocity (normalized, absolute)
        3) Actor Velocity (normalized and relative to ego velocity)
        4) Actor Acceleration (normalized)
        5) Actor Lateral index (lane adjacency to ego lane, in road coordinate frame)

    See parents & mixins for arguments descriptions
    """

    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 absolute_num_lanes: int, max_actors: int = 20):
        ActorListEncoder.__init__(self, max_actors)
        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        HasMultipleLanesInformation.__init__(self, absolute_num_lanes)

        # Setup normalization (scaling) vector for kinematics and lane difference
        actors_scaling_array = np.concatenate([
            self.kinematics_scale_array[[FS_SX, FS_SV, FS_SV, FS_SA]],
            [self.lanes_scale_factor]]
        )[:, np.newaxis].astype(np.float32)
        ScaleNormalization.__init__(self, scaling_array=actors_scaling_array)

    @property
    def _num_actor_channels(self) -> int:
        """ Actor channels: sx, sv, d(sv), sa, relative lane (rightward-negative) """
        return 5

    def _empty_channels_template(self, state: EgoCentricState) -> np.ndarray:
        return np.array([-self.station_norm_const, 0, -state.ego_state.fstate[FS_SV], 0, 0], dtype=np.float32)

    def _actor_channels(self, actor_state: EgoCentricActorState, state: EgoCentricState) -> np.ndarray:
        return np.array([
            actor_state.s_relative_to_ego,
            actor_state.velocity,
            actor_state.velocity - state.ego_state.fstate[FS_SV],
            actor_state.acceleration,
            actor_state.lane_difference
        ], dtype=np.float32)


class MultiLaneActorListEncoderV3(ScaleNormalization, HasLonKinematicInformation, HasMultipleLanesInformation,
                                  ActorListEncoder):
    """
    A ActorListEncoder with the following channels:
        1) Actor Longitudinal Road Coordinate (normalized)
        2) Actor Velocity (normalized and relative to ego velocity)
        3) Actor Acceleration (normalized)
        4) Actor Lateral index (lane adjacency to ego lane, in road coordinate frame)

    See parents & mixins for arguments descriptions
    """

    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 absolute_num_lanes: int, max_actors: int = 20):
        ActorListEncoder.__init__(self, max_actors)
        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        HasMultipleLanesInformation.__init__(self, absolute_num_lanes)

        # Setup normalization (scaling) vector for kinematics and lane difference
        ScaleNormalization.__init__(self, scaling_array=np.concatenate([
            self.kinematics_scale_array[[FS_SX, FS_SV, FS_SA]],
            [self.lanes_scale_factor]])[:, np.newaxis].astype(np.float32))

    @property
    def _num_actor_channels(self) -> int:
        """ Actor channels: sx, sv, d(sv), sa, relative lane (rightward-negative) """
        return 4

    def _empty_channels_template(self, state: EgoCentricState) -> np.ndarray:
        return np.array([-self.station_norm_const, -state.ego_state.fstate[FS_SV], 0, 0], dtype=np.float32)

    def _actor_channels(self, actor_state: EgoCentricActorState, state: EgoCentricState) -> np.ndarray:
        return np.array([
            actor_state.s_relative_to_ego,
            actor_state.velocity - state.ego_state.fstate[FS_SV],
            actor_state.acceleration,
            actor_state.lane_difference
        ], dtype=np.float32)


class SingleLaneActorListEncoder(ScaleNormalization, HasLonKinematicInformation, ActorListEncoder):
    """
    A ActorListEncoder for single lane actors state with the following channels:
        1) Actor Longitudinal Road Coordinate (normalized)
        2) Actor Velocity (normalized and relative to ego velocity)

    See parents & mixins for arguments descriptions
    """

    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 max_actors: int = 20):
        ActorListEncoder.__init__(self, max_actors)
        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        ScaleNormalization.__init__(
            self, scaling_array=self.kinematics_scale_array[[FS_SX, FS_SV]][np.newaxis, :].astype(np.float32))

    @property
    def _num_actor_channels(self) -> int:
        """ Actor channels: sx, d(sv) """
        return 2

    def _empty_channels_template(self, state: EgoCentricState) -> np.ndarray:
        return np.array([-self.station_norm_const, 0], dtype=np.float32)

    def _actor_channels(self, actor_state: EgoCentricActorState, state: EgoCentricState) -> np.ndarray:
        return np.array([actor_state.s_relative_to_ego, actor_state.velocity - state.ego_state.fstate[FS_SV]], dtype=np.float32)


class SingleLaneActorListEncoderV2(ScaleNormalization, HasLonKinematicInformation, ActorListEncoder):
    """
    A ActorListEncoder for single lane actors state with the following channels:
        1) Actor Longitudinal Road Coordinate (normalized)
        2) Actor Velocity (normalized and relative to ego velocity)
        3) Actor Velocity (normalized, absolute)
        4) Actor Acceleration (normalized)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 max_actors: int = 20):
        ActorListEncoder.__init__(self, max_actors)
        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)
        ScaleNormalization.__init__(
            self, scaling_array=self.kinematics_scale_array[[FS_SX, FS_SV, FS_SV, FS_SA]][:, np.newaxis])

    @property
    def _num_actor_channels(self) -> int:
        """ Actor channels: sx, d(sv), sv, sa. Note the default velocity for non-actor """
        return 4

    def _empty_channels_template(self, state: EgoCentricState) -> np.ndarray:
        return np.array([-self.station_norm_const, -self.kinematics_scale_array[FS_SV] - state.ego_state.fstate[FS_SV],
                         -self.kinematics_scale_array[FS_SV], 0], dtype=np.float32)

    def _actor_channels(self, actor_state: EgoCentricActorState, state: EgoCentricState) -> np.ndarray:
        return np.array([actor_state.s_relative_to_ego, actor_state.velocity - state.ego_state.fstate[FS_SV],
                         actor_state.velocity, actor_state.acceleration], dtype=np.float32)