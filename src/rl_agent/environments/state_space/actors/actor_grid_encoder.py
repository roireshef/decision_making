from abc import ABCMeta
from abc import abstractmethod
from collections import defaultdict
from traceback import format_exc
from typing import Tuple, List

import numpy as np
from decision_making.src.global_constants import LANE_MERGE_STATE_OCCUPANCY_GRID_ONESIDED_LENGTH as GRID_HORIZON, \
    LANE_MERGE_STATE_OCCUPANCY_GRID_RESOLUTION as GRID_RESOLUTION
from decision_making.src.planning.types import FS_SV
from gym.spaces import Box
from decision_making.src.rl_agent.environments.state_space.actors.actor_encoder import ActorEncoder
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState, \
    EgoCentricActorState
from decision_making.src.rl_agent.environments.state_space.common.state_mixins import \
    HasLonKinematicInformation, HasMultipleLanesInformation
from decision_making.src.rl_agent.environments.state_space.common.state_mixins import ScaleNormalization


class ActorGridEncoder(ActorEncoder, metaclass=ABCMeta):
    ACTOR_MISSING = 0
    ACTOR_EXISTS = 1

    def __init__(self, num_lateral_cells: int, num_longitudinal_cells: int, longitundinal_horizon: float):
        """
        Abstract class for encoder that projects vehicles to multi-channel grid cells
        :param num_lateral_cells: total number of cells on the lateral dimension (usually lanes)
        :param num_longitudinal_cells: total number of cells on the longitudinal dimension
        :param longitundinal_horizon: longitudinal horizon towards front and back (one-sided, [m]) for cropping actors
        """
        ActorEncoder.__init__(self)
        self.num_longitudinal_cells = num_longitudinal_cells
        self.num_lateral_cells = num_lateral_cells
        self.longitudinal_horizon = longitundinal_horizon

    @abstractmethod
    def _actors_cells(self, actor_states: List[EgoCentricActorState], state: EgoCentricState) -> \
            Tuple[np.ndarray, List[bool]]:
        """
        Implement a method that takes an actor's states and returns a 2D array of its cell coordinates
        :param actor_states: list of actor to find cells for
        :param state: the full state used to extract any other relevant information (baseline reductions, etc.)
        :return: 2D numpy lat-lon matrix coordinates for actors, mask list (if actor is in previous output)
        """
        pass

    def _filter_actors(self, state: EgoCentricState) -> List[EgoCentricActorState]:
        """ returns a filtered list of EgoCentricActorState instances based on a given state """
        return [actor for actor in state.actors_state if abs(actor.s_relative_to_ego) <= self.longitudinal_horizon]

    def _encode(self, state: EgoCentricState) -> Tuple[np.ndarray, np.ndarray]:
        # create a 3d array of default values of shape: Channels x Lanes x Long_Cells
        empty_actor_template = self._empty_channels_template(state)
        encoded_actors_states = np.tile(empty_actor_template[..., None, None],
                                        (self.num_lateral_cells, self.num_longitudinal_cells))

        # clip actors to only those within the horizon (backward and forward)
        filtered_actors = self._filter_actors(state)

        # In case of no actors, return the default-value-initialized array
        if len(filtered_actors) == 0:
            return encoded_actors_states, np.array(encoded_actors_states.shape)

        # Construct a dense 2D representation (actor x features) of the data for all relevant actors and their
        # cell-coordinate information
        cell_coordinates, mask = self._actors_cells(filtered_actors, state)
        actor_features = np.array([self._actor_channels(actor, state)
                                   for actor, mask in zip(filtered_actors, mask) if mask])

        # Project actor-wise information on the spatial 3D grid, taking into account correct longitudinal cell
        # assignment and relative lane assignment. Projection is using a channel-major (instead of actor-major) version
        # of <actor_features>
        try:
            encoded_actors_states[:, cell_coordinates[mask, 0], cell_coordinates[mask, 1]] = actor_features.transpose()
        except Exception as e:
            raise Exception("Error in HostActorsStateEncoder with state: %s, local variables: %s" %
                            (state, locals()), format_exc(), str(e))

        return encoded_actors_states, np.array(encoded_actors_states.shape)


class MultiLaneActorPositionalGridEncoder(ActorGridEncoder, HasMultipleLanesInformation, metaclass=ABCMeta):
    def __init__(self, absolute_num_lanes: int, longitudinal_resolution: float, longitundinal_horizon: float):
        """
        Encodes actors states as a 3D grid based on lanes on the latitude and constant sized cells on the longitude.
        This is similar to the grid used in Convolutional Social Pooling for Vehicle Trajectory Prediction
        (https://arxiv.org/abs/1805.06771) paper. The result is a 3D numpy array of shape (C, L, G), where:
        C is the Channels dimension (actor features)
        L is the Lane assignment dimension (lateral)
        G is the 1D grid size dimension (longitude cell)
        :param absolute_num_lanes: the total number of lanes on the road (see HasMultipleLanesInformation for details)
        :param longitudinal_resolution:
        :param longitundinal_horizon:
        """
        HasMultipleLanesInformation.__init__(self, absolute_num_lanes=absolute_num_lanes),
        super().__init__(longitundinal_horizon=longitundinal_horizon, num_lateral_cells=self.num_lane_values,
                         num_longitudinal_cells=2*np.ceil(longitundinal_horizon / longitudinal_resolution).astype(int))
        self.longitudinal_resolution = longitudinal_resolution

    @property
    def _actors_state_space(self) -> Box:
        return Box(low=-np.inf, high=np.inf, dtype=np.float32,
                   shape=(self._num_actor_channels, self.num_lateral_cells, self.num_longitudinal_cells))

    def _actors_cells(self, actor_states: List[EgoCentricActorState], state: EgoCentricState) -> \
            Tuple[np.ndarray, List[bool]]:
        """
        Implement a method that takes an actor's states and returns a 2D array of its channel-values
        :param actor_states: list of actor to find cells for
        :param state: the full state used to extract any other relevant information (baseline reductions, etc.)
        :return: 2D numpy lat-lon matrix coordinates for actors, mask list (if actor is in previous output)
        """
        # Extract grid positions from actors_states
        coords = np.array([[actor.lane_difference, actor.s_relative_to_ego] for actor in actor_states])

        coords[:, 1] = np.floor(coords[:, 1] / self.longitudinal_resolution) + self.num_longitudinal_cells / 2
        coords[:, 0] = coords[:, 0] + (self.absolute_num_lanes - 1)

        # all actors are projected
        return coords.astype(int), [True] * len(actor_states)


class MultiLaneActorPositionalGridEncoderV1(ScaleNormalization, HasLonKinematicInformation,
                                            MultiLaneActorPositionalGridEncoder):
    """
    A MultiLaneActorPositionalGridEncoder with the following channels:
        1) Actor existence
        2) Actor Velocity (normalized)

    See parents & mixins for arguments descriptions
    """

    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 absolute_num_lanes: int, longitudinal_resolution: float, longitundinal_horizon: float):
        MultiLaneActorPositionalGridEncoder.__init__(self, absolute_num_lanes=absolute_num_lanes,
                                                     longitudinal_resolution=longitudinal_resolution,
                                                     longitundinal_horizon=longitundinal_horizon)
        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)

        scaling_array = np.array([1, self.velocity_norm_const])[:, np.newaxis, np.newaxis].astype(np.float32)
        ScaleNormalization.__init__(self, scaling_array=scaling_array)

    @property
    def _num_actor_channels(self) -> int:
        """ Actor channels: exists (binary), velocity """
        return 2

    def _empty_channels_template(self, state: EgoCentricState) -> np.ndarray:
        return np.array([self.ACTOR_MISSING, 0], dtype=np.float32)

    def _actor_channels(self, actor_state: EgoCentricActorState, state: EgoCentricState) -> np.ndarray:
        return np.array([self.ACTOR_EXISTS, actor_state.velocity], dtype=np.float32)


class SingleLanePositionalGridEncoder(MultiLaneActorPositionalGridEncoder, metaclass=ABCMeta):
    """
    A utility class that inherits MultiLaneActorPositionalGridEncoder and used to project multi lane information on a
    single lane. Encodes actors states on any lane into a 2D numpy array of shape (C, G) representing longitudes on a
    single lane where C is the Channels dimension (actor features) and G is the 1D longitudinal grid size dimension
    """

    def __init__(self, longitudinal_resolution: float, longitundinal_horizon: float):
        super().__init__(absolute_num_lanes=1, longitudinal_resolution=longitudinal_resolution,
                         longitundinal_horizon=longitundinal_horizon)

    @property
    def _actors_state_space(self) -> Box:
        return Box(low=-np.inf, high=np.inf, dtype=np.float32,
                   shape=(self._num_actor_channels, self.num_longitudinal_cells))

    def _encode(self, state: EgoCentricState) -> Tuple[np.ndarray, np.ndarray]:
        encoded_actors_states, mask_shape = super()._encode(state)  # Initializes tensor of empty cells
        encoded_actors_states = encoded_actors_states[:, 0, :]  # squashes lane dimension
        return encoded_actors_states, np.array(encoded_actors_states.shape)

    def _actors_cells(self, actor_states: List[EgoCentricActorState], state: EgoCentricState) -> \
            Tuple[np.ndarray, List[bool]]:
        """
        Implement a method that takes an actor's states and returns a 2D array of its channel-values
        :param actor_states: list of actor to find cells for
        :param state: the full state used to extract any other relevant information (baseline reductions, etc.)
        :return: 2D numpy lat-lon matrix coordinates for actors, mask list (if actor is in previous output)
        """
        cell_coordinates, mask = super()._actors_cells(actor_states, state)
        cell_coordinates[:, 0] = 0  # project all actors on the single lane
        return cell_coordinates, mask


class SingleLaneActorPositionalGridEncoderV1(ScaleNormalization, SingleLanePositionalGridEncoder):
    """
    A SingleLanePositionalGridEncoder with the following channels:
        1) Actor existence
        2) Actor Velocity (normalized)

    See parents & mixins for arguments descriptions
    """
    def __init__(self, velocity_norm_const: float, missing_actor_vel: float,
                 grid_resolution: float = GRID_RESOLUTION,
                 grid_horizon: float = GRID_HORIZON):
        SingleLanePositionalGridEncoder.__init__(self, grid_resolution, grid_horizon)
        self.missing_actor_vel = missing_actor_vel
        self.velocity_norm_const = velocity_norm_const

        scaling_array = np.array([self.ACTOR_EXISTS, velocity_norm_const])
        ScaleNormalization.__init__(self, scaling_array=scaling_array[:, np.newaxis].astype(np.float32))

    @property
    def _num_actor_channels(self) -> int:
        """ Actor channels: exists (binary), velocity """
        return 2

    def _empty_channels_template(self, state: EgoCentricState) -> np.ndarray:
        return np.array([self.ACTOR_MISSING, self.missing_actor_vel], dtype=np.float32)

    def _actor_channels(self, actor_state: EgoCentricActorState, state: EgoCentricState) -> np.ndarray:
        return np.array([self.ACTOR_EXISTS, actor_state.velocity], dtype=np.float32)


class SingleLaneActorPositionalGridEncoderV2(ScaleNormalization, SingleLanePositionalGridEncoder):
    """
    A SingleLanePositionalGridEncoder with the following channels:
        1) Actor Existence (binary)
        2) Actor Velocity (normalized)
        3) Actor Acceleration (normalized)

    See parents & mixins for arguments descriptions
    """

    MISSING_ACCELERATION = 0

    def __init__(self, velocity_norm_const: float, acceleration_norm_const: float,
                 grid_resolution: float = GRID_RESOLUTION, grid_horizon: float = GRID_HORIZON):
        SingleLanePositionalGridEncoder.__init__(self, grid_resolution, grid_horizon)
        self.velocity_norm_const = velocity_norm_const

        scaling_array = np.array([self.ACTOR_EXISTS, velocity_norm_const, acceleration_norm_const])
        ScaleNormalization.__init__(self, scaling_array=scaling_array[:, np.newaxis].astype(np.float32))

    @property
    def _num_actor_channels(self) -> int:
        """ Actor channels: exists (binary), velocity, acceleration """
        return 3

    def _empty_channels_template(self, state: EgoCentricState) -> np.ndarray:
        return np.array([self.ACTOR_MISSING, -self.velocity_norm_const, self.MISSING_ACCELERATION],
                        dtype=np.float32)

    def _actor_channels(self, actor_state: EgoCentricActorState, state: EgoCentricState) -> np.ndarray:
        return np.array([self.ACTOR_EXISTS, actor_state.velocity, actor_state.acceleration], dtype=np.float32)


class ActorRelationalGridEncoder(ActorGridEncoder, metaclass=ABCMeta):
    def __init__(self, lambda_lateral: int, lambda_ahead: int, lambda_behind: int, longitudinal_horizon: float):
        """
        Encoder that encodes actors information into a 2D/3D grid based on proximity (directional) cells
        :param lambda_lateral: number of lanes to include for each side (in addition to host vehicle lane)
        :param lambda_ahead: number of vehicles ahead to include in each lane (sorted by proximity to host vehicle)
        :param lambda_behind: number of vehicles behind to include in each lane (sorted by proximity to host vehicle)
        :param longitudinal_horizon: longitudinal perception horizon in [m] (used to ignore vehicles far away)
        """
        super().__init__(num_lateral_cells=1 + 2 * lambda_lateral,
                         num_longitudinal_cells=(lambda_behind + 1 + lambda_ahead),
                         longitundinal_horizon=longitudinal_horizon)
        self.longitundinal_horizon = longitudinal_horizon
        self.lambda_behind = lambda_behind
        self.lambda_ahead = lambda_ahead
        self.lambda_lateral = lambda_lateral

    @property
    def _actors_state_space(self) -> Box:
        return Box(low=-np.inf, high=np.inf, dtype=np.float32,
                   shape=(self._num_actor_channels, self.num_lateral_cells, self.num_longitudinal_cells))

    def _filter_actors(self, state: EgoCentricState):
        relevant_lane_diffs = set(range(-self.lambda_lateral, self.lambda_lateral + 1))
        return [actor for actor in state.actors_state
                if abs(actor.s_relative_to_ego) <= self.longitudinal_horizon and
                actor.lane_difference in relevant_lane_diffs]

    def _actors_cells(self, actor_states: List[EgoCentricActorState], state: EgoCentricState) -> \
            Tuple[np.ndarray, List[bool]]:
        grid_occupancy = defaultdict(bool)
        ego_centered_cell_coords = np.full((len(actor_states), 2), np.nan, dtype=int)
        mask = [False] * len(actor_states)

        for idx, actor_state in sorted(enumerate(actor_states), key=lambda t: abs(t[1].s_relative_to_ego)):
            # if actor intersects with host vehicle on the longitudinal dimension, start looking at longitudinal cell 0.
            # Otherwise, jump to -1 or +1 based on directionality
            if abs(actor_state.s_relative_to_ego) <= (state.ego_state.size.length + actor_state.size.length) / 2:
                lon_diff = 0
            else:
                lon_diff = int(np.sign(actor_state.s_relative_to_ego))

            # continue looking for empty cell at that same lane towards the direction of actor relative to host
            while grid_occupancy[(actor_state.lane_difference, lon_diff)]:
                lon_diff += int(np.sign(actor_state.s_relative_to_ego))

            # if an empty cell is within "lambda" grid sizes, update coordinates and mask for this actor
            if -self.lambda_behind <= lon_diff <= self.lambda_ahead:
                grid_occupancy[(actor_state.lane_difference, lon_diff)] = True
                ego_centered_cell_coords[idx, :] = np.array([actor_state.lane_difference, lon_diff], dtype=int)
                mask[idx] = True

        # ego_centered_cell_coords is ego centric (behind/right have negative coordinates), shift everything to start
        # from zero so we can use those coordinates to populate the grid
        return ego_centered_cell_coords + np.array([self.lambda_lateral, self.lambda_behind]), mask


class ActorRelationalGridEncoderV1(ScaleNormalization, HasLonKinematicInformation, ActorRelationalGridEncoder):
    """
    An ActorRelationalGridEncoder with the following channels:
        1) Actor Existence (binary)
        2) Actor Longitudinal Road Coordinate (normalized)
        3) Actor Velocity (normalized and relative to ego velocity)
        4) Actor Acceleration (normalized)

    See parents & mixins for arguments descriptions
    """

    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float,
                 lambda_lateral: int, lambda_ahead: int, lambda_behind: int, longitudinal_horizon: float):
        ActorRelationalGridEncoder.__init__(self, lambda_lateral=lambda_lateral,
                                            lambda_ahead=lambda_ahead,
                                            lambda_behind=lambda_behind,
                                            longitudinal_horizon=longitudinal_horizon)
        HasLonKinematicInformation.__init__(self, station_norm_const, velocity_norm_const, acceleration_norm_const)

        scaling_array = np.array([self.ACTOR_EXISTS, self.station_norm_const, self.velocity_norm_const,
                                  self.acceleration_norm_const])[:, np.newaxis, np.newaxis].astype(np.float32)
        ScaleNormalization.__init__(self, scaling_array=scaling_array)

    @property
    def _num_actor_channels(self) -> int:
        """ Actor channels: exists (binary), velocity """
        return 4

    def _empty_channels_template(self, state: EgoCentricState) -> np.ndarray:
        return np.array([self.ACTOR_MISSING, 0, 0, 0], dtype=np.float32)

    def _actor_channels(self, actor_state: EgoCentricActorState, state: EgoCentricState) -> np.ndarray:
        return np.array([self.ACTOR_EXISTS,
                         actor_state.s_relative_to_ego,
                         actor_state.velocity - state.ego_state.fstate[FS_SV],
                         actor_state.acceleration], dtype=np.float32)
