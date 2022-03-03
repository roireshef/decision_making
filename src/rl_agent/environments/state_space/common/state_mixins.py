import numpy as np
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState, LaneChangeState


class HasLonKinematicInformation(object):
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float):
        """
        Mixin for states with longitudinal kinematics
        :param station_norm_const: the constant used to normalize station (frenet s) values
        :param velocity_norm_const: the constant used to normalize velocity values
        :param acceleration_norm_const: the constant used to normalize acceleration values
        """
        self.station_norm_const = station_norm_const
        self.velocity_norm_const = velocity_norm_const
        self.acceleration_norm_const = acceleration_norm_const

    @property
    def kinematics_scale_array(self) -> np.ndarray:
        return np.array([self.station_norm_const, self.velocity_norm_const, self.acceleration_norm_const],
                        dtype=np.float32)


class HasFullKinematicInformation(object):
    def __init__(self, station_norm_const: float, velocity_norm_const: float, acceleration_norm_const: float):
        """
        Mixin for states with longitudinal and lateral kinematics. Uses no scaling for lateral kinematics!
        :param station_norm_const: the constant used to normalize station (frenet s) values
        :param velocity_norm_const: the constant used to normalize velocity values
        :param acceleration_norm_const: the constant used to normalize acceleration values
        """
        self.station_norm_const = station_norm_const
        self.velocity_norm_const = velocity_norm_const
        self.acceleration_norm_const = acceleration_norm_const

    @property
    def kinematics_scale_array(self) -> np.ndarray:
        return np.array([self.station_norm_const, self.velocity_norm_const, self.acceleration_norm_const,
                         1, 1, 1], dtype=np.float32)


class HasMultipleLanesInformation(object):
    def __init__(self, absolute_num_lanes: int):
        """
        This creates an array of indices that correspond to relative indices of lanes (rightward negative) from
        farthest right lane possible to the farthest left lane possible. For 3-lane road, this will be
        [-2, -1, 0, 1, 2] because host vehicle can be on the rightmost lane (will require [2, 1, 0] relative to host),
        and also on the leftmost lane (will require [0, -1, -2] relative to host)
        :param absolute_num_lanes: total number of lanes the scenario (in a 3-lane road, this is 3)
        """
        self.absolute_num_lanes = absolute_num_lanes
        self.relative_lanes_indices = np.arange(0, 2 * self.absolute_num_lanes - 1) - (self.absolute_num_lanes - 1)
        self.num_lane_values = len(self.relative_lanes_indices)

    @property
    def lanes_scale_factor(self):
        return self.absolute_num_lanes - 1

    @property
    def binary_lanes_scale_array(self) -> np.ndarray:
        return np.ones_like(self.relative_lanes_indices)

    def binarize_lane_indices(self, lane_idxs: np.array):
        """
        Generates a fixed-sized binary 1D numpy array, based on self.relative_lanes_indices,
        that represent whether a relative lane exists in the given list of indices or not.
        :param lane_idxs: 1D of lane indices
        :return: Fixed-sized binary vector corresponds to lanes in self.relative_lanes_indices
        """
        return np.isin(self.relative_lanes_indices, lane_idxs).astype(np.float)


class HasLaneChangeCommitOnlyInformation(object):
    def __init__(self, lane_change_time_norm_const: float):
        self.lane_change_time_norm_const = lane_change_time_norm_const

    @property
    def lane_change_scale_array(self) -> np.ndarray:
        return np.array([1, self.lane_change_time_norm_const])

    @staticmethod
    def get_lane_change_vector(state: EgoCentricState):
        """
        Lane Change information is encoded into two scalars (direction - see RelativeLane int values, and time
        from start of lane change, that can take values in the range [0, 6])
        Note that time from start for inactive lane change (originally None) is represented by 0
        """

        return np.array([state.ego_state.lane_change_state.target_relative_lane.value,
                         state.ego_state.lane_change_state.time_since_start(state.timestamp_in_sec) or 0])


class HasLaneChangeCommitSignOnlyInformation(object):
    @property
    def lane_change_scale_array(self) -> np.ndarray:
        return np.array([1])

    @staticmethod
    def get_lane_change_vector(state: EgoCentricState):
        """
        Lane Change information is encoded into two scalars (direction - see RelativeLane int values, and time
        from start of lane change, that can take values in the range [0, 6])
        Note that time from start for inactive lane change (originally None) is represented by 0
        """

        return np.array([state.ego_state.lane_change_state.target_relative_lane.value])


class HasLaneChangeFSMInformation(object):
    def __init__(self, lane_change_time_norm_const: float):
        self.lane_change_time_norm_const = lane_change_time_norm_const
        self.lc_fsm_scale_array = np.concatenate((np.ones_like(LaneChangeState.get_inactive().status_vector),
                                                  np.full_like(LaneChangeState.get_inactive().time_since_vector(0),
                                                               fill_value=self.lane_change_time_norm_const)))

    @property
    def lane_change_scale_array(self) -> np.ndarray:
        return self.lc_fsm_scale_array

    @staticmethod
    def get_lane_change_vector(state: EgoCentricState):
        """
        Lane Change information is encoded into two scalars (direction - see RelativeLane int values, and time
        from start of lane change, that can take values in the range [0, 6])
        Note that time from start for inactive lane change (originally None) is represented by 0
        """
        return np.concatenate((state.ego_state.lane_change_state.status_vector,
                               state.ego_state.lane_change_state.time_since_vector(state.timestamp_in_sec)))


class ScaleNormalization(object):
    def __init__(self, scaling_array: np.ndarray):
        """
        Mixin for scaling-based normalization in encoders
        :param scaling_array: array used to scale encoded results
        """
        assert scaling_array.dtype == np.float32, "ScaleNormalization expected float32 but got %s in %s" % \
                                                  (scaling_array.dtype, self.__class__)
        self.scaling_array = scaling_array

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        assert self._is_valid(arr), "scaling array of shape %s is not valid for scaling observation space of shape " \
                                    "%s (in %s)" % (self.scaling_array.shape, arr.shape, self)

        return arr / self.scaling_array

    def _is_valid(self, arr: np.array):
        return len(arr.shape) == len(self.scaling_array.shape) and \
               np.all(np.logical_or(np.array(self.scaling_array.shape) == 1,
                                    np.array(self.scaling_array.shape) == np.array(arr.shape)))


class ShiftAndScaleNormalization(ScaleNormalization):
    def __init__(self, shift_array: np.ndarray, scaling_array: np.ndarray):
        super(ShiftAndScaleNormalization, self).__init__(scaling_array)
        self.shift_array = shift_array

        assert np.all(self.shift_array.shape == self.scaling_array.shape), \
            "shift and scaling arrays have different shapes: %s, %s" % \
            (self.shift_array.shape, self.scaling_array.shape)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        return super()._normalize(arr - self.shift_array)