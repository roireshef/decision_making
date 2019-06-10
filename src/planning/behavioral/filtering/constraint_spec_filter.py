import six
from abc import ABCMeta, abstractmethod
from decision_making.src.exceptions import ConstraintFilterHaltWithValue
from decision_making.src.global_constants import MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON, FILTER_V_0_GRID, \
    FILTER_V_T_GRID
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFilter
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.kinematics_utils import BrakingDistances
from typing import Any, List
import numpy as np


@six.add_metaclass(ABCMeta)
class BeyondSpecBrakingFilter(ActionSpecFilter):
    """
    An ActionSpecFilter which implements a predefined constraint.
     The filter is defined by:
     (1) x-axis (select_points method)
     (2) the function to test on these points (_target_function)
     (3) the constraint function (_constraint_function)
     (4) the condition function between target and constraints (_condition function)

     Usage:
        extend BeyondSpecConstraintFilter class and implement the appropriate functions: (at least the four methods
        described above).
        To terminate the filter calculation use _raise_true/_raise_false  at any stage.
    """
    def __init__(self):
        self.distances = BrakingDistances.create_braking_distances()

    @abstractmethod
    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> [np.array, np.array]:
        """
         selects relevant points (or other information) based on the action_spec (e.g., select all points in the
          trajectory defined by action_spec that require slowing down due to curvature).
         This method is not restricted to returns a specific type, and can be used to pass any relevant information to
         the target and constraint functions.
        :param behavioral_state:  The behavioral_state as the context of this filtering
        :param action_spec:  The action spec in question
        :return: Any type that should be used by the _target_function and _constraint_function
        """
        pass

    def _braking_distances(self, action_spec: ActionSpec, points: np.array) -> np.ndarray:
        """
        The braking distance required by using the CALM aggressiveness level to brake from the spec velocity
        to the given points' velocity limits.
        :param action_spec:
        :param points: s coordinates and the appropriate velocity limits
        :return: braking distances from the spec velocity to the given velocities
        """
        _, slow_points_velocity_limits = points
        return self.distances[FILTER_V_0_GRID.get_index(action_spec.v),
                              FILTER_V_T_GRID.get_indices(slow_points_velocity_limits)]

    def _actual_distances(self, action_spec: ActionSpec, points: Any) -> np.ndarray:
        """
        The distance from current points to the 'slow points'
        :param action_spec:
        :param points: s coordinates and the appropriate velocity limits
        :return: distances from the spec's endpoint (spec.s) to the given points
        """
        slow_points_s, _ = points
        return slow_points_s - action_spec.s

    def _raise_false(self):
        """
        Terminates the execution of the filter with a False value.
        No need to implement this method in your subclass
        :return: None
        """
        raise ConstraintFilterHaltWithValue(False)

    def _raise_true(self):
        """
        Terminates the execution of the filter with a True value.
        No need to implement this method in your subclass
        :return:
        """
        raise ConstraintFilterHaltWithValue(True)

    @staticmethod
    def _extend_spec(spec: ActionSpec) -> ActionSpec:
        """
        Compensate for delays in super short action specs to increase filter efficiency.
        If we move with constant velocity by using very short actions, at some point we reveal that there is no ability
        to brake before a slow point, and it's too late to brake calm.
        Therefore, in case of actions shorter than 2 seconds we extend these actions by assuming constant velocity
        following the spec.
        :param spec: action spec
        :return: A list of action specs with potentially extended s
        """
        min_t = MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
        return spec if spec.t >= min_t else ActionSpec(min_t, spec.v, spec.s + (min_t - spec.t) * spec.v, spec.d, spec.recipe)

    def _get_beyond_spec_frenet_range(self, action_spec: ActionSpec, frenet_frame: GeneralizedFrenetSerretFrame) -> np.array:
        """
        Returns range of Frenet frame indices of type np.array([from_idx, till_idx]) beyond the action_spec goal
        :param action_spec:
        :param frenet_frame:
        :return: the range of indices beyond the action_spec goal
        """
        # get the worst case braking distance from spec.v to 0
        max_braking_distance = self.distances[FILTER_V_0_GRID.get_index(action_spec.v), FILTER_V_T_GRID.get_index(0)]
        max_relevant_s = min(action_spec.s + max_braking_distance, frenet_frame.s_max)
        # get the Frenet point indices near spec.s and near the worst case braking distance beyond spec.s
        return frenet_frame.get_index_on_frame_from_s(np.array([action_spec.s, max_relevant_s]))[0] + 1

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """
        The filter function
        No need to implement this method in your subclass
        :param action_specs:
        :param behavioral_state:
        :return:
        """
        mask = []
        for action_spec in action_specs:
            try:
                # select ahead points, which require braking according to their curvature
                points_under_test = self._select_points(behavioral_state, action_spec)
                # for the case when the action is very short, use extended spec, since TP output has minimal
                # trajectory length
                extended_spec = BeyondSpecBrakingFilter._extend_spec(action_spec)
                # test the ability to brake before all selected points given their curvature
                mask_value = (self._braking_distances(extended_spec, points_under_test) <
                              self._actual_distances(extended_spec, points_under_test)).all()
            except ConstraintFilterHaltWithValue as e:
                mask_value = e.value
            mask.append(mask_value)
        return mask

