import six
from abc import ABCMeta, abstractmethod
from decision_making.src.exceptions import ConstraintFilterHaltWithValue
from decision_making.src.global_constants import MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFilter
from decision_making.src.planning.types import BoolArray, FrenetTrajectories2D, CartesianExtendedTrajectories
from typing import Any, List
import numpy as np


@six.add_metaclass(ABCMeta)
class ConstraintSpecFilter(ActionSpecFilter):
    """
    An ActionSpecFilter which implements a predefined constraint.
     The filter is defined by:
     (1) x-axis (select_points method)
     (2) the function to test on these points (_target_function)
     (3) the constraint function (_constraint_function)
     (4) the condition function between target and constraints (_condition function)

     Usage:
        extend ConstraintSpecFilter class and implement the appropriate functions: (at least the four methods described
        above).
        To terminate the filter calculation use _raise_true/_raise_false  at any stage.
    """

    def __init__(self, extend_short_action_specs=True):
        """
        :param extend_short_action_specs:  Determines whether very short action should be extended (assuming constant velocity)
        """
        self._extend_short_action_specs = extend_short_action_specs

    @abstractmethod
    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> Any:
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

    @abstractmethod
    def _target_function(self, behavioral_state: BehavioralGridState,
                         action_spec: ActionSpec, points: Any) -> np.array:
        """
        The definition of the function to be tested.
        :param behavioral_state:  A behavioral grid state
        :param action_spec: the action spec which to filter
        :return: the result of the target function as an np.ndarray
        """
        pass

    @abstractmethod
    def _constraint_function(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec, points: Any) \
            -> np.array:
        """
        Defines the constraint function over points.

        :param behavioral_state:  A behavioral grid state at the context of filtering
        :param action_spec: the action spec which to filter
        :return: the result of the constraint function as an np.ndarray
        """
        pass

    @abstractmethod
    def _condition(self, target_values: np.array, constraints_values: np.array) -> bool:
        """
        The test condition to apply on the results of target and constraint values
        :param target_values: the (externally calculated) target function
        :param constraints_values: the (externally calculated) constraint values
        :return: a single boolean indicating whether this action_spec should be filtered or not
        """
        pass

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

    def _check_condition(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> bool:
        """
        Tests the condition defined by this filter
        No need to implement this method in your subclass

        :param behavioral_state:
        :param action_spec:
        :return:
        """
        points_under_test = self._select_points(behavioral_state, action_spec)
        return self._condition(self._target_function(behavioral_state, action_spec, points_under_test),
                               self._constraint_function(behavioral_state, action_spec, points_under_test))

    @staticmethod
    def extend_action_specs(action_specs: List[ActionSpec],
                            min_action_time: float = MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON ) -> List[ActionSpec]:
        """
        Compensate for delays in super short action specs to increase filter efficiency.
        If we move with constant velocity by using very short actions, at some point we reveal that there is no ability
        to brake before a slow point, and it's too late to brake calm.
        Therefore, in case of actions shorter than 2 seconds we extend these actions by assuming constant velocity
        following the spec.
        :param action_specs:
        :return: A list of action specs with potentially extended copies of short ones
        """
        return [spec if spec.t >= min_action_time else
                ActionSpec(min_action_time, spec.t_d, spec.v, spec.s + (min_action_time - spec.t) * spec.v, spec.d, spec.recipe)
                for spec in action_specs]

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState,
               ftrajectories: FrenetTrajectories2D, ctrajectories: CartesianExtendedTrajectories) -> BoolArray:
        """
        The filter function
        No need to implement this method in your subclass
        :param action_specs:
        :param behavioral_state:
        :return:
        """
        mask = []
        if self._extend_short_action_specs:
            action_specs = self.extend_action_specs(action_specs)
        for action_spec in action_specs:
            try:
                mask_value = self._check_condition(behavioral_state, action_spec)
            except ConstraintFilterHaltWithValue as e:
                mask_value = e.value
            mask.append(mask_value)
        return np.array(mask)

