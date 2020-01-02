import numpy as np
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.filtering.constraint_spec_filter import ConstraintSpecFilter


class DummyConstraintSpecFilter(ConstraintSpecFilter):
    def __init__(self, constraint, points):
        super(DummyConstraintSpecFilter, self).__init__(False)
        self._constraint = constraint
        self._points = points

    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> np.ndarray:
        return self._points

    def _target_function(self, behavioral_state: BehavioralGridState, action_spec:
    ActionSpec, points: np.ndarray) -> np.ndarray:
        return points ** 2 + 10

    def _constraint_function(self, behavioral_state: BehavioralGridState,
                             action_spec: ActionSpec, points: np.ndarray):
        return np.full(100, self._constraint)

    def _condition(self, target_values, constraints_values) -> bool:
        return target_values < constraints_values


def test_filter_FilteringChangesInMiddleOfRange():
    constraint_value = 40
    points = np.arange(0, 100, 1)
    dummy_filter = DummyConstraintSpecFilter(constraint_value, points)
    expected = points ** 2 + 10 < constraint_value
    v = dummy_filter.filter(behavioral_state=None, action_specs=[1], ftrajectories=np.array([]), ctrajectories=np.array([]))
    for (expected_value, actual_value) in zip(expected, v[0]):
        assert expected_value == actual_value

def test_filter_filterAlwaysTrue():
    constraint_value = 100 ** 2 + 11
    points = np.arange(0, 100, 1)
    dummy_filter = DummyConstraintSpecFilter(constraint_value, points)
    expected = [True] * len(points)
    v = dummy_filter.filter(behavioral_state=None, action_specs=[1], ftrajectories=np.array([]), ctrajectories=np.array([]))
    for (expected_value, actual_value) in zip(expected, v[0]):
        assert expected_value == actual_value


def test_filter_filterAlwaysFalse():
    constraint_value = -1
    points = np.arange(0, 100, 1)
    dummy_filter = DummyConstraintSpecFilter(constraint_value, points)
    expected = [False] * len(points)
    v = dummy_filter.filter(behavioral_state=None, action_specs=[1], ftrajectories=np.array([]), ctrajectories=np.array([]))
    for (expected_value, actual_value) in zip(expected, v[0]):
        assert expected_value == actual_value


def test_raiseFalse_filterAlwaysFalse():
    class RaiseFalseDummyConstraintSpecFilter(DummyConstraintSpecFilter):
        def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> np.ndarray:
            self._raise_false()
            return super(RaiseFalseDummyConstraintSpecFilter, self)._select_points(behavioral_state, action_spec)

    constraint_value = -1
    points = np.arange(0, 100, 1)
    dummy_filter = RaiseFalseDummyConstraintSpecFilter(constraint_value, points)
    v = dummy_filter.filter(behavioral_state=None, action_specs=[1], ftrajectories=np.array([]), ctrajectories=np.array([]))
    assert not v[0]


def test_raiseTrue_filterAlwaysTrue():
    class RaiseTrueDummyConstraintSpecFilter(DummyConstraintSpecFilter):
        def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> np.ndarray:
            self._raise_true()
            return super(RaiseTrueDummyConstraintSpecFilter, self)._select_points(behavioral_state, action_spec)

    constraint_value = -1
    points = np.arange(0, 100, 1)
    dummy_filter = RaiseTrueDummyConstraintSpecFilter(constraint_value, points)
    v = dummy_filter.filter(behavioral_state=None, action_specs=[1], ftrajectories=np.array([]), ctrajectories=np.array([]))
    assert v[0]
