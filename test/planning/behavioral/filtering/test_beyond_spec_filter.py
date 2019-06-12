import numpy as np
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.filtering.beyond_spec_filter import BeyondSpecBrakingFilter


class DummyBeyondSpecFilter(BeyondSpecBrakingFilter):
    def __init__(self, points):
        super(DummyBeyondSpecFilter, self).__init__()
        self._points = points

    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> [np.ndarray, np.ndarray]:
        return self._points


def test_filter_filterReturnsFalse():
    points_s = np.arange(20, 100, 1)
    points_v = np.full(points_s.shape[0], 10)
    dummy_filter = DummyBeyondSpecFilter((points_s, points_v))
    expected = [False]
    v = dummy_filter.filter(behavioral_state=None, action_specs=[ActionSpec(5, 20, 0, 0, None)])
    for (expected_value, actual_value) in zip(expected, v):
        assert expected_value == actual_value

def test_filter_filterReturnsTrue():
    points_s = np.arange(300, 400, 1)
    points_v = np.full(points_s.shape[0], 10)
    dummy_filter = DummyBeyondSpecFilter((points_s, points_v))
    expected = [True]
    v = dummy_filter.filter(behavioral_state=None, action_specs=[ActionSpec(5, 20, 0, 0, None)])
    for (expected_value, actual_value) in zip(expected, v):
        assert expected_value == actual_value
