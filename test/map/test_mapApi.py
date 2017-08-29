from decision_making.src.map.map_api import MapAPI
import pytest
import numpy as np


class TestableMapApi(MapAPI):
    def call_shift_road_vector(self, points: np.ndarray, shift: np.ndarray) -> np.ndarray:
        return self._shift_road_vector(points, shift)

    def call_shift_road_vector_in_lat(self, points: np.ndarray, lat_shift: float) -> np.ndarray:
        return self._shift_road_vector_in_lat(points, lat_shift)


@pytest.fixture()
def testable_map_api():
    yield TestableMapApi(None, None)


def test_shiftRoadVector_simpleRoadShift1MRight_accurateShift(testable_map_api):
    points = np.array([[0, 0], [1, -1], [1, -2]])
    shift = np.array([0, -1])
    shifted_points = testable_map_api.call_shift_road_vector(points, shift)
    expected_shifted_points = np.array([[-1/np.sqrt(2), -1/np.sqrt(2)], [0, -1], [0, -2]])

    np.testing.assert_array_almost_equal(shifted_points, expected_shifted_points)

    originally_shifted_points = testable_map_api.call_shift_road_vector_in_lat(points.transpose(), -1).transpose()
    np.testing.assert_array_almost_equal(shifted_points, originally_shifted_points)
