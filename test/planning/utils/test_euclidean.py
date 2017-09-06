import numpy as np

from decision_making.src.exceptions import LongitudeOutOfRoad, OutOfSegmentFront, OutOfSegmentBack
from decision_making.src.planning.utils.geometry_utils import CartesianFrame, Euclidean
import pytest

# # TODO: change this to test Euclidean
# def test_calc_point_segment_dist_success():
#     point = np.array([1.0, 1.0])
#     p_start = np.array([0.0, 0.0])
#     p_end = np.array([2.0, 0.0])
#     sign, dist, proj = CartesianFrame.calc_point_segment_dist(point, p_start, p_end)
#     assert sign == 1.0 and dist == 1.0 and proj == 1.0
#
#     point = np.array([0.5, -0.7])
#     p_start = np.array([0.0, 0.0])
#     p_end = np.array([2.0, 0.0])
#     sign, dist, proj = CartesianFrame.calc_point_segment_dist(point, p_start, p_end)
#     assert sign == -1.0 and dist == 0.7 and proj == 0.5

def test_projectOnSegment2D_projectionFallsOnSegment_returnsValidResult():
    seg_init = np.array([0, 0])
    seg_end = np.array([2, 2])
    point = np.array([0, 2])

    expected_projection = np.array([1, 1])

    projection = Euclidean.project_on_segment_2d(point, seg_init, seg_end)
    np.testing.assert_array_almost_equal(projection, expected_projection)

def test_projectOnSegment2D_projectionFallsOnEndOfSegment_returnsValidResult():
    seg_init = np.array([0, 0])
    seg_end = np.array([2, 2])
    point = np.array([0, 4])

    expected_projection = seg_end

    projection = Euclidean.project_on_segment_2d(point, seg_init, seg_end)
    np.testing.assert_array_almost_equal(projection, expected_projection)

def test_projectOnSegment2D_projectionFallsOutOfSegment_raisesException():
    seg_init = np.array([0, 0])
    seg_end = np.array([2, 2])

    point = np.array([0, 5])

    with pytest.raises(OutOfSegmentFront,
                       message="Tried to project out-of-segment point on a segment, and it didn't throw exception"):
        projection = Euclidean.project_on_segment_2d(point, seg_init, seg_end)

    point = np.array([0, -2])

    with pytest.raises(OutOfSegmentBack,
                       message="Tried to project out-of-segment point on a segment, and it didn't throw exception"):
        projection = Euclidean.project_on_segment_2d(point, seg_init, seg_end)