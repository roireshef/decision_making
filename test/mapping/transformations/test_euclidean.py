import numpy as np
import pytest

from decision_making.src.map_exceptions import OutOfSegmentFront, OutOfSegmentBack
from decision_making.src.mapping.service.map_service import MapService
from decision_making.src.mapping.transformations.geometry_utils import Euclidean
from decision_making.test.mapping.model.testable_map_fixtures import ROAD_WIDTH, MAP_INFLATION_FACTOR, navigation_fixture, simple_testable_map_api


# def test_projectOnPiecewiseLinearCurve_OnRoad_exact(simple_testable_map_api):
#     map_fixture = simple_testable_map_api
#
#     # Check that a point on the first road is exactly localized (on road curve, out of road curve)
#     seg_idxs, progress = Euclidean.project_on_piecewise_linear_curve(np.array([[150.0, 10.0], [500, 0]]),
#                                                                      map_fixture.get_road(1)._points[:, 0:2])
#
#     assert seg_idxs[0] == 0
#     assert progress[0] == 0.5
#     assert seg_idxs[1] == 1
#     assert progress[1] == 2/3
#
#     # Check that a point on the second road is exactly localized (on exact points, the second is the last point in road)
#     seg_idxs, progress = Euclidean.project_on_piecewise_linear_curve(np.array([[0, 150], [0, 30]]),
#                                                                      map_fixture.get_road(2)._points[:, 0:2])
#
#     assert seg_idxs[0] == 1
#     assert progress[0] == 1.0
#     assert seg_idxs[1] == 2
#     assert progress[1] == 1.0


def test_projectOnPiecewiseLinearCurve_OnFunnel_projectsOnEndOfFirstSegment():
    path = np.array([[0,0], [1,1], [2,2], [3,2], [4,2]])

    seg_idxs, progress = Euclidean.project_on_piecewise_linear_curve(np.array([[[0.0, 1.0], [0.0, 2.0]],[[1.5, 3.0], [0.0, 5.0]]]), path)

    expected_seg_idx = np.array([[0, 0], [1, 1]])
    expected_progress = np.array([[0.5, 1.0], [1.0, 1.0]])

    np.testing.assert_array_almost_equal(seg_idxs, expected_seg_idx)
    np.testing.assert_array_almost_equal(progress, expected_progress)

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


def test_distToSegment2D_projectionFallsBeforeSegment_returnsDistToSegmentStart():
    seg_init = np.array([0, 0])
    seg_end = np.array([2, 2])

    point = np.array([-1, 0])

    expected_distance = 1.0

    distance = Euclidean.dist_to_segment_2d(point, seg_init, seg_end)
    assert distance == expected_distance


def test_distToSegment2D_projectionFallsAfterSegment_returnsDistToSegmentEnd():
    seg_init = np.array([0, 0])
    seg_end = np.array([2, 2])

    point = np.array([5, 0])

    expected_distance = np.linalg.norm(point - seg_end)

    distance = Euclidean.dist_to_segment_2d(point, seg_init, seg_end)
    assert distance == expected_distance


def test_distToSegment2D_projectionFallsOnSegment_returnsDistToSegment():
    seg_init = np.array([0, 0])
    seg_end = np.array([2, 2])

    point = np.array([0, 2])

    expected_distance = np.sqrt(2)

    distance = Euclidean.dist_to_segment_2d(point, seg_init, seg_end)
    assert distance == expected_distance


def test_singedDistToLine_pointLeftOfLine_returnsCorrectDistance():
    seg_init = np.array([0, 0])
    seg_end = np.array([2, 2])

    point = np.array([-np.sqrt(2), 0])

    expected_distance = 1.0

    distance = Euclidean.signed_dist_to_line_2d(point, seg_init, seg_end)
    assert distance == expected_distance


def test_singedDistToLine_pointRightOfLine_returnsCorrectDistance():
    seg_init = np.array([0, 0])
    seg_end = np.array([2, 2])

    point = np.array([np.sqrt(2), 0])

    expected_distance = -1.0

    distance = Euclidean.signed_dist_to_line_2d(point, seg_init, seg_end)
    assert distance == expected_distance

