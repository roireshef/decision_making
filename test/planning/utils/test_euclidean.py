import numpy as np
from decision_making.src.planning.utils.geometry_utils import CartesianFrame

# TODO: change this to test Euclidean
def test_calc_point_segment_dist_success():
    point = np.array([1.0, 1.0])
    p_start = np.array([0.0, 0.0])
    p_end = np.array([2.0, 0.0])
    sign, dist, proj = CartesianFrame.calc_point_segment_dist(point, p_start, p_end)
    assert sign == 1.0 and dist == 1.0 and proj == 1.0

    point = np.array([0.5, -0.7])
    p_start = np.array([0.0, 0.0])
    p_end = np.array([2.0, 0.0])
    sign, dist, proj = CartesianFrame.calc_point_segment_dist(point, p_start, p_end)
    assert sign == -1.0 and dist == 0.7 and proj == 0.5

