import numpy as np
from decision_making.src.global_constants import EPS

from decision_making.src.utils.geometry_utils import CartesianFrame

ACCURACY_DECIMAL = np.log10(EPS)

def test_convertGlobalToRelativeFrame_vecOnlyTranslation_successful():
    global_pos = np.array([2, 1])
    global_yaw = 1
    ego_pos = np.array([1, 1])
    ego_yaw = 0

    relative_pos, relative_yaw = CartesianFrame.convert_global_to_relative_frame(global_pos, global_yaw,
                                                                                 ego_pos, ego_yaw)
    expected_relative_pos = np.array([1.0, 0.0])
    expected_relative_yaw = 1.0
    np.testing.assert_array_almost_equal(relative_pos, expected_relative_pos, decimal=ACCURACY_DECIMAL)
    np.testing.assert_almost_equal(relative_yaw, expected_relative_yaw, decimal=ACCURACY_DECIMAL)

def test_convertGlobalToRelativeFrame_vecWithFortyFiveDegRotation_successful():
    global_pos = np.array([2, 1])
    global_yaw = np.pi / 2
    ego_pos = np.array([1, 1])
    ego_yaw = np.pi / 4

    relative_pos, relative_yaw = CartesianFrame.convert_global_to_relative_frame(global_pos, global_yaw,
                                                                                 ego_pos, ego_yaw)
    expected_relative_pos = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
    expected_relative_yaw = np.pi / 4
    np.testing.assert_array_almost_equal(relative_pos, expected_relative_pos, decimal=ACCURACY_DECIMAL)
    np.testing.assert_almost_equal(relative_yaw, expected_relative_yaw, decimal=ACCURACY_DECIMAL)


def test_convertGlobalToRelativeFrame_matrixWithFortyFiveDegRotation_successful():
    ego_pos = np.array([1, 1])
    ego_yaw = np.pi / 4
    global_pos = np.array([[2, 1], ego_pos])
    global_yaw = np.pi / 2

    relative_pos, relative_yaw = CartesianFrame.convert_global_to_relative_frame(global_pos, global_yaw, ego_pos, ego_yaw)
    expected_relative_pos = np.array([[1/np.sqrt(2), -1/np.sqrt(2)], [0, 0]]) #TODO: not accurate
    expected_relative_yaw = np.pi / 4
    np.testing.assert_array_almost_equal(relative_pos, expected_relative_pos, decimal=ACCURACY_DECIMAL)
    np.testing.assert_almost_equal(relative_yaw, expected_relative_yaw, decimal=ACCURACY_DECIMAL)


def test_convertRelativeToGlobalFrame_vecOnlyTranslation_successful():
    rel_pos = np.array([2, 1])
    rel_yaw = 0
    ego_pos = np.array([1, 1])
    ego_yaw = 0

    global_pos, global_yaw = CartesianFrame.convert_relative_to_global_frame(rel_pos, rel_yaw, ego_pos, ego_yaw)
    expected_global_pos = np.array([3, 2])
    np.testing.assert_array_almost_equal(global_pos, expected_global_pos, decimal=ACCURACY_DECIMAL)


def test_convertRelativeToGlobalFrame_vecWithFortyFiveDegRotation_successful():
    rel_pos = np.array([0, 1])
    rel_yaw = 0
    ego_pos = np.array([1, 1])
    ego_yaw = np.pi / 4

    global_pos, global_yaw = CartesianFrame.convert_relative_to_global_frame(rel_pos, rel_yaw, ego_pos, ego_yaw)
    expected_global_pos = np.array([1 - 1/np.sqrt(2), 1 + 1/np.sqrt(2)])
    np.testing.assert_array_almost_equal(global_pos, expected_global_pos, decimal=ACCURACY_DECIMAL)


def test_convertRelativeToGlobalFrame_matrixWithFortyFiveDegRotation_successful():
    rel_pos = np.array([[0, 1], [0, 0]])
    rel_yaw = 0
    ego_pos = np.array([1, 1])
    ego_yaw = np.pi / 4

    global_pos, global_yaw = CartesianFrame.convert_relative_to_global_frame(rel_pos, rel_yaw, ego_pos, ego_yaw)
    expected_global_pos = np.array([[1 - 1/np.sqrt(2), 1 + 1/np.sqrt(2)], ego_pos])
    np.testing.assert_array_almost_equal(global_pos, expected_global_pos, decimal=ACCURACY_DECIMAL)
