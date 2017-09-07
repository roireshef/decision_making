from decision_making.src.global_constants import DIVISION_FLOATING_ACCURACY
from decision_making.src.planning.utils.geometry_utils import CartesianFrame
import numpy as np

ACCURACY_DECIMAL = np.log10(DIVISION_FLOATING_ACCURACY)

def test_convertGlobalToRelativeFrame_vecOnlyTranslation_successful():
    global_pos = np.array([2, 1, 0])
    ego_pos = np.array([1, 1, 0])
    ego_yaw = 0

    relative_pos = CartesianFrame.convert_global_to_relative_frame(global_pos, ego_pos, ego_yaw)
    expected_relative_pos = np.array([1.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(relative_pos, expected_relative_pos, decimal=ACCURACY_DECIMAL)

def test_convertGlobalToRelativeFrame_vecWithFortyFiveDegRotation_successful():
    global_pos = np.array([2, 1, 0])
    ego_pos = np.array([1, 1, 0])
    ego_yaw = np.pi / 4

    relative_pos = CartesianFrame.convert_global_to_relative_frame(global_pos, ego_pos, ego_yaw)
    expected_relative_pos = np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.0])
    np.testing.assert_array_almost_equal(relative_pos, expected_relative_pos, decimal=ACCURACY_DECIMAL)


def test_convertGlobalToRelativeFrame_matrixWithFortyFiveDegRotation_successful():
    ego_pos = np.array([1, 1, 0])
    ego_yaw = np.pi / 4
    global_pos = np.array([[2, 1, 0], ego_pos])

    relative_pos = CartesianFrame.convert_global_to_relative_frame(global_pos, ego_pos, ego_yaw)
    expected_relative_pos = np.array([[1/np.sqrt(2), -1/np.sqrt(2), 0.0], [0, 0, 0]]) #TODO: not accurate
    np.testing.assert_array_almost_equal(relative_pos, expected_relative_pos, decimal=ACCURACY_DECIMAL)


def test_convertRelativeToGlobalFrame_vecOnlyTranslation_successful():
    rel_pos = np.array([2, 1, 0])
    ego_pos = np.array([1, 1, 0])
    ego_yaw = 0

    global_pos = CartesianFrame.convert_relative_to_global_frame(rel_pos, ego_pos, ego_yaw)
    expected_global_pos = np.array([3, 2, 0])
    np.testing.assert_array_almost_equal(global_pos, expected_global_pos, decimal=ACCURACY_DECIMAL)


def test_convertRelativeToGlobalFrame_vecWithFortyFiveDegRotation_successful():
    rel_pos = np.array([0, 1, 0])
    ego_pos = np.array([1, 1, 0])
    ego_yaw = np.pi / 4

    global_pos = CartesianFrame.convert_relative_to_global_frame(rel_pos, ego_pos, ego_yaw)
    expected_global_pos = np.array([1 - 1/np.sqrt(2), 1 + 1/np.sqrt(2), 0])
    np.testing.assert_array_almost_equal(global_pos, expected_global_pos, decimal=ACCURACY_DECIMAL)


def test_convertRelativeToGlobalFrame_matrixWithFortyFiveDegRotation_successful():
    rel_pos = np.array([[0, 1, 0], [0, 0, 0]])
    ego_pos = np.array([1, 1, 0])
    ego_yaw = np.pi / 4

    global_pos = CartesianFrame.convert_relative_to_global_frame(rel_pos, ego_pos, ego_yaw)
    expected_global_pos = np.array([[1 - 1/np.sqrt(2), 1 + 1/np.sqrt(2), 0], ego_pos])
    np.testing.assert_array_almost_equal(global_pos, expected_global_pos, decimal=ACCURACY_DECIMAL)
