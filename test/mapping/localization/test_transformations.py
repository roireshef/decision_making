import numpy as np
import numpy.testing as npt
import pytest
import utm
from scipy.constants import g as G

from decision_making.src.mapping.localization.coords_transformer import CoordsTransformer
from decision_making.src.mapping.localization.transformations import Transformations

from rte.ctm.src.CommonTypes import Point3D


def test_convertGeoToUtm_None_RaisesException():
    with pytest.raises(ValueError):
        Transformations.convert_geo_to_utm(None)


def test_convertGeoToUtm_List_RaisesException():
    with pytest.raises(ValueError):
        Transformations.convert_geo_to_utm([32,34])


def test_convertGeoToUtm_1dArray_RaisesException():
    with pytest.raises(ValueError):
        Transformations.convert_geo_to_utm(np.array([32,34]))


def test_convertGeoToUtm_2dArrayOfPoints_ReturnsExpectedCoordinates():
    coords = np.array([[32, 34], [34, 32]])
    points = Transformations.convert_geo_to_utm(coords)

    for i in range(len(coords)):
        points_utm = utm.from_latlon(coords[i][0], coords[i][1])
        npt.assert_almost_equal(points[i], np.array([points_utm[1], points_utm[0]]))


def test_convertCoordinatesToFrame_BadCoordinates_Exception():
    #None
    with pytest.raises(ValueError):
        Transformations.convert_coordinates_to_frame(None, 0, [31, 32])

    #Not nd array
    with pytest.raises(ValueError):
        Transformations.convert_coordinates_to_frame([32, 34], 0, [31, 32])

    #wrong size
    with pytest.raises(ValueError):
        Transformations.convert_coordinates_to_frame(np.array([32, 34, 36]).reshape((1, 1, 3)), 0, [31, 32])


def test_convertCoordinatesToFrame_ValidCoordinates_ReturnsExpectedFrameCoordinates():
    result = Transformations.convert_coordinates_to_frame(np.array([32, 34]).reshape((1,2)), 0, [31, 32])
    # 'Y' of the relative map coordinate is negative, since Map frame is NWU.
    # '34' east longitude is greater than '32', so it's to the east of the
    # map frame origin, making it negative in the NWU frame!
    npt.assert_almost_equal(result, np.array([[110841.48266016, -189925.81706858]]))


def test_TransformLocationDerivativesToAgentFrame_NEDvalues_returnsAgentvalues():
    north_val = 3
    east_val = -43
    down_val = 23

    x_val, y_val, z_val = Transformations.transform_location_derivatives_to_agent_frame(north_val, east_val, down_val)

    assert north_val == pytest.approx(x_val)
    assert east_val == pytest.approx(-y_val)
    assert down_val == pytest.approx(-z_val)


def test_TransformOrientationToAgentFrame_NEDorientations_returnsAgentOrientations():
    _run_TransformOrientationToAgentFrame(15, 3, 24)
    _run_TransformOrientationToAgentFrame(-15, -3, 24)
    _run_TransformOrientationToAgentFrame(15, -3, -24)
    _run_TransformOrientationToAgentFrame(315, -3, -24)
    _run_TransformOrientationToAgentFrame(15, -406, -24)
    _run_TransformOrientationToAgentFrame(15, -3, -424)


def test_TransformLocationToAgentFrame_NEDlocationAndZeroOrientations_returnsExpectedAgentLocation():
    imu_to_agent = CoordsTransformer().get_translation('imu', 'agent')
    _runTransformLocationToAgentFrame(0, 0, 0, imu_to_agent[0], imu_to_agent[1], -imu_to_agent[2])

def test_TransformLocationToAgentFrame_NEDlocationAndYawIs180Deg_returnsExpectedAgentLocation():
    imu_to_agent = CoordsTransformer().get_translation('imu', 'agent')
    _runTransformLocationToAgentFrame(0, 0, 180, -imu_to_agent[0], -imu_to_agent[1],
                                      -imu_to_agent[2])


def test_TransformLocationToAgentFrame_NEDlocationAndYawIs90Deg_returnsExpectedAgentLocation():
    imu_to_agent = CoordsTransformer().get_translation('imu', 'agent')
    _runTransformLocationToAgentFrame(0, 0, 90, -imu_to_agent[1], -imu_to_agent[0],
                                      -imu_to_agent[2])


def test_calculateVehicleFrameAcceleration_FlatOrientation_OnlyZAccelerationChanges():
    acc_x, acc_y, acc_z = 3, 5, -9
    result_x, result_y, result_z = Transformations.recalculate_motion_acceleration([acc_x, acc_y, acc_z], [0, 0, 0])

    assert result_x == acc_x
    assert result_y == acc_y
    assert result_z == pytest.approx(acc_z + G)


def test_calculateVehicleFrameAcceleration_RollOrientation_OnlyYAndZAccelerationChanges():
    acc_x, acc_y, acc_z = 3, 5, -9
    ori_roll = np.deg2rad(45)
    result_x, result_y, result_z = Transformations.recalculate_motion_acceleration([acc_x, acc_y, acc_z], [ori_roll, 0, 0])

    assert result_x == acc_x
    assert result_y == pytest.approx(acc_y + np.sin(ori_roll) * G)
    assert result_z == pytest.approx(acc_z + np.cos(ori_roll) * G)


def test_calculateVehicleFrameAcceleration_PitchOrientation_OnlyxAndZAccelerationChanges():
    acc_x, acc_y, acc_z = 3, 5, -9
    ori_pitch = np.deg2rad(45)
    result_x, result_y, result_z = Transformations.recalculate_motion_acceleration([acc_x, acc_y, acc_z], [0, ori_pitch, 0])

    assert result_x == pytest.approx(acc_x - np.sin(ori_pitch) * G)
    assert result_y == acc_y
    assert result_z == pytest.approx(acc_z + np.cos(ori_pitch) * G)


def test_calculateVehicleFrameAcceleration_HeadingOrientation_OnlyZAccelerationChanges():
    acc_x, acc_y, acc_z = 3, 5, -9
    ori_heading = np.deg2rad(45)
    result_x, result_y, result_z = Transformations.recalculate_motion_acceleration([acc_x, acc_y, acc_z], [0, 0, ori_heading])

    assert result_x == acc_x
    assert result_y == acc_y
    assert result_z == pytest.approx(acc_z + G)


def test_calculateMotionAcceleration_ComplexOrientation_ExpectedAcceleration():
    acc_x, acc_y, acc_z = -0.1292, 0.0687, -9.677100000000001
    ori_roll, ori_pitch, ori_heading = -0.0013419999999999999, -0.035595999999999996, 0.10251399999999998,
    result_x, result_y, result_z = Transformations.recalculate_motion_acceleration([acc_x, acc_y, acc_z],
                                                                                 [ori_roll, ori_pitch, ori_heading])

    assert result_x == pytest.approx(0.21980380015921722)
    assert result_y == pytest.approx(0.0555478164545)
    assert result_z == pytest.approx(0.123328949288)


def _run_TransformOrientationToAgentFrame(ori_x, ori_y, ori_z, is_in_rad=False):

    x_orientation, y_orientation, z_orientation = [ori_x, ori_y, ori_z] if is_in_rad else np.deg2rad([ori_x, ori_y, ori_z])

    # imu (FRD) -> agent (FLU): roll (x) remains the same, pitch (y) and roll (z) change sign
    x_val, y_val, z_val = Transformations.transform_orientation_to_agent_frame(x_orientation, y_orientation,
                                                                               z_orientation)
    expected_x = _normalize_angle(x_orientation)
    expected_y = _normalize_angle(-y_orientation)
    expected_z = _normalize_angle(-z_orientation)

    assert _normalize_angle(x_val) == pytest.approx(expected_x)
    assert _normalize_angle(y_val) == pytest.approx(expected_y)
    assert _normalize_angle(z_val) == pytest.approx(expected_z)


def _normalize_angle(rads):
    rads = rads if rads >= 0 else rads + 2 * np.pi
    rads = rads if rads <= 2*np.pi else rads - 2 * np.pi

    return rads


def _runTransformLocationToAgentFrame(roll_deg, pitch_deg, yaw_deg, imu_to_agent_dx, imu_to_agent_dy, imu_to_agent_dz):
    lat, lon, alt = 32.214029, 34.837579, 2
    roll, pitch, yaw = np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(yaw_deg),
    x_val, y_val, z_val = Transformations.transform_location_to_agent_frame(lat, lon, alt, roll, pitch, yaw)

    from rte.ctm.src import CtmService
    imu_pos_map = CtmService.get_ctm().transform_point('imu', 'map', Point3D(0.0, 0.0, 0.0))

    expected_x = imu_pos_map.x - imu_to_agent_dx
    expected_y = imu_pos_map.y - imu_to_agent_dy
    expected_z = imu_pos_map.z + imu_to_agent_dz

    assert expected_x == pytest.approx(x_val)
    assert expected_y == pytest.approx(y_val)
    assert expected_z == pytest.approx(z_val)

