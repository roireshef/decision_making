import numpy as np
import numpy.testing as npt
import pytest

from decision_making.src.mapping.localization.coords_transformer import CoordsTransformer


def test_init_allIsWell():
    CoordsTransformer()


def test_getTransform_UnknownFrames_RaisesException():
    with pytest.raises(Exception):
        CoordsTransformer().get_transform('lidar', 'camera_front_left')


def test_getTransform_KnownFrames_ReturnsExpectedTypesAndSizes():
    a = CoordsTransformer().get_transform('lidar', 'camera_front_left_0')
    assert type(a) == np.ndarray
    assert a.shape == (4, 4)


def test_TransformOrientation_FrameAndOrientation_Expected():
    a = CoordsTransformer().transform_orientation('lidar', 'camera_front_left_0', [0, 0, 0])
    assert type(a) == np.ndarray
    assert a.shape == (1, 3)
