import numpy as np
from decision_making.src.utils.geometry_utils import CartesianFrame
import matplotlib.pyplot as plt


def test_3d_linear_interpolation_NoError():
    """
    Test 3d curve interpolation
    :return:
    """
    points = np.array([
        [0, 0, 0],
        [0, 9, 1],
        [1, 10, 2],
        [10, 10, 1],
    ])
    points[:, (2, 1)] = points[:, (1, 2)]
    resampled_points = CartesianFrame.resample_curve(curve=points, step_size=1, num_dimensions=3)[1]
    expected_num_points = 20
    assert len(resampled_points) == expected_num_points, "got {} points after interpolation, expecting {}".format(len(resampled_points), expected_num_points)
    if __name__ == '__main__':
        fig = plt.figure()

        ax = fig.add_subplot(1, 2, 1)
        plt.plot(points[:, 0], points[:, 1], 'r-*')
        plt.plot(resampled_points[:, 0], resampled_points[:, 1], 'g*')
        ax.set_aspect('equal', adjustable='box')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(points[:, 0], points[:, 2], 'r-*')
        plt.plot(resampled_points[:, 0], resampled_points[:, 2], 'g*')
        ax.set_aspect('equal', adjustable='box')
        plt.show()
