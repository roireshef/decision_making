from decision_making.src.planning.trajectory.fixed_trajectory_planner import FixedSamplableTrajectory
import numpy as np

from decision_making.src.planning.trajectory.samplable_trajectory import ExtendedSamplableTrajectory


def test_sample_fixedTrajectoriesValidTimes_returnValidCoordinates():
    x = np.arange(0, 100, 1)
    y = np.arange(0, 200, 2)
    x_grad, y_grad = np.gradient(x, axis=0), np.gradient(y, axis=0)
    yaw = np.arctan2(x_grad, y_grad)
    v = np.linalg.norm(np.c_[x_grad, y_grad], axis=1)
    coordinates = np.c_[x, y, yaw, v, np.zeros(100), np.zeros(100)]

    trajectory1 = FixedSamplableTrajectory(coordinates[:40], 0)
    trajectory2 = FixedSamplableTrajectory(coordinates[39:], trajectory1.max_sample_time)

    ext_traj = ExtendedSamplableTrajectory([trajectory1, trajectory2])

    times = np.array([[0, 1.0, 2.5], [5, 6, 7]])

    assert ext_traj.sample(times)