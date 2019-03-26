from decision_making.src.planning.trajectory.samplable_trajectory import CombinedSamplableTrajectory
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D

from decision_making.test.planning.trajectory.utils import RouteFixture
import numpy as np

def test_sample_twoTrajectories_returnCorrectValues():
    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)

    #WerlingPlanner._solve_1d_poly(np.array([[10, 0, 0, 50, 5, 0]]), 5, QuinticPoly1D)
    poly_s = np.array([0.0528, -0.6800000000000002, 2.4, 0.0, -3.552713678800501e-15, 10.0])
    poly_d = np.array([0, 0, 0, 0, 0.1, 1])

    traj1 = SamplableWerlingTrajectory(0, 5, 5, full_frenet, poly_s, poly_d)


    #WerlingPlanner._solve_1d_poly(np.array([[50, 5, 0, 100, 0, 0]]), 3, QuinticPoly1D)
    poly_s = np.array([1.0493827160493827, -7.777777777777777, 15.185185185185176, 0, 5, 50])
    poly_d = np.array([0, 0, 0, 0, -0.1, 1.5])

    traj2 = SamplableWerlingTrajectory(5, 3, 3, full_frenet, poly_s, poly_d)

    combined_traj = CombinedSamplableTrajectory([traj1, traj2])

    times = np.linspace(0, 8, 100)
    times_traj1 = times[times<5]
    times_traj2 = times[times>=5]

    assert np.array_equal(combined_traj.sample(times), np.concatenate((traj1.sample(times_traj1), traj2.sample(times_traj2))))
