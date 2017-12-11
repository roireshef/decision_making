from decision_making.src.planning.types import C_A
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.test.planning.trajectory.utils import RouteFixture
from mapping.src.transformations.geometry_utils import *


def test_cpointsToFpointsToCpoints_pointTwoWayConversion_accurate():
    ACCURACY_TH = 10 ** -3

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    cpoints = np.array([[220.0, 0.0], [150.0, 0.0],
                        [280.0, 40.0], [320.0, 60.0],
                        [370.0, 0.0]
                        ])

    frenet = FrenetSerret2DFrame(route_points, ds=1)

    fpoints = frenet.cpoints_to_fpoints(cpoints)
    new_cpoints = frenet.fpoints_to_cpoints(fpoints)

    errors = np.linalg.norm(cpoints - new_cpoints, axis=1)

    for error in errors.__iter__():
        assert error < ACCURACY_TH, 'FrenetMovingFrame point conversions aren\'t accurate enough'

    import matplotlib.pyplot as plt

    fig = plt.figure()
    p1 = fig.add_subplot(111)
    p1.plot(route_points[:, 0], route_points[:, 1], '-r')
    p1.plot(frenet.O[:, 0], frenet.O[:, 1], '-k')

    for i in range(len(cpoints)):
        p1.plot(cpoints[i, 0], cpoints[i, 1], '*b')
        p1.plot(new_cpoints[i, 0], new_cpoints[i, 1], '.r')

    print(errors)

    fig.show()
    fig.clear()


def test_ctrajectoryToFtrajectoryToCtrajectory_pointTwoWayConversion_accuratePoseAndVelocity():
    ACCURACY_TH = 10 ** -3

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    cpoints = np.array([[220.0, 0.0, np.pi/6, 20.0, 1.1, 1e-3], [150.0, 0.0, -np.pi/8, 4.0, 1.0, 1e-2],
                        [280.0, 40.0, np.pi/7, 10.0, -0.9, -5*1e-3], [320.0, 60.0, np.pi/7, 3, -0.5, -5*1e-4],
                        [370.0, 0.0, np.pi/9, 2.5, -2.0, 0.0]
                        ])

    frenet = FrenetSerret2DFrame(route_points, ds=1)

    fpoints = frenet.ctrajectory_to_ftrajectory(cpoints)
    new_cpoints = frenet.ftrajectory_to_ctrajectory(fpoints)

    # currently there is no guarantee on the accuracy of acceleration and curvature
    errors = np.linalg.norm((cpoints - new_cpoints)[:, :C_A], axis=1)

    for error in errors.__iter__():
        assert error < ACCURACY_TH, 'FrenetMovingFrame point conversions aren\'t accurate enough'
