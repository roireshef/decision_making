from decision_making.src.planning.utils.frenet_moving_frame import FrenetMovingFrame
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.test.planning.trajectory.utils import RouteFixture
from mapping.src.transformations.geometry_utils import *


def test_frenetSerretFrame_pointTwoWayConversion_accurate():
    ACCURACY_TH = 10 ** -6

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
