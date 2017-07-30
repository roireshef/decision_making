from src.planning.utils.geometry_utils import *
from test.planning.trajectory.utils import RouteFixture


def test_fpointToCpoint_simpleConversion_accurateConversion():
    ACCURACY_TH = 0.1

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    cpoints = np.array([[220.0, 0.0], [150.0, 0.0],
                        [280.0, 40.0], [320.0, 60.0],
                        [350.0, 0.0]
                        ])

    frenet = FrenetMovingFrame(route_points)

    fpoints = np.array([frenet.cpoint_to_fpoint(cpoint) for cpoint in cpoints])
    new_cpoints = np.array([frenet.fpoint_to_cpoint(fpoint) for fpoint in fpoints])

    errors = np.linalg.norm(cpoints - new_cpoints, axis=1)

    for error in errors.__iter__():
        assert error < ACCURACY_TH, 'FrenetMovingFrame point conversions aren\'t accurate enough'

    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # p1 = fig.add_subplot(111)
    # p1.plot(route_points[:, 0], route_points[:, 1], '-r')
    # p1.plot(frenet.curve[:, 0], frenet.curve[:, 1], '-k')
    #
    # for i in range(len(cpoints)):
    #     p1.plot(cpoints[i, 0], cpoints[i, 1], '*b')
    #     p1.plot(new_cpoints[i, 0], new_cpoints[i, 1], '.r')
    #
    # print(errors)
    #
    # fig.show()
    # fig.clear()
