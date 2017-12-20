from decision_making.src.planning.types import C_A
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.test.planning.trajectory.utils import RouteFixture
from decision_making.src.planning.types import CartesianPoint2D
import numpy as np

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

    ## FOR DEBUG PURPOSES
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # p1 = fig.add_subplot(111)
    # p1.plot(route_points[:, 0], route_points[:, 1], '-r')
    # p1.plot(frenet.O[:, 0], frenet.O[:, 1], '-k')
    #
    # for i in range(len(cpoints)):
    #     p1.plot(cpoints[i, 0], cpoints[i, 1], '*b')
    #     p1.plot(new_cpoints[i, 0], new_cpoints[i, 1], '.r')
    #
    # print(errors)
    #
    # fig.show()
    # fig.clear()


def test_ctrajectoryToFtrajectoryToCtrajectory_pointTwoWayConversion_accuratePoseAndVelocity():
    ACCURACY_TH = 10 ** -3

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    cpoints = np.array([[150.0, 0.0, -np.pi/8, 10.0, 1.0, 1e-2], [250.0, 0.0, np.pi/6, 20.0, 1.1, -1e-2],
                        [280.0, 40.0, np.pi/7, 10.0, -0.9, -5*1e-2], [320.0, 60.0, np.pi/7, 3, -0.5, -5*1e-2],
                        [370.0, 0.0, np.pi/9, 2.5, -2.0, 0.0]
                        ])

    frenet = FrenetSerret2DFrame(route_points, ds=1)

    fpoints = frenet.ctrajectory_to_ftrajectory(cpoints)
    new_cpoints = frenet.ftrajectory_to_ctrajectory(fpoints)

    # currently there is no guarantee on the accuracy of acceleration and curvature
    errors = np.linalg.norm((cpoints - new_cpoints)[:, :C_A], axis=1)

    for error in errors.__iter__():
        assert error < ACCURACY_TH, 'FrenetMovingFrame point conversions aren\'t accurate enough'

    ## FOR DEBUG PURPOSES
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # p1 = fig.add_subplot(111)
    # p1.plot(route_points[:, 0], route_points[:, 1], '-r')
    # p1.plot(frenet.O[:, 0], frenet.O[:, 1], '-k')
    #
    # for i in range(len(cpoints)):
    #     p1.plot(cpoints[i, 0], cpoints[i, 1], '*b')
    #     p1.plot(new_cpoints[i, 0], new_cpoints[i, 1], '.r')
    #
    # print(errors)
    #
    # fig.show()
    # fig.clear()

def test_projectCartesianPoint_fivePointsProjection_accurate():
    ACCURACY_TH = 10 ** -7
    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    cpoints = np.array([[220.0, 0.0], [150.0, 0.0],
                        [280.0, 40.0], [320.0, 60.0],
                        [370.0, 0.0]
                        ])
    frenet = FrenetSerret2DFrame(route_points, ds=1)

    correct_s = np.array([225.92931598, 149.93270474, 334.6431725, 368.37501329, 390.22811868])
    correct_a_s = np.array([[226.09287019, -48.59049741], [149.93758273, -49.97189866],
                            [309.65786895, 15.55022231], [334.78430841, 38.45807098],
                            [354.47302725, 48.50425854]
                            ])
    correct_T_s = np.array([[0.99222944, 0.12442161], [0.99999922, -0.00124904],
                            [0.6361048,  0.77160267], [0.82450236, 0.56585851],
                            [0.95239446, 0.30486849]
                            ])
    correct_N_s = np.array([[-0.12442161, 0.99222944], [0.00124904, 0.99999922],
                            [-0.77160267, 0.6361048], [-0.56585851, 0.82450236],
                            [-0.30486849, 0.95239446]
                            ])
    correct_k_s = np.array([0.0111573838619, -0.000306650767641, -0.00842721944009, -0.0076813879679, -0.015775437513])
    correct_k_s_tag = np.array([5.2208243733e-05, -4.87381176358e-08, -5.94632191054e-06, 4.61577294664e-05, -7.69965827048e-05])

    s, a_s, T_s, N_s, k_s, k_s_tag = frenet._project_cartesian_point(cpoints)
    s_error = s - correct_s
    a_s_error = a_s - correct_a_s
    T_s_error = T_s - correct_T_s
    N_s_error = N_s - correct_N_s
    k_s_error = k_s - correct_k_s
    k_s_tag_error = k_s_tag - correct_k_s_tag

    assert np.all(s_error < ACCURACY_TH)
    assert np.all(a_s_error < ACCURACY_TH)
    assert np.all(T_s_error < ACCURACY_TH)
    assert np.all(N_s_error < ACCURACY_TH)
    assert np.all(k_s_error < ACCURACY_TH)
    assert np.all(k_s_tag_error < ACCURACY_TH)

def test__taylorInterp_fivePointsInterpolation_accurate():
    ACCURACY_TH = 10 ** -7
    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    cpoints = np.array([[220.0, 0.0], [150.0, 0.0],
                        [280.0, 40.0], [320.0, 60.0],
                        [370.0, 0.0]
                        ])
    frenet = FrenetSerret2DFrame(route_points, ds=1)
    s = np.array([225.9293159836173, 149.93270473898278, 334.64317250108013, 368.37501329013446, 390.22811867678485])

    a_s, T_s, N_s, k_s, k_s_tag = frenet._taylor_interp(s)

    correct_a_s = np.array([[226.09287019, -48.59049741], [149.93758273, -49.97189866],
                            [309.65786895, 15.55022231], [334.78430841, 38.45807098],
                            [354.47302725, 48.50425854]
                            ])
    correct_T_s = np.array([[0.99222944, 0.12442161], [0.99999922, -0.00124904],
                            [0.6361048,  0.77160267], [0.82450236, 0.56585851],
                            [0.95239446, 0.30486849]
                            ])
    correct_N_s = np.array([[-0.12442161, 0.99222944], [0.00124904, 0.99999922],
                            [-0.77160267, 0.6361048], [-0.56585851, 0.82450236],
                            [-0.30486849, 0.95239446]
                            ])
    correct_k_s = np.array([0.0111573838619, -0.000306650767641, -0.00842721944009, -0.0076813879679, -0.015775437513])
    correct_k_s_tag = np.array([5.2208243733e-05, -4.87381176358e-08, -5.94632191054e-06, 4.61577294664e-05, -7.69965827048e-05])

    a_s_error = a_s - correct_a_s
    T_s_error = T_s - correct_T_s
    N_s_error = N_s - correct_N_s
    k_s_error = k_s - correct_k_s
    k_s_tag_error = k_s_tag - correct_k_s_tag

    assert np.all(a_s_error < ACCURACY_TH)
    assert np.all(T_s_error < ACCURACY_TH)
    assert np.all(N_s_error < ACCURACY_TH)
    assert np.all(k_s_error < ACCURACY_TH)
    assert np.all(k_s_tag_error < ACCURACY_TH)
