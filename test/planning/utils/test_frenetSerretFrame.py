import numpy as np

from decision_making.src.global_constants import TRAJECTORY_CURVE_SPLINE_FIT_ORDER, ROAD_MAP_REQUIRED_RES
from decision_making.src.planning.types import C_A, C_V, C_K, FP_SX, FS_SX, FS_DX, FS_DV, FS_SV, FS_DA, FS_SA, FP_DX, \
    C_YAW
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.planning.trajectory.utils import RouteFixture
from decision_making.src.utils.geometry_utils import CartesianFrame
from decision_making.test.messages.scene_static_fixture import scene_static_pg_no_split


def test_cpointsToFpointsToCpoints_pointTwoWayConversion_accurate():
    ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    cpoints = np.array([[220.0, 0.0], [150.0, 0.0],
                        [280.0, 40.0], [320.0, 60.0],
                        [370.0, 0.0]
                        ])

    frenet = FrenetSerret2DFrame.fit(route_points)

    fpoints = frenet.cpoints_to_fpoints(cpoints)
    new_cpoints = frenet.fpoints_to_cpoints(fpoints)

    errors = np.linalg.norm(cpoints - new_cpoints, axis=1)

    np.testing.assert_array_less(errors, ACCURACY_TH, 'FrenetMovingFrame point conversions aren\'t accurate enough')

    ## FOR DEBUG PURPOSES
    # import matplotlib.pyplot as plt
    # plt.switch_backend('QT5Agg')
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
    POSITION_ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance
    VEL_ACCURACY_TH = 1e-3  # up to 1 [mm/sec] error in velocity
    ACC_ACCURACY_TH = 1e-3  # up to 1 [mm/sec^2] error in acceleration
    CURV_ACCURACY_TH = 1e-4  # up to 0.0001 [m] error in curvature which accounts to radius of 10,000[m]

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    cpoints = np.array([[150.0, 0.0, -np.pi/8, 10.0, 1.0, 1e-2], [250.0, 0.0, np.pi/6, 20.0, 1.1, -1e-2],
                        [280.0, 40.0, np.pi/7, 10.0, -0.9, -5*1e-2], [320.0, 60.0, np.pi/7, 3, -0.5, -5*1e-2],
                        [370.0, 0.0, np.pi/9, 2.5, -2.0, 0.0]
                        ])

    frenet = FrenetSerret2DFrame.fit(route_points)

    fstates = frenet.ctrajectory_to_ftrajectory(cpoints)
    new_cstates = frenet.ftrajectory_to_ctrajectory(fstates)

    position_errors = np.linalg.norm(cpoints - new_cstates, axis=1)
    vel_errors = np.abs(cpoints[:, C_V] - new_cstates[:, C_V])
    acc_errors = np.abs(cpoints[:, C_A] - new_cstates[:, C_A])
    curv_errors = np.abs(cpoints[:, C_K] - new_cstates[:, C_K])

    np.testing.assert_array_less(position_errors, POSITION_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame position conversions aren\'t accurate enough')
    np.testing.assert_array_less(vel_errors, VEL_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame velocity conversions aren\'t accurate enough')
    np.testing.assert_array_less(acc_errors, ACC_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame acceleration conversions aren\'t accurate enough')
    np.testing.assert_array_less(curv_errors, CURV_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame curvature conversions aren\'t accurate enough')

    ## FOR DEBUG PURPOSES
    # import matplotlib.pyplot as plt
    # plt.switch_backend('QT5Agg')
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


def test_ftrajectoryToCtrajectoryToFtrajectory_pointTwoWayConversion_accuratePoseAndVelocity():
    POSITION_ACCURACY_TH = 1e-3  # up to 1 [mm] error in positions
    VEL_ACCURACY_TH = 1e-3  # up to 1 [mm/sec] error in velocity
    ACC_ACCURACY_TH = 1e-3  # up to 1 [mm/sec^2] error in acceleration

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    fstates = np.array([[1.49932706e+02,   9.10405803e+00,   1.30695915e+00,
                         4.99719376e+01,  -3.81529167e+00,   5.68633517e-01],
                       [2.76252304e+02,   2.88384989e+01,   5.51670492e-01,
                        3.63734600e+01,  -2.56097695e+00,  -9.01756345e+00],
                       [3.34644464e+02,   6.85768466e+00,  -1.84373678e+00,
                        3.84367118e+01,  -4.19194237e+00,  -3.63752908e+00],
                       [3.68358154e+02,   2.46945361e+00,  -4.47959102e-01,
                        2.61271977e+01,  -4.56255049e-01,  -3.12473058e-01],
                       [3.90229981e+02,   1.27084880e+01,  -7.14379434e+00,
                        -5.09288718e+01,   9.82118772e-02,   4.22253295e-01]])

    frenet = FrenetSerret2DFrame.fit(route_points)

    projected_cstates = frenet.ftrajectory_to_ctrajectory(fstates)
    new_fstates = frenet.ctrajectory_to_ftrajectory(projected_cstates)

    # currently there is no guarantee on the accuracy of acceleration and curvature
    position_errors = np.abs(fstates[:, [FS_SX, FS_DX]] - new_fstates[:, [FS_SX, FS_DX]])
    vel_errors = np.abs(fstates[:, [FS_SV, FS_DV]] - new_fstates[:, [FS_SV, FS_DV]])
    acc_errors = np.abs(fstates[:, [FS_SA, FS_DA]] - new_fstates[:, [FS_SA, FS_DA]])

    np.testing.assert_array_less(position_errors, POSITION_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame position conversions aren\'t accurate enough')
    np.testing.assert_array_less(vel_errors, VEL_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame velocity conversions aren\'t accurate enough')
    np.testing.assert_array_less(acc_errors, ACC_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame acceleration conversions aren\'t accurate enough')

    # # FOR DEBUG PURPOSES
    # import matplotlib.pyplot as plt
    # plt.switch_backend('QT5Agg')
    # fig = plt.figure()
    # p1 = fig.add_subplot(111)
    #
    # for i in range(len(projected_cstates)):
    #     p1.plot(fstates[i, FS_SX], fstates[i, FS_DX], '*b')
    #     p1.plot(new_fstates[i, FS_SX], new_fstates[i, FS_DX], '.r')
    #
    # fig.show()
    # fig.clear()

def test_projectCartesianPoint_fivePointsProjection_accurate():
    ACCURACY_TH = 1e-3  # accuracy of at least 1[mm]

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    fpoints = np.array([[220.0, 0.0], [150.0, 0.0],
                        [280.0, 40.0], [320.0, 60.0],
                        [370.0, 0.0]
                        ])
    frenet = FrenetSerret2DFrame.fit(route_points)
    projected_cpoints = frenet.fpoints_to_cpoints(fpoints)

    s, _, _, _, _, _ = frenet._project_cartesian_points(projected_cpoints)

    correct_s = fpoints[:, FP_SX]

    s_error = np.abs(s - correct_s)

    np.testing.assert_array_less(s_error, ACCURACY_TH, 'FrenetSerret2DFrame._project_cartesian_points is not accurate')


def test_fitFrenet_originalRoutePointsAreProjected_errorsAreLowEnough(scene_static_pg_no_split):

    POSITION_ACCURACY_TH = 1e-1  # up to 10 [cm] error in euclidean distance

    upsampling_factor_for_test = 4

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_no_split)
    road_segment_id = MapUtils.get_road_segment_ids()[0]
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)[0]
    route_points = MapUtils.get_lane_frenet_frame(lane_id).points

    # "Train" points: assumed to be sampled sufficiently dense (according to ROAD_MAP_REQUIRED_RES)
    _, route_points_upsampled_to_required_res, _ = \
        CartesianFrame.resample_curve(curve=route_points,
                                      step_size=ROAD_MAP_REQUIRED_RES,
                                      preserve_step_size=False,
                                      spline_order=TRAJECTORY_CURVE_SPLINE_FIT_ORDER)

    # "Test" points: upsampling of the train points, used for accuracy testing
    _, test_points, _ = CartesianFrame.resample_curve(curve=route_points,
                                                      step_size=ROAD_MAP_REQUIRED_RES/upsampling_factor_for_test,
                                                      preserve_step_size=False,
                                                      spline_order=TRAJECTORY_CURVE_SPLINE_FIT_ORDER)

    frenet = FrenetSerret2DFrame.fit(route_points_upsampled_to_required_res)

    # project the original route points unto the fitted curve - last point can be outside the curve
    # (due to length estimation)
    test_points = test_points[1:-1]
    fprojections = frenet.cpoints_to_fpoints(test_points)
    fprojections[:, FP_DX] = 0

    new_test_points = frenet.fpoints_to_cpoints(fprojections)

    position_errors = np.linalg.norm(test_points - new_test_points, axis=1)

    np.testing.assert_array_less(position_errors, POSITION_ACCURACY_TH,
                                err_msg='FrenetMovingFrame position conversions aren\'t accurate enough')

    # # FOR DEBUG PURPOSES
    # import matplotlib.pyplot as plt
    # plt.switch_backend('QT5Agg')
    # fig = plt.figure()
    # p1 = fig.add_subplot(111)
    # p1.plot(route_points[:, 0], route_points[:, 1], '*r')
    # p1.plot(test_points[:, 0], test_points[:, 1], '*g')
    # p1.plot(frenet.O[:, 0], frenet.O[:, 1], '-k')
    #
    # fig.show()
    # fig.clear()


def test_fitFrenet_useAnchors_errorsInAnchorsAreLowEnough(scene_static_pg_no_split):

    POSITION_ACCURACY_TH = 1e-1  # up to 10 [cm] error in euclidean distance
    ANCHOR_POSITION_ACCURACY_TH = 1e-2  # up to 1 [cm] error in euclidean distance

    upsampling_factor_for_test = 4

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_no_split)
    road_segment_id = MapUtils.get_road_segment_ids()[0]
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)[0]
    route_points = MapUtils.get_lane_frenet_frame(lane_id).points

    # "Train" points: assumed to be sampled sufficiently dense (according to ROAD_MAP_REQUIRED_RES)
    _, route_points_upsampled_to_required_res, _ = \
        CartesianFrame.resample_curve(curve=route_points,
                                      step_size=ROAD_MAP_REQUIRED_RES,
                                      preserve_step_size=False,
                                      spline_order=TRAJECTORY_CURVE_SPLINE_FIT_ORDER)

    # "Test" points: upsampling of the train points, used for accuracy testing
    _, test_points, _ = CartesianFrame.resample_curve(curve=route_points,
                                                      step_size=ROAD_MAP_REQUIRED_RES/upsampling_factor_for_test,
                                                      preserve_step_size=False,
                                                      spline_order=TRAJECTORY_CURVE_SPLINE_FIT_ORDER)

    frenet = FrenetSerret2DFrame.fit(route_points_upsampled_to_required_res,
                                     prefix_anchor=route_points_upsampled_to_required_res[0][np.newaxis],
                                     suffix_anchor=route_points_upsampled_to_required_res[-1][np.newaxis])

    # project the original route points unto the fitted curve - last point can be outside the curve
    # (due to length estimation)
    test_points = test_points[1:-1]
    fprojections = frenet.cpoints_to_fpoints(test_points)
    fprojections[:, FP_DX] = 0

    new_test_points = frenet.fpoints_to_cpoints(fprojections)

    position_errors = np.linalg.norm(test_points - new_test_points, axis=1)

    np.testing.assert_array_less(position_errors, POSITION_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame position conversions aren\'t accurate enough')
    np.testing.assert_array_less(np.array([position_errors[0], position_errors[-1]]), ANCHOR_POSITION_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame position conversions aren\'t accurate enough')


def test_ctrajectoryToFtrajectoryToCtrajectory_zeroVelocityTwoWayConversion_accuratePoseAndVelocity():
    POSITION_ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance
    VEL_ACCURACY_TH = 1e-3  # up to 1 [mm/sec] error in velocity
    YAW_ACCURACY_TH = 1e-3  # up to 0.01 [rad] error in yaw

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=1, lat=100, offset=-50.0)
    cpoints = np.array([[100.0, 0.0, -np.pi/8, 0, 1.0, 1e-2], [130.0, 0.0, np.pi/6, 0, 1.1, 1e-2],
                        [150.0, 40.0, np.pi/7, 10.0, -0.9, 1e-2], [450.0, 50.0, np.pi/8, 3, -0.5, -5*1e-2],
                        [460.0, 50.0, np.pi/9, 0, -2, 0]
                        ])

    frenet = FrenetSerret2DFrame.fit(route_points)

    fstates = frenet.ctrajectory_to_ftrajectory(cpoints)
    new_cstates = frenet.ftrajectory_to_ctrajectory(fstates)

    # currently there is no guarantee on the accuracy of acceleration and curvature
    position_errors = np.linalg.norm(cpoints[:, :C_YAW] - new_cstates[:, :C_YAW], axis=1)
    vel_errors = np.abs(cpoints[:, C_V] - new_cstates[:, C_V])

    # verify that zero velocity points take their road-relative yaw from the nearest non-zero-vel point on the trajectory
    assert abs(new_cstates[0, C_YAW]) < YAW_ACCURACY_TH
    assert abs(new_cstates[1, C_YAW]) < YAW_ACCURACY_TH
    assert abs(new_cstates[-1, C_YAW]) < YAW_ACCURACY_TH

    np.testing.assert_array_less(position_errors, POSITION_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame position conversions aren\'t accurate enough')
    np.testing.assert_array_less(vel_errors, VEL_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame velocity conversions aren\'t accurate enough')
