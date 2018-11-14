import numpy as np

from decision_making.src.planning.types import C_A, C_V, C_K
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, \
    FrenetSubSegment
from decision_making.test.planning.trajectory.utils import RouteFixture


def test_cpointsToFpointsToCpoints_pointTwoWayConversionExactSegmentationSameDs_accurate():
    ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    cpoints = np.array([[1.98, -40], [150.0, 0.0],
                        [280.0, 40.0], [320.0, 60.0],
                        [370.0, 0.0]
                        ])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(route_points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n+1)], full_frenet.T[:(n+1)], full_frenet.N[:(n+1)],
                                          full_frenet.k[:(n+1)], full_frenet.k_tag[:(n+1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n:], full_frenet.T[n:], full_frenet.N[n:],
                                          full_frenet.k[n:], full_frenet.k_tag[n:], full_frenet.ds)

    upstream_s_start = upstream_frenet.ds * 3
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max - downstream_frenet.ds * 3
    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frame = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                           sub_segments=segmentation)

    fpoints = generalized_frame.cpoints_to_fpoints(cpoints)
    new_cpoints = generalized_frame.fpoints_to_cpoints(fpoints)

    errors = np.linalg.norm(cpoints - new_cpoints, axis=1)

    np.testing.assert_array_less(errors, ACCURACY_TH, 'FrenetMovingFrame point conversions aren\'t accurate enough')


def test_cpointsToFpointsToCpoints_pointTwoWayConversionNonExactSegmentationSameDs_accurate():
    ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    cpoints = np.array([[1.98, -40], [150.0, 0.0],
                        [280.0, 40.0], [320.0, 60.0],
                        [370.0, 0.0]
                        ])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(route_points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n+1)], full_frenet.T[:(n+1)], full_frenet.N[:(n+1)],
                                          full_frenet.k[:(n+1)], full_frenet.k_tag[:(n+1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n:], full_frenet.T[n:], full_frenet.N[n:],
                                          full_frenet.k[n:], full_frenet.k_tag[n:], full_frenet.ds)

    upstream_s_start = upstream_frenet.ds * 2.8  # not an existing point
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max - downstream_frenet.ds * 2.7   # not an existing point
    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frame = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                           sub_segments=segmentation)

    fpoints = generalized_frame.cpoints_to_fpoints(cpoints)
    new_cpoints = generalized_frame.fpoints_to_cpoints(fpoints)

    errors = np.linalg.norm(cpoints - new_cpoints, axis=1)

    np.testing.assert_array_less(errors, ACCURACY_TH, 'FrenetMovingFrame point conversions aren\'t accurate enough')


def test_cpointsToFpointsToCpoints_pointTwoWayConversionNonExactSegmentationDifferentDs_accurate():
    ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    cpoints = np.array([[1.98, -40], [150.0, 0.0],
                        [280.0, 40.0], [320.0, 60.0],
                        [370.0, 0.0]
                        ])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(route_points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n+1)], full_frenet.T[:(n+1)], full_frenet.N[:(n+1)],
                                          full_frenet.k[:(n+1)], full_frenet.k_tag[:(n+1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = upstream_frenet.ds * 2.8  # not an existing point
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max - downstream_frenet.ds * 2.7   # not an existing point
    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frame = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                           sub_segments=segmentation)

    fpoints = generalized_frame.cpoints_to_fpoints(cpoints)
    new_cpoints = generalized_frame.fpoints_to_cpoints(fpoints)

    errors = np.linalg.norm(cpoints - new_cpoints, axis=1)

    np.testing.assert_array_less(errors, ACCURACY_TH, 'FrenetMovingFrame point conversions aren\'t accurate enough')


def test_ctrajectoryToFtrajectoryToCtrajectory_pointTwoWayConversionTwoFullFrenetFramesDifferentDs_accuratePoseAndVelocity():
    POSITION_ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance
    VEL_ACCURACY_TH = 1e-3  # up to 1 [mm/sec] error in velocity
    ACC_ACCURACY_TH = 1e-3  # up to 1 [mm/sec^2] error in acceleration
    CURV_ACCURACY_TH = 1e-4  # up to 0.0001 [m] error in curvature which accounts to radius of 10,000[m]

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    cpoints = np.array([[2.94, 0.003, -np.pi/8, 10.0, 1.0, 1e-2], [250.0, 0.0, np.pi/6, 20.0, 1.1, -1e-2],
                        [280.0, 40.0, np.pi/7, 10.0, -0.9, -5*1e-2], [320.0, 60.0, np.pi/7, 3, -0.5, -5*1e-2],
                        [370.0, 0.0, np.pi/9, 2.5, -2.0, 0.0]
                        ])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(route_points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n+1)], full_frenet.T[:(n+1)], full_frenet.N[:(n+1)],
                                          full_frenet.k[:(n+1)], full_frenet.k_tag[:(n+1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = 0
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max

    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frame = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                           sub_segments=segmentation)

    fstates = generalized_frame.ctrajectory_to_ftrajectory(cpoints)
    new_cstates = generalized_frame.ftrajectory_to_ctrajectory(fstates)

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


def test_convertFromSegmentState_x_y():
    route_points = RouteFixture.get_route(lng=200, k=0.05, step=1, lat=100, offset=-50.0)
    cpoints = np.array([[100.0, 0.0, -np.pi/8, 0, 1.0, 1e-2], [130.0, 0.0, np.pi/6, 0, 1.1, 1e-2],
                        [150.0, 40.0, np.pi/7, 10.0, -0.9, 1e-2], [450.0, 50.0, np.pi/8, 3, -0.5, -5*1e-2],
                        [460.0, 50.0, np.pi/9, 0, -2, 0]
                        ])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(route_points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n+1)], full_frenet.T[:(n+1)], full_frenet.N[:(n+1)],
                                          full_frenet.k[:(n+1)], full_frenet.k_tag[:(n+1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = 0
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max

    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frame = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                           sub_segments=segmentation)

    upstream_cpoints = cpoints[:3]
    downstream_cpoints = cpoints[3:]

    upstream_fpoints = upstream_frenet.ctrajectory_to_ftrajectory(upstream_cpoints)
    downstream_fpoints = downstream_frenet.ctrajectory_to_ftrajectory(downstream_cpoints)

    upstream_gen_fpoints = generalized_frame.convert_from_segment_state(upstream_fpoints, [0] * len(upstream_fpoints))
    downstream_gen_fpoints = generalized_frame.convert_from_segment_state(downstream_fpoints, [1] * len(downstream_fpoints))

    new_upstream_cpoints = generalized_frame.ftrajectory_to_ctrajectory(upstream_gen_fpoints)
    new_downstream_cpoints = generalized_frame.ftrajectory_to_ctrajectory(downstream_gen_fpoints)

    np.testing.assert_array_equal(new_upstream_cpoints, upstream_cpoints)
    np.testing.assert_array_equal(new_downstream_cpoints, downstream_cpoints)


