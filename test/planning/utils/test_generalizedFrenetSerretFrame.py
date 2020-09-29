import numpy as np

from decision_making.src.planning.types import C_A, C_V, C_K, C_X, C_Y, FS_SX, FP_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, \
    FrenetSubSegment, GFFType
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
    n = (len(full_frenet.points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n + 1)], full_frenet.T[:(n + 1)],
                                          full_frenet.N[:(n + 1)],
                                          full_frenet.k[:(n + 1)], full_frenet.k_tag[:(n + 1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n:], full_frenet.T[n:], full_frenet.N[n:],
                                            full_frenet.k[n:], full_frenet.k_tag[n:], full_frenet.ds)

    upstream_s_start = upstream_frenet.ds * 3
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max - downstream_frenet.ds * 3
    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                            sub_segments=segmentation)

    fpoints = generalized_frenet.cpoints_to_fpoints(cpoints)
    new_cpoints = generalized_frenet.fpoints_to_cpoints(fpoints)

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
    n = (len(full_frenet.points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n + 1)], full_frenet.T[:(n + 1)],
                                          full_frenet.N[:(n + 1)],
                                          full_frenet.k[:(n + 1)], full_frenet.k_tag[:(n + 1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n:], full_frenet.T[n:], full_frenet.N[n:],
                                            full_frenet.k[n:], full_frenet.k_tag[n:], full_frenet.ds)

    upstream_s_start = upstream_frenet.ds * 2.8  # not an existing point
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max - downstream_frenet.ds * 2.7  # not an existing point
    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                            sub_segments=segmentation)

    fpoints = generalized_frenet.cpoints_to_fpoints(cpoints)
    new_cpoints = generalized_frenet.fpoints_to_cpoints(fpoints)

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
    n = (len(full_frenet.points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n + 1)], full_frenet.T[:(n + 1)],
                                          full_frenet.N[:(n + 1)],
                                          full_frenet.k[:(n + 1)], full_frenet.k_tag[:(n + 1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = upstream_frenet.ds * 2.8  # not an existing point
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max - downstream_frenet.ds * 2.7  # not an existing point
    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                            sub_segments=segmentation)

    fpoints = generalized_frenet.cpoints_to_fpoints(cpoints)
    new_cpoints = generalized_frenet.fpoints_to_cpoints(fpoints)

    errors = np.linalg.norm(cpoints - new_cpoints, axis=1)

    np.testing.assert_array_less(errors, ACCURACY_TH, 'FrenetMovingFrame point conversions aren\'t accurate enough')


def test_ctrajectoryToFtrajectoryToCtrajectory_pointTwoWayConversionTwoFullFrenetFramesDifferentDs_accuratePoseAndVelocity():
    POSITION_ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance
    VEL_ACCURACY_TH = 1e-3  # up to 1 [mm/sec] error in velocity
    ACC_ACCURACY_TH = 1e-3  # up to 1 [mm/sec^2] error in acceleration
    CURV_ACCURACY_TH = 1e-4  # up to 0.0001 [m] error in curvature which accounts to radius of 10,000[m]

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    cpoints = np.array([[2.94, 0.003, -np.pi / 8, 10.0, 1.0, 1e-2], [250.0, 0.0, np.pi / 6, 20.0, 1.1, -1e-2],
                        [280.0, 40.0, np.pi / 7, 10.0, -0.9, -5 * 1e-2], [320.0, 60.0, np.pi / 7, 3, -0.5, -5 * 1e-2],
                        [370.0, 0.0, np.pi / 9, 2.5, -2.0, 0.0]
                        ])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(full_frenet.points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n + 1)], full_frenet.T[:(n + 1)],
                                          full_frenet.N[:(n + 1)],
                                          full_frenet.k[:(n + 1)], full_frenet.k_tag[:(n + 1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = 0
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max

    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                            sub_segments=segmentation)

    fstates = generalized_frenet.ctrajectory_to_ftrajectory(cpoints)
    new_cstates = generalized_frenet.ftrajectory_to_ctrajectory(fstates)

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


def test_ctrajectoriesToFtrajectoriesToCtrajectories_pointTwoWayConversionTwoFullFrenetFramesDifferentDs_accuratePoseAndVelocity():
    POSITION_ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance
    VEL_ACCURACY_TH = 1e-3  # up to 1 [mm/sec] error in velocity
    ACC_ACCURACY_TH = 1e-3  # up to 1 [mm/sec^2] error in acceleration
    CURV_ACCURACY_TH = 1e-4  # up to 0.0001 [m] error in curvature which accounts to radius of 10,000[m]

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0)
    c_trajectory = np.array([[2.94, 0.003, -np.pi / 8, 10.0, 1.0, 1e-2], [250.0, 0.0, np.pi / 6, 20.0, 1.1, -1e-2],
                        [280.0, 40.0, np.pi / 7, 10.0, -0.9, -5 * 1e-2], [320.0, 60.0, np.pi / 7, 3, -0.5, -5 * 1e-2],
                        [370.0, 0.0, np.pi / 9, 2.5, -2.0, 0.0]
                        ])
    c_trajectories = np.array([c_trajectory, c_trajectory, c_trajectory])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(full_frenet.points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n + 1)], full_frenet.T[:(n + 1)],
                                          full_frenet.N[:(n + 1)],
                                          full_frenet.k[:(n + 1)], full_frenet.k_tag[:(n + 1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = 0
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max

    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                            sub_segments=segmentation)

    f_trajectories = generalized_frenet.ctrajectories_to_ftrajectories(c_trajectories)
    new_c_trajectories = generalized_frenet.ftrajectories_to_ctrajectories(f_trajectories)

    position_errors = np.linalg.norm(c_trajectories[:, :, C_X:C_Y] - new_c_trajectories[:, :, C_X:C_Y])
    vel_errors = np.abs(c_trajectories[:, :, C_V] - new_c_trajectories[:, :, C_V])
    acc_errors = np.abs(c_trajectories[:, :, C_A] - new_c_trajectories[:, :, C_A])
    curv_errors = np.abs(c_trajectories[:, :, C_K] - new_c_trajectories[:, :, C_K])

    np.testing.assert_array_less(position_errors, POSITION_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame position conversions aren\'t accurate enough')
    np.testing.assert_array_less(vel_errors, VEL_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame velocity conversions aren\'t accurate enough')
    np.testing.assert_array_less(acc_errors, ACC_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame acceleration conversions aren\'t accurate enough')
    np.testing.assert_array_less(curv_errors, CURV_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame curvature conversions aren\'t accurate enough')


def test_convertFromSegmentState_x_y():
    ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance
    route_points = RouteFixture.get_route(lng=200, k=0.05, step=1, lat=100, offset=-50.0)
    cstate = np.array([460.0, 50.0, np.pi / 9, 0.1, -2, 0])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(full_frenet.points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n + 1)], full_frenet.T[:(n + 1)],
                                          full_frenet.N[:(n + 1)],
                                          full_frenet.k[:(n + 1)], full_frenet.k_tag[:(n + 1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = 0
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max

    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                            sub_segments=segmentation)

    fstate = downstream_frenet.cstate_to_fstate(cstate)

    gen_fstate = generalized_frenet.convert_from_segment_state(fstate, 1)

    new_cstate = generalized_frenet.fstate_to_cstate(gen_fstate)

    errors = np.abs(cstate - new_cstate)

    np.testing.assert_array_less(errors, ACCURACY_TH,
                                 'FrenetMovingFrame point conversions aren\'t accurate enough')


def test_convertFromSegmentStates_x_y():
    ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance
    route_points = RouteFixture.get_route(lng=200, k=0.05, step=1, lat=100, offset=-50.0)
    cpoints = np.array([[100.0, 0.0, -np.pi / 8, 0.1, 1.0, 1e-2], [130.0, 0.0, np.pi / 6, 0.1, 1.1, 1e-2],
                        [150.0, 40.0, np.pi / 7, 10.0, -0.9, 1e-2], [450.0, 50.0, np.pi / 8, 3, -0.5, -5 * 1e-2],
                        [460.0, 50.0, np.pi / 9, 0.1, -2, 0]
                        ])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(full_frenet.points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n + 1)], full_frenet.T[:(n + 1)],
                                          full_frenet.N[:(n + 1)],
                                          full_frenet.k[:(n + 1)], full_frenet.k_tag[:(n + 1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = 0
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max

    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                            sub_segments=segmentation)

    upstream_cpoints = cpoints[:3]
    downstream_cpoints = cpoints[3:]

    upstream_fpoints = upstream_frenet.ctrajectory_to_ftrajectory(upstream_cpoints)
    downstream_fpoints = downstream_frenet.ctrajectory_to_ftrajectory(downstream_cpoints)

    upstream_gen_fpoints = generalized_frenet.convert_from_segment_states(upstream_fpoints, [0] * len(upstream_fpoints))
    downstream_gen_fpoints = generalized_frenet.convert_from_segment_states(downstream_fpoints,
                                                                            [1] * len(downstream_fpoints))

    new_upstream_cpoints = generalized_frenet.ftrajectory_to_ctrajectory(upstream_gen_fpoints)
    new_downstream_cpoints = generalized_frenet.ftrajectory_to_ctrajectory(downstream_gen_fpoints)

    errors_upstream = np.linalg.norm(upstream_cpoints - new_upstream_cpoints, axis=1)
    errors_downstream = np.linalg.norm(downstream_cpoints - new_downstream_cpoints, axis=1)

    np.testing.assert_array_less(errors_upstream, ACCURACY_TH,
                                 'FrenetMovingFrame point conversions aren\'t accurate enough')
    np.testing.assert_array_less(errors_downstream, ACCURACY_TH, 'FrenetMovingFrame point conversions aren\'t accurate enough')


def test_convertToAndFromSegmentStates_x_y():
    ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance
    route_points = RouteFixture.get_route(lng=200, k=0.05, step=1, lat=100, offset=-50.0)
    cpoints = np.array([[100.0, 0.0, -np.pi / 8, 0.1, 1.0, 1e-2], [130.0, 0.0, np.pi / 6, 0.1, 1.1, 1e-2],
                        [150.0, 40.0, np.pi / 7, 10.0, -0.9, 1e-2], [450.0, 50.0, np.pi / 8, 3, -0.5, -5 * 1e-2],
                        [460.0, 50.0, np.pi / 9, 0.1, -2, 0]
                        ])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(full_frenet.points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n + 1)], full_frenet.T[:(n + 1)],
                                          full_frenet.N[:(n + 1)],
                                          full_frenet.k[:(n + 1)], full_frenet.k_tag[:(n + 1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = 0
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max

    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                            sub_segments=segmentation)

    upstream_cpoints = cpoints[:3]
    downstream_cpoints = cpoints[3:]

    upstream_fpoints = upstream_frenet.ctrajectory_to_ftrajectory(upstream_cpoints)
    downstream_fpoints = downstream_frenet.ctrajectory_to_ftrajectory(downstream_cpoints)

    upstream_segment_idxs = [0] * len(upstream_fpoints)
    upstream_gen_fpoints = generalized_frenet.convert_from_segment_states(upstream_fpoints, upstream_segment_idxs)
    new_upstream_segment_idxs, new_upstream_fpoints = generalized_frenet.convert_to_segment_states(upstream_gen_fpoints)

    downstream_segment_idxs = [1] * len(downstream_fpoints)
    downstream_gen_fpoints = generalized_frenet.convert_from_segment_states(downstream_fpoints, downstream_segment_idxs)
    new_downstream_segment_idxs, new_downstream_fpoints = generalized_frenet.convert_to_segment_states(downstream_gen_fpoints)

    errors_upstream = np.linalg.norm(upstream_fpoints - new_upstream_fpoints, axis=1)
    errors_downstream = np.linalg.norm(downstream_fpoints - new_downstream_fpoints, axis=1)

    np.testing.assert_array_less(errors_upstream, ACCURACY_TH, 'Conversions aren\'t accurate enough')
    np.testing.assert_array_less(errors_downstream, ACCURACY_TH, 'Conversions aren\'t accurate enough')
    np.testing.assert_array_equal(upstream_segment_idxs, new_upstream_segment_idxs, 'Segment indices came out wrong')
    np.testing.assert_array_equal(downstream_segment_idxs, new_downstream_segment_idxs, 'Segment indices came out wrong')


def test_convertToAndFromSegmentStates_firstSegmentStartsInMiddle_x_y():
    ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance
    route_points = RouteFixture.get_route(lng=200, k=0.05, step=1, lat=100, offset=-50.0)
    cpoints = np.array([[100.0, 0.0, -np.pi / 8, 0.1, 1.0, 1e-2], [130.0, 0.0, np.pi / 6, 0.1, 1.1, 1e-2],
                        [150.0, 40.0, np.pi / 7, 10.0, -0.9, 1e-2], [450.0, 50.0, np.pi / 8, 3, -0.5, -5 * 1e-2],
                        [460.0, 50.0, np.pi / 9, 0.1, -2, 0]
                        ])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(full_frenet.points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n + 1)], full_frenet.T[:(n + 1)],
                                          full_frenet.N[:(n + 1)],
                                          full_frenet.k[:(n + 1)], full_frenet.k_tag[:(n + 1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = upstream_frenet.ds * 100.8
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max - downstream_frenet.ds * 100.8

    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                            sub_segments=segmentation)

    upstream_cpoints = cpoints[:3]
    downstream_cpoints = cpoints[3:]

    upstream_fpoints = upstream_frenet.ctrajectory_to_ftrajectory(upstream_cpoints)
    downstream_fpoints = downstream_frenet.ctrajectory_to_ftrajectory(downstream_cpoints)

    upstream_segment_idxs = [0] * len(upstream_fpoints)
    upstream_gen_fpoints = generalized_frenet.convert_from_segment_states(upstream_fpoints, upstream_segment_idxs)
    new_upstream_segment_idxs, new_upstream_fpoints = generalized_frenet.convert_to_segment_states(upstream_gen_fpoints)
    cpoints_to_fpoints = generalized_frenet.ctrajectory_to_ftrajectory(upstream_cpoints)
    errors_c2f = (cpoints_to_fpoints - upstream_gen_fpoints)[:, 0]
    np.testing.assert_array_less(errors_c2f, ACCURACY_TH, 'cpoint_to_fpoint conversions aren\'t accurate enough')

    downstream_segment_idxs = [1] * len(downstream_fpoints)
    downstream_gen_fpoints = generalized_frenet.convert_from_segment_states(downstream_fpoints, downstream_segment_idxs)
    new_downstream_segment_idxs, new_downstream_fpoints = generalized_frenet.convert_to_segment_states(downstream_gen_fpoints)

    errors_upstream = np.linalg.norm(upstream_fpoints - new_upstream_fpoints, axis=1)
    errors_downstream = np.linalg.norm(downstream_fpoints - new_downstream_fpoints, axis=1)

    np.testing.assert_array_less(errors_upstream, ACCURACY_TH, 'Conversions aren\'t accurate enough')
    np.testing.assert_array_less(errors_downstream, ACCURACY_TH, 'Conversions aren\'t accurate enough')
    np.testing.assert_array_equal(upstream_segment_idxs, new_upstream_segment_idxs, 'Segment indices came out wrong')
    np.testing.assert_array_equal(downstream_segment_idxs, new_downstream_segment_idxs, 'Segment indices came out wrong')


def test_convertToAndFromSegmentStates_firstSegmentStartsInMiddle_validate_FtoC_and_CtoF_conversions():
    ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance
    route_points = RouteFixture.get_route(lng=200, k=0.05, step=1, lat=100, offset=-50.0)
    cpoints = np.array([[100.0, 0.0, -np.pi / 8, 0.1, 1.0, 1e-2], [130.0, 0.0, np.pi / 6, 0.1, 1.1, 1e-2],
                        [150.0, 40.0, np.pi / 7, 10.0, -0.9, 1e-2], [450.0, 50.0, np.pi / 8, 3, -0.5, -5 * 1e-2],
                        [460.0, 50.0, np.pi / 9, 0.1, -2, 0]
                        ])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(full_frenet.points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n + 1)], full_frenet.T[:(n + 1)],
                                          full_frenet.N[:(n + 1)],
                                          full_frenet.k[:(n + 1)], full_frenet.k_tag[:(n + 1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = upstream_frenet.ds * 100.9
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max - downstream_frenet.ds * 100.3

    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                            sub_segments=segmentation)

    upstream_cpoints = cpoints[:3]
    downstream_cpoints = cpoints[3:]

    upstream_fpoints = upstream_frenet.ctrajectory_to_ftrajectory(upstream_cpoints)
    downstream_fpoints = downstream_frenet.ctrajectory_to_ftrajectory(downstream_cpoints)

    upstream_segment_idxs = [0] * len(upstream_fpoints)
    upstream_gen_fpoints = generalized_frenet.convert_from_segment_states(upstream_fpoints, upstream_segment_idxs)

    upstream_cpoints_to_fpoints = generalized_frenet.ctrajectory_to_ftrajectory(upstream_cpoints)
    upstream_errors_ctof = (upstream_cpoints_to_fpoints - upstream_gen_fpoints)[:, FS_SX]
    np.testing.assert_array_less(upstream_errors_ctof, ACCURACY_TH, 'upstream_cpoint_to_fpoint conversions aren\'t accurate enough')

    upstream_fpoints_to_cpoints = generalized_frenet.ftrajectory_to_ctrajectory(upstream_gen_fpoints)
    upstream_errors_ftoc = (upstream_fpoints_to_cpoints - upstream_cpoints)[:, FS_SX]
    np.testing.assert_array_less(upstream_errors_ftoc, ACCURACY_TH, 'upstream_fpoint_to_cpoint conversions aren\'t accurate enough')

    downstream_segment_idxs = [1] * len(downstream_fpoints)
    downstream_gen_fpoints = generalized_frenet.convert_from_segment_states(downstream_fpoints, downstream_segment_idxs)

    downstream_cpoints_to_fpoints = generalized_frenet.ctrajectory_to_ftrajectory(downstream_cpoints)
    downstream_errors_ctof = (downstream_cpoints_to_fpoints - downstream_gen_fpoints)[:, FS_SX]
    np.testing.assert_array_less(downstream_errors_ctof, ACCURACY_TH, 'downstream_cpoint_to_fpoint conversions aren\'t accurate enough')

    downstream_fpoints_to_cpoints = generalized_frenet.ftrajectory_to_ctrajectory(downstream_gen_fpoints)
    downstream_errors_ftoc = (downstream_fpoints_to_cpoints - downstream_cpoints)[:, FS_SX]
    np.testing.assert_array_less(downstream_errors_ftoc, ACCURACY_TH, 'downstream_fpoint_to_cpoint conversions aren\'t accurate enough')


def test_convertToSegmentStates_multiDimensionalFrenetStates_correctOuputShape():
    route_points = RouteFixture.get_route(lng=200, k=0.05, step=1, lat=100, offset=-50.0)
    cpoints = np.array([[100.0, 0.0, -np.pi / 8, 0.1, 1.0, 1e-2], [130.0, 0.0, np.pi / 6, 0.1, 1.1, 1e-2],
                        [150.0, 40.0, np.pi / 7, 10.0, -0.9, 1e-2], [450.0, 50.0, np.pi / 8, 3, -0.5, -5 * 1e-2],
                        [460.0, 50.0, np.pi / 9, 0.1, -2, 0]
                        ])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(full_frenet.points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n + 1)], full_frenet.T[:(n + 1)],
                                          full_frenet.N[:(n + 1)],
                                          full_frenet.k[:(n + 1)], full_frenet.k_tag[:(n + 1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = 0
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max

    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                            sub_segments=segmentation)

    upstream_cpoints = cpoints[:3]

    upstream_fpoints = upstream_frenet.ctrajectory_to_ftrajectory(upstream_cpoints)

    upstream_3D_fpoints = np.array([upstream_fpoints, upstream_fpoints])

    new_upstream_segment_idxs, new_upstream_fpoints = generalized_frenet.convert_to_segment_states(upstream_3D_fpoints)

    assert upstream_3D_fpoints.shape == new_upstream_fpoints.shape
    assert upstream_3D_fpoints.shape[:-1] == new_upstream_segment_idxs.shape


def test_hasSegmentIds_testMultiDimnesionalArrayOfIndices_validResults():
    ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance
    route_points = RouteFixture.get_route(lng=200, k=0.05, step=1, lat=100, offset=-50.0)
    cpoints = np.array([[100.0, 0.0, -np.pi / 8, 0.1, 1.0, 1e-2], [130.0, 0.0, np.pi / 6, 0.1, 1.1, 1e-2],
                        [150.0, 40.0, np.pi / 7, 10.0, -0.9, 1e-2], [450.0, 50.0, np.pi / 8, 3, -0.5, -5 * 1e-2],
                        [460.0, 50.0, np.pi / 9, 0.1, -2, 0]
                        ])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(full_frenet.points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n + 1)], full_frenet.T[:(n + 1)],
                                          full_frenet.N[:(n + 1)],
                                          full_frenet.k[:(n + 1)], full_frenet.k_tag[:(n + 1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = 0
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max

    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet],
                                                            sub_segments=segmentation)

    segment_idxs_to_query = np.array([[[0, 1], [2, 3]], [[1, 3], [0, 2]]])
    result = generalized_frenet.has_segment_ids(segment_idxs_to_query)
    expectation = np.array([[[True, True], [False, False]], [[True, False], [True, False]]])

    np.testing.assert_equal(result, expectation)

    assert generalized_frenet.has_segment_id(0) == True
    assert generalized_frenet.has_segment_id(2) == False


def test_buildAndConvert_singleFrenetFrame_conversionsAreAccurate():
    ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance
    route_points = RouteFixture.get_route(lng=200, k=0.05, step=1, lat=100, offset=-50.0)
    cpoints = np.array([[100.0, 0.0, -np.pi / 8, 0.1, 1.0, 1e-2], [130.0, 0.0, np.pi / 6, 0.1, 1.1, 1e-2],
                        [150.0, 40.0, np.pi / 7, 10.0, -0.9, 1e-2], [450.0, 50.0, np.pi / 8, 3, -0.5, -5 * 1e-2],
                        [460.0, 50.0, np.pi / 9, 0.1, -2, 0]
                        ])

    full_frenet = FrenetSerret2DFrame.fit(route_points)
    point_idx_start, point_idx_end = len(full_frenet.points)//10, 9*len(full_frenet.points)//10
    segmentation = [FrenetSubSegment(0, full_frenet.ds*point_idx_start, full_frenet.ds*point_idx_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[full_frenet],
                                                            sub_segments=segmentation)

    fpoints = full_frenet.ctrajectory_to_ftrajectory(cpoints)

    segment_idxs = [0] * len(fpoints)
    gen_fpoints = generalized_frenet.convert_from_segment_states(fpoints, segment_idxs)
    new_segment_idxs, new_fpoints = generalized_frenet.convert_to_segment_states(gen_fpoints)

    errors = np.linalg.norm(fpoints - new_fpoints, axis=1)

    np.testing.assert_array_less(errors, ACCURACY_TH, 'Conversions aren\'t accurate enough')
    np.testing.assert_array_equal(segment_idxs, new_segment_idxs, 'Segment indices came out wrong')

    assert len(generalized_frenet.points) == point_idx_end - point_idx_start + 1, 'Segment indices came out wrong'


def test_buildGFF_hasGFFType():
    route_points = RouteFixture.get_route(lng=200, k=0.05, step=1, lat=100, offset=-50.0)
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    point_idx_start, point_idx_end = len(full_frenet.points) // 10, 9 * len(full_frenet.points) // 10
    segmentation = [FrenetSubSegment(0, full_frenet.ds * point_idx_start, full_frenet.ds * point_idx_end)]
    generalized_frenet = GeneralizedFrenetSerretFrame.build(frenet_frames=[full_frenet],
                                                            sub_segments=segmentation)
    assert generalized_frenet.gff_type == GFFType.Normal


def test_lateralConsistency_twoOverlappingGFFs_cartesianPointHasSameLatitudeWRTbothGFFs():
    route_points = RouteFixture.create_cubic_route()
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    segmentation1 = [FrenetSubSegment(0, 0.4, 40)]
    gff1 = GeneralizedFrenetSerretFrame.build(frenet_frames=[full_frenet], sub_segments=segmentation1)
    segmentation2 = [FrenetSubSegment(0, 0.1, 40)]
    gff2 = GeneralizedFrenetSerretFrame.build(frenet_frames=[full_frenet], sub_segments=segmentation2)
    cpoint = np.array([-7, 0])
    fpoint1 = gff1.cpoint_to_fpoint(cpoint)
    fpoint2 = gff2.cpoint_to_fpoint(cpoint)
    assert fpoint1[FP_DX] == fpoint2[FP_DX] and \
           fpoint1[FS_SX] + segmentation1[0].e_i_SStart == fpoint2[FS_SX] + segmentation2[0].e_i_SStart


def test_points_out_of_gff():
    POSITION_ACCURACY_TH = 1e-3  # up to 1 [mm] error in euclidean distance
    VEL_ACCURACY_TH = 1e-3  # up to 1 [mm/sec] error in velocity
    ACC_ACCURACY_TH = 1e-3  # up to 1 [mm/sec^2] error in acceleration
    CURV_ACCURACY_TH = 1e-4  # up to 0.0001 [m] error in curvature which accounts to radius of 10,000[m]

    route_points = RouteFixture.get_route(lng=200, k=0.05, step=1, lat=100, offset=-50.0)
    cstates = np.array([[-100.0, 0.0, -np.pi / 8, 0.1, 1.0, 1e-2], [130.0, 0.0, np.pi / 6, 0.1, 1.1, 1e-2],
                        [150.0, 40.0, np.pi / 7, 10.0, -0.9, 1e-2], [450.0, 50.0, np.pi / 8, 3, -0.5, -5 * 1e-2],
                        [660.0, 50.0, np.pi / 9, 0.1, -2, 0]
                        ])

    # split into two frenet frames that coincide in their last and first points
    full_frenet = FrenetSerret2DFrame.fit(route_points)
    n = (len(full_frenet.points) // 2)
    upstream_frenet = FrenetSerret2DFrame(full_frenet.points[:(n + 1)], full_frenet.T[:(n + 1)],
                                          full_frenet.N[:(n + 1)],
                                          full_frenet.k[:(n + 1)], full_frenet.k_tag[:(n + 1)], full_frenet.ds)
    downstream_frenet = FrenetSerret2DFrame(full_frenet.points[n::2], full_frenet.T[n::2], full_frenet.N[n::2],
                                            full_frenet.k[n::2], full_frenet.k_tag[n::2], full_frenet.ds * 2)

    upstream_s_start = upstream_frenet.ds * 100.8
    upstream_s_end = upstream_frenet.s_max
    downstream_s_start = 0
    downstream_s_end = downstream_frenet.s_max - downstream_frenet.ds * 100.8

    segmentation = [FrenetSubSegment(0, upstream_s_start, upstream_s_end),
                    FrenetSubSegment(1, downstream_s_start, downstream_s_end)]
    gff = GeneralizedFrenetSerretFrame.build(frenet_frames=[upstream_frenet, downstream_frenet], sub_segments=segmentation)

    fstates = gff.ctrajectory_to_ftrajectory(cstates, raise_on_points_out_of_frame=False)
    valid_fstate = fstates[~np.isnan(fstates[:, 0])]
    assert valid_fstate.shape[0] == fstates.shape[0] - 2

    valid_cstates = cstates[~np.isnan(fstates[:, 0])]
    new_cstates = gff.ftrajectory_to_ctrajectory(valid_fstate)

    position_errors = np.linalg.norm(valid_cstates - new_cstates, axis=1)
    vel_errors = np.abs(valid_cstates[:, C_V] - new_cstates[:, C_V])
    acc_errors = np.abs(valid_cstates[:, C_A] - new_cstates[:, C_A])
    curv_errors = np.abs(valid_cstates[:, C_K] - new_cstates[:, C_K])

    np.testing.assert_array_less(position_errors, POSITION_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame position conversions aren\'t accurate enough')
    np.testing.assert_array_less(vel_errors, VEL_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame velocity conversions aren\'t accurate enough')
    np.testing.assert_array_less(acc_errors, ACC_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame acceleration conversions aren\'t accurate enough')
    np.testing.assert_array_less(curv_errors, CURV_ACCURACY_TH,
                                 err_msg='FrenetMovingFrame curvature conversions aren\'t accurate enough')
