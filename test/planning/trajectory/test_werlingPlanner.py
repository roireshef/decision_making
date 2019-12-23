
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.test.utils.scene_static_utils import SceneStaticUtils
from decision_making.src.state.map_state import MapState
from decision_making.src.utils.map_utils import MapUtils
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from decision_making.src.global_constants import EGO_LENGTH, EGO_WIDTH, \
    VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS, \
    DEFAULT_ACCELERATION, DEFAULT_CURVATURE, EGO_HEIGHT, LON_JERK_COST_WEIGHT, LAT_JERK_COST_WEIGHT, \
    LON_MARGIN_FROM_EGO, ROAD_SHOULDERS_WIDTH, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams, SigmoidFunctionParams
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from decision_making.src.planning.trajectory.cost_function import TrajectoryPlannerCosts, Jerk
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner, \
    SamplableWerlingTrajectory
from decision_making.src.planning.types import C_X, C_Y, C_YAW, C_V, FP_SX, FP_DX, FS_DX, \
    CartesianExtendedState, CartesianTrajectory
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import State, ObjectSize, DynamicObject, EgoState
from decision_making.test.planning.trajectory.utils import RouteFixture, PlottableSigmoidBoxObstacle, \
    WerlingVisualizer
from decision_making.src.utils.geometry_utils import CartesianFrame
from rte.python.logger.AV_logger import AV_Logger


@patch('decision_making.src.planning.trajectory.werling_planner.SX_STEPS', 5)
@patch('decision_making.src.planning.trajectory.werling_planner.DX_STEPS', 5)
@patch('decision_making.src.planning.trajectory.werling_planner.SX_OFFSET_MIN', -8)
@patch('decision_making.src.planning.trajectory.werling_planner.SX_OFFSET_MAX', 0)
@patch('decision_making.src.planning.trajectory.werling_planner.DX_OFFSET_MIN', -1.6)
@patch('decision_making.src.planning.trajectory.werling_planner.DX_OFFSET_MAX', 1.6)
def test_werlingPlanner_toyScenario_noException():
    logger = AV_Logger.get_logger('test_werlingPlanner_toyScenario_noException')
    reference_route = FrenetSerret2DFrame.fit(RouteFixture.get_route(lng=10, k=1, step=1, lat=1, offset=-.5))

    v0 = 5
    vT = 5
    Ts = 2

    predictor = RoadFollowingPredictor(logger)

    goal_pos = np.array([15, 0.005])
    goal_s = reference_route.cpoint_to_fpoint(goal_pos)[0]
    goal = np.concatenate((goal_pos, reference_route.get_yaw(np.array([goal_s])), [vT, DEFAULT_ACCELERATION, DEFAULT_CURVATURE]))

    pos1 = np.array([7, -.5])
    yaw1 = 0
    pos2 = np.array([11, 1.5])
    yaw2 = np.pi / 4

    obs = list([
        DynamicObject.create_from_cartesian_state(obj_id=0, timestamp=950 * 10e6,
                                                  cartesian_state=np.array([pos1[0], pos1[1], yaw1, 0, 0, 0]),
                                                  size=ObjectSize(1.5, 0.5, 0), confidence=1.0, off_map=False),
        DynamicObject.create_from_cartesian_state(obj_id=1, timestamp=950 * 10e6,
                                                  cartesian_state=np.array([pos2[0], pos2[1], yaw2, 0, 0, 0]),
                                                  size=ObjectSize(1.5, 0.5, 0), confidence=1.0, off_map=False)
    ])

    # set ego starting longitude > 0 in order to prevent the starting point to be outside the reference route
    ego = EgoState.create_from_cartesian_state(obj_id=-1, timestamp=1000 * 10e6,
                                               cartesian_state=np.array([LON_MARGIN_FROM_EGO, 0, 0, v0, 0.0, 0.0]),
                                               size=ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT), confidence=1.0,
                                               off_map=False)

    state = State(is_sampled=False, occupancy_state=None, dynamic_objects=obs, ego_state=ego)

    cost_params = TrajectoryCostParams(left_lane_cost=SigmoidFunctionParams(10, 1.0, 1.0),
                                       right_lane_cost=SigmoidFunctionParams(10, 1.0, 1.0),
                                       left_road_cost=SigmoidFunctionParams(10, 1.0, 1.5),
                                       right_road_cost=SigmoidFunctionParams(10, 1.0, 1.5),
                                       left_shoulder_cost=SigmoidFunctionParams(10, 1.0, 2),
                                       right_shoulder_cost=SigmoidFunctionParams(10, 1.0, 2),
                                       obstacle_cost_x=SigmoidFunctionParams(100, 10.0, 0.3),
                                       obstacle_cost_y=SigmoidFunctionParams(100, 10.0, 0.3),
                                       dist_from_goal_cost=SigmoidFunctionParams(100, 10.0, 0.3),
                                       dist_from_goal_lat_factor=1.0,
                                       lon_jerk_cost_weight=LON_JERK_COST_WEIGHT,
                                       lat_jerk_cost_weight=LAT_JERK_COST_WEIGHT,
                                       velocity_limits=VELOCITY_LIMITS,
                                       lon_acceleration_limits=LON_ACC_LIMITS,
                                       lat_acceleration_limits=LAT_ACC_LIMITS,
                                       desired_velocity=BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED)

    planner = WerlingPlanner(logger, predictor)

    samplable, ctrajectories, costs = planner.plan(state=state, reference_route=reference_route, goal=goal,
                                                   T_target_horizon=Ts, T_trajectory_end_horizon=Ts, cost_params=cost_params)

    samplable.sample(np.arange(0, 1, 0.01) + ego.timestamp_in_sec)

    assert True

    # import matplotlib.pyplot as plt
    # #plt.switch_backend('QT5Agg')
    #
    # fig = plt.figure()
    # p1 = fig.add_subplot(211)
    # plt.title('A sample from possible trajectories, Ts=%s, TD_STEPS=%s' % (Ts, TD_STEPS))
    # p2 = fig.add_subplot(212)
    # plt.title('Chosen trajectory')
    # time_samples = np.arange(0.0, Ts, 0.1) + ego.timestamp_in_sec
    # plottable_obs = [PlottableSigmoidDynamicBoxObstacle(o, cost_params.obstacle_cost_x.k,
    #                                                     np.array([cost_params.obstacle_cost_x.offset,
    #                                                               cost_params.obstacle_cost_y.offset]),
    #                                                     time_samples, predictor)
    #                  for o in state.dynamic_objects]
    # WerlingVisualizer.plot_obstacles(p1, plottable_obs)
    # WerlingVisualizer.plot_obstacles(p2, plottable_obs)
    # WerlingVisualizer.plot_route(p1, route_points[:, :2])
    # WerlingVisualizer.plot_route(p2, route_points[:, :2])
    #
    # WerlingVisualizer.plot_best(p2, ctrajectories[0])
    # WerlingVisualizer.plot_alternatives(p1, ctrajectories, costs)
    #
    # print(costs)
    # print('\n minimal is: ', np.min(costs))
    #
    # WerlingVisualizer.plot_route(p1, route_points)
    # plt.show()
    # fig.clear()


# remove this skip if you want to run the test
@pytest.mark.skip(reason="takes too long.")
def test_werlingPlanner_testCostsShaping_saveImagesForVariousScenarios():
    """
    The route is set to route_points by calling to RouteFixture.get_route(). Currently it is a straight line.
    The test runs with 16 iterations. In each iteration one or more obstacles move.
    At each iteration the image with costs and calculated trajectories is saved in a file.
    The number of obstacles is determined by the length of obs_poses.
    """

    logger = AV_Logger.get_logger('test_werlingPlanner_twoStaticObjScenario_withCostViz')
    predictor = RoadFollowingPredictor(logger)
    road_id = 1
    lane_width = 3.6
    num_lanes = 2
    road_width = num_lanes * lane_width
    lng = 40  # [m] route_points length
    ext = 4  # [m] extension for route_points to prevent out-of-range Frenet-Seret projection
    T = 4.6  # planning time
    v0 = 6  # start velocity
    vT = 10  # end velocity

    for test_idx in range(0, 49):

        reference_route_latitude = 3 * lane_width / 2
        start_ego_lat = lane_width / 2

        if test_idx < 20 or test_idx >= 40:
            curvature = 0.0
        else:
            curvature = 0.2

        if test_idx < 40:  # test safety vs deviations vs goal, and consistency for small changes
            obs_poses = np.array([np.array([4, 0]), np.array([22, -0.0 - (test_idx % 20) * 0.2])])
            goal_latitude = lane_width / 2
        else:  # test jerk vs. goal
            obs_poses = np.array([])
            goal_latitude = reference_route_latitude = start_ego_lat = lane_width / 2
            v0 = 8
            vT = 8
            T = 2 * lng / (v0 + vT + (test_idx - 40))

        # Create reference route (normal and extended). The extension is intended to prevent
        # overflow of projection on the ref route
        route_points, ext_route_points = \
            create_route_for_test_werlingPlanner(road_id, num_lanes, lane_width, reference_route_latitude, lng, ext, curvature)

        frenet = FrenetSerret2DFrame.fit(ext_route_points[:, :2])

        # create state and goal based on ego parameters and obstacles' location
        state, goal = create_state_for_test_werlingPlanner(frenet, obs_poses, reference_route_latitude, ext, lng,
                                                           v0, vT, start_ego_lat, goal_latitude)
        goal_map_state = MapState(frenet.cstate_to_fstate(goal), MapUtils.get_lanes_ids_from_road_segment_id(road_id)[0])

        cost_params = BasePlanner._generate_cost_params(map_state=goal_map_state, ego_size=state.ego_state.size)

        # run Werling planner
        planner = WerlingPlanner(logger, predictor)
        _, ctrajectories, costs = planner.plan(state=state, reference_route=frenet,
                                               goal=goal, T_target_horizon=T, cost_params=cost_params)

        time_samples = np.arange(0, T + np.finfo(np.float16).eps, planner.dt) + \
                       state.ego_state.timestamp_in_sec
        assert time_samples.shape[0] == ctrajectories.shape[1]

        offsets = np.array([cost_params.obstacle_cost_x.offset, cost_params.obstacle_cost_y.offset])
        plottable_obs = [PlottableSigmoidBoxObstacle(state, o, cost_params.obstacle_cost_x.k, offsets, time_samples,
                                                     planner.predictor)
                         for o in state.dynamic_objects]

        # create pixels grid of the visualization image and compute costs for these pixels for given time samples
        t = 0  # time index of time_samples
        pixels, pixel_costs = compute_pixel_costs(route_points, reference_route_latitude, road_width, state,
                                                  cost_params, time_samples[t:(t + 1)], planner, frenet)

        visualize_test_scenario(route_points, reference_route_latitude, road_width, state, goal, ctrajectories, costs,
                                pixels, pixel_costs, plottable_obs, 'test_costs' + str(test_idx) + '.png')


def create_route_for_test_werlingPlanner(road_id: int, num_lanes: int, lane_width: float,
                                         reference_route_latitude: float, lng: float, ext: float, curvature: float) -> \
        [np.array, np.array]:
    """
    Create reference route for test_werlingPlanner visualization.
    :param road_id: road id
    :param num_lanes: number of lanes
    :param lane_width: [m] lane width
    :param reference_route_latitude: [m] latitude of the reference route
    :param lng: [m] length of the reference route
    :param ext: [m] extension of the reference route (in two sides)
    :param curvature: curvature of the reference route
    :return: route_points (reference route), ext_route_points (extended reference route)
    """
    step = 0.2
    route_xy = RouteFixture.create_cubic_route(lng=lng, lat=reference_route_latitude, ext=0, step=step, curvature=curvature)
    ext_route_xy = RouteFixture.create_cubic_route(lng=lng, lat=reference_route_latitude, ext=ext, step=step,
                                                   curvature=curvature)

    test_scene_static = SceneStaticUtils.create_scene_static_from_points(road_segment_ids=[road_id],
                                                                         num_lanes=num_lanes, lane_width=lane_width,
                                                                         points_of_roads=[ext_route_xy])
    SceneStaticModel.get_instance().set_scene_static(test_scene_static)

    route_points = CartesianFrame.add_yaw_and_derivatives(route_xy)
    ext_route_points = CartesianFrame.add_yaw_and_derivatives(ext_route_xy)
    return route_points, ext_route_points


def create_state_for_test_werlingPlanner(frenet: FrenetSerret2DFrame, obs_poses: np.array,
                                         reference_route_latitude: float, route_ext: float, route_lng: float,
                                         v0: float, vT: float,
                                         start_ego_lat: float, goal_latitude: float) -> [State, np.array]:
    """
    Given Frenet frame, ego parameters and obstacles in Frenet, create state and goal that are consistent with
    the Frenet frame.
    :param frenet: Frenet frame
    :param obs_poses: [FP_SX, FP_DX] Frenet location of the obstacles
    :param reference_route_latitude: [m] reference route latitude
    :param route_ext: [m] extended part of the reference route (to prevent "out of route projection")
    :param route_lng: [m] reference route length (without extensions)
    :param start_ego_lat: [m] latitude of ego
    :param goal_latitude: [m] latitude of the goal
    :return: state and goal
    """
    # Convert two points (start and goal) from Frenet to cartesian coordinates.
    ftraj_start_goal = np.array([np.array([route_ext, v0, 0, start_ego_lat - reference_route_latitude, 0, 0]),
                                 np.array([route_lng + route_ext, vT, 0, goal_latitude - reference_route_latitude, 0, 0])])
    ctraj_start_goal = frenet.ftrajectory_to_ctrajectory(ftraj_start_goal)

    ego = EgoState.create_from_cartesian_state(obj_id=-1, timestamp=0, size=ObjectSize(EGO_LENGTH, EGO_WIDTH, 0),
                                               confidence=1.0,
                                               cartesian_state=np.array([ctraj_start_goal[0][C_X], ctraj_start_goal[0][C_Y],
                                                                   ctraj_start_goal[0][C_YAW], ctraj_start_goal[0][C_V],
                                                                   0.0, 0.0]))

    goal = ctraj_start_goal[1]
    goal[C_X] -= 0.001

    obs = []
    for i, pose in enumerate(obs_poses):
        fobs = np.array([pose[FP_SX], pose[FP_DX]])
        cobs = frenet.fpoint_to_cpoint(fobs)
        dynamic_object = DynamicObject.create_from_cartesian_state(obj_id=i, timestamp=0,
                                                                   cartesian_state=np.array([cobs[C_X], cobs[C_Y],
                                                                        frenet.get_yaw(pose[FP_SX]), 0.0, 0.0, 0.0]),
                                                                   size=ObjectSize(4, 1.8, 0), confidence=1.0)
        obs.append(dynamic_object)

    state = State(is_sampled=False, occupancy_state=None, dynamic_objects=obs, ego_state=ego)
    return state, goal


def compute_pixel_costs(route_points: np.array, reference_route_latitude: float, road_width: float,
                        state: State, cost_params: TrajectoryCostParams, time_samples: np.array,
                        planner: WerlingPlanner, frenet: FrenetSerret2DFrame) -> \
        [np.array, np.array]:
    """
    1. Create visualization image, whose size fits to route_points.
    2. Given a scenario (road, state, goal), trajectory cost params, create grid of pixels for the visualization image.
    3. Compute costs for these pixels for the given time samples.
    :param route_points: route points for Werling planner
    :param reference_route_latitude: [m] reference route latitude relatively to the right edge of the road
    :param road_width: [m] road width
    :param state: state including ego and obstacles
    :param cost_params: trajectory cost parameters, obtained from semantic_actions_grid_policy
    :param time_samples: time_samples of the planning
    :param planner: Werling planner
    :param frenet: Frenet frame based on the route_points
    :return: 2D pixels array, pixel costs
    """
    # create pixels array
    xrange = (route_points[0, C_X], route_points[-1, C_X])
    yrange = (np.min(route_points[:, C_Y]) - reference_route_latitude - ROAD_SHOULDERS_WIDTH,
              np.max(route_points[:, C_Y]) - reference_route_latitude + road_width + ROAD_SHOULDERS_WIDTH)
    x = np.arange(xrange[0], xrange[1], 0.1)
    y = np.arange(yrange[0], yrange[1], 0.1)
    width = x.shape[0]
    height = y.shape[0]
    pixels = np.transpose([np.tile(x, y.shape[0]), np.repeat(y, x.shape[0])])

    # create cartesian pixels array (like pixels but with additional 4 zero columns)
    cartesian_pixels = np.c_[pixels, np.zeros((height * width, 4))]
    # duplicate cartesian_pixels for all time samples
    cartesian_pixels = np.repeat(cartesian_pixels[:, np.newaxis, :], time_samples.shape[0], axis=1)

    # create frenet pixels array by projecting pixels on the Frenet frame
    pixels2D = pixels.reshape(height, width, 2)
    s_x, a_r, _, N_r, _, _ = frenet._project_cartesian_points(pixels2D)
    d_x = np.einsum('tpi,tpi->tp', pixels2D - a_r, N_r)
    frenet_pixels = np.c_[s_x.flatten() - s_x.flatten()[0], np.zeros((height * width, 2)), d_x.flatten(),
                          np.zeros((height * width, 2))]
    # duplicate frenet_pixels for all time samples
    frenet_pixels = np.repeat(frenet_pixels[:, np.newaxis, :], time_samples.shape[0], axis=1)

    # calculate cost components for all image pixels by building a static "trajectory" for every pixel
    pointwise_costs = TrajectoryPlannerCosts.compute_pointwise_costs(cartesian_pixels, frenet_pixels, state, cost_params,
                                                                     time_samples, planner.predictor, planner.dt, frenet)

    pixel_costs = (pointwise_costs[:, :, 0] + pointwise_costs[:, :, 1]).reshape(height, width, time_samples.shape[0])
    return pixels2D, pixel_costs


def visualize_test_scenario(route_points: np.array, reference_route_latitude: float, road_width: float,
                            state: State, goal: CartesianExtendedState,
                            ctrajectories: np.array, traj_costs: np.array,
                            pixels: np.array, pixel_costs: np.array,
                            plottable_obs: List[PlottableSigmoidBoxObstacle],
                            image_file_name: str):
    """
    Given running results (trajectories and their costs) of Werling planner on some scenario (road, state, goal),
    draw image (including trajectory alternatives and pixel-wise scores), and save it as PNG file.
    :param route_points: route points for Werling planner
    :param reference_route_latitude: [m] reference route latitude relatively to the right edge of the road
    :param road_width: [m] road width
    :param state: state including ego and obstacles
    :param goal: cartesian extended state of the goal
    :param ctrajectories: cartesian trajectories, obtained from Werling planner, operated on the state and the goal
    :param traj_costs: costs of the trajectories, obtained from Werling planner
    :param pixels: [height, width, 2], 2D array of pairs (x, y) coordinates of the visualization image pixels
    :param pixel_costs: [height, width] array of costs of pixels
    :param image_file_name: string
    :return:
    """

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(22, 11))
    p1 = fig.add_subplot(211)
    p2 = fig.add_subplot(212)

    x = pixels[:, :, 0]
    y = pixels[:, :, 1]
    z = pixel_costs[:, :, 0]  # the third index is time

    diff = np.diff(route_points[:, :2], axis=0)
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    angles = np.concatenate((angles, np.array([angles[-1]])))

    for p in list([p1, p2]):
        WerlingVisualizer.plot_obstacles(p, plottable_obs)
        WerlingVisualizer.plot_obstacles(p, plottable_obs)
        WerlingVisualizer.plot_route(p, route_points[:, :2])
        d = reference_route_latitude
        WerlingVisualizer.plot_route(p, np.c_[
            route_points[:, 0] + d * np.sin(angles), route_points[:, 1] - d * np.cos(angles)], '-k')
        d = road_width - reference_route_latitude
        WerlingVisualizer.plot_route(p, np.c_[
            route_points[:, 0] - d * np.sin(angles), route_points[:, 1] + d * np.cos(angles)], '-k')
        d = road_width / 2 - reference_route_latitude
        WerlingVisualizer.plot_route(p, np.c_[
            route_points[:, 0] - d * np.sin(angles), route_points[:, 1] + d * np.cos(angles)], '--k')

        # plot ego's best position for both obstacles
        origin = pixels[0, 0, :]
        ego = state.ego_state
        for obs in state.dynamic_objects:
            min_cost_y = np.argmin(z[:, int((obs.x - origin[0]) / 0.1)]) * 0.1 + origin[1]
            range_x = np.arange(-ego.size.length / 2, ego.size.length / 2 + 0.01) + obs.x
            p.plot(range_x, np.repeat(np.array([min_cost_y + ego.size.width / 2]), np.ceil(ego.size.length) + 1), '*w')
            p.plot(range_x, np.repeat(np.array([min_cost_y - ego.size.width / 2]), np.ceil(ego.size.length) + 1), '*w')
            range_y = np.arange(min_cost_y - ego.size.width / 2, min_cost_y + ego.size.width / 2 + 0.01)
            p.plot(np.repeat(np.array([-ego.size.length / 2 + obs.x]), np.ceil(ego.size.width) + 1), range_y, '*w')
            p.plot(np.repeat(np.array([ego.size.length / 2 + obs.x]), np.ceil(ego.size.width) + 1), range_y, '*w')

    d = reference_route_latitude + ROAD_SHOULDERS_WIDTH
    WerlingVisualizer.plot_route(p2, np.c_[
        route_points[:, 0] + d * np.sin(angles), route_points[:, 1] - d * np.cos(angles)], '-r')
    d = road_width - reference_route_latitude + ROAD_SHOULDERS_WIDTH
    WerlingVisualizer.plot_route(p2, np.c_[
        route_points[:, 0] - d * np.sin(angles), route_points[:, 1] + d * np.cos(angles)], '-r')

    z = np.log(1 + z)
    p2.contourf(x, y, z, 100)

    WerlingVisualizer.plot_best(p2, ctrajectories[0])
    WerlingVisualizer.plot_alternatives(p1, ctrajectories, traj_costs)

    WerlingVisualizer.plot_goal(p1, goal)
    WerlingVisualizer.plot_goal(p2, goal)

    WerlingVisualizer.plot_route(p1, route_points)

    fig.savefig(image_file_name)

    # fig.show()
    fig.clear()


def test_samplableWerlingTrajectory_sampleAfterTd_correctLateralPosition():
    route_points = RouteFixture.get_route(lng=10, k=1, step=1, lat=3, offset=-.5)

    frenet = FrenetSerret2DFrame.fit(route_points)

    trajectory = SamplableWerlingTrajectory(
        timestamp_in_sec=10.0,
        T_s=1.5,
        T_d=1.0,
        T_extended=1.5,
        frenet_frame=frenet,
        poly_s_coefs=np.array([-2.53400421e+00, 8.90980541e+00, -7.72383669e+00, -3.76008007e-03, 6.00604195e+00, 1.00520801e+00]),
        poly_d_coefs=np.array([-1.44408865e+01, 3.62482582e+01, -2.42818417e+01, -3.62145365e-02, 1.03423064e-02, 5.01250837e-01])
    )

    fstate_terminal = frenet.cstate_to_fstate(trajectory.sample(
        np.array([trajectory.timestamp_in_sec + trajectory.T_s]))[0])

    fstate_after_T_d = frenet.cstate_to_fstate(trajectory.sample(
        np.array([trajectory.timestamp_in_sec + (trajectory.T_s + trajectory.T_d) / 2]))[0])

    np.testing.assert_allclose(fstate_after_T_d[FS_DX], fstate_terminal[FS_DX])


def test_computeJerk_simpleTrajectory():
    p1: CartesianExtendedState = np.array([0, 0, 0, 1, 0, 0.1])
    p2: CartesianExtendedState = np.array([0, 0, 0, 2, 1, 0.1])
    ctrajectory: CartesianTrajectory = np.array([p1, p2])
    lon_jerks, lat_jerks = Jerk.compute_jerks(np.array([ctrajectory]), 0.1)
    assert np.isclose(lon_jerks[0][0], 10) and np.isclose(lat_jerks[0][0], 0.9)
