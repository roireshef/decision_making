import time
from unittest.mock import patch

import numpy as np
import pytest

from decision_making.src.global_constants import OBSTACLE_SIGMOID_K_PARAM, LATERAL_SAFETY_MARGIN_FROM_OBJECT, \
    OBSTACLE_SIGMOID_COST, DEVIATION_FROM_ROAD_COST, DEVIATION_TO_SHOULDER_COST, DEVIATION_FROM_LANE_COST, \
    ROAD_SIGMOID_K_PARAM, EGO_LENGTH, EGO_WIDTH, \
    SHOULDER_SIGMOID_OFFSET, SHOULDER_SIGMOID_K_PARAM, VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS, \
    DEFAULT_ACCELERATION, DEFAULT_CURVATURE, EGO_HEIGHT, LANE_SIGMOID_K_PARAM, \
    DEVIATION_FROM_GOAL_LAT_FACTOR, DEVIATION_FROM_GOAL_COST, GOAL_SIGMOID_K_PARAM, GOAL_SIGMOID_OFFSET, TD_STEPS
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams, SigmoidFunctionParams
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.types import CURVE_X, CURVE_Y, CURVE_YAW
from decision_making.src.planning.types import C_Y
from decision_making.src.planning.utils.math import Math
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import State, ObjectSize, EgoState, DynamicObject
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.planning.trajectory.utils import RouteFixture, PlottableSigmoidDynamicBoxObstacle, \
    WerlingVisualizer, PlottableSigmoidStaticBoxObstacle
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.transformations.geometry_utils import CartesianFrame
from mapping.test.model.testable_map_fixtures import map_api_mock
from rte.python.logger.AV_logger import AV_Logger


mock_td_steps = 5

# @patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
# @patch('decision_making.test.planning.trajectory.test_werlingPlanner.TD_STEPS', mock_td_steps)
# @patch('decision_making.src.planning.trajectory.optimal_control.werling_planner.TD_STEPS', mock_td_steps)
# @patch('decision_making.src.planning.trajectory.optimal_control.werling_planner.SX_STEPS', 5)
# @patch('decision_making.src.planning.trajectory.optimal_control.werling_planner.DX_STEPS', 5)
def test_werlingPlanner_toyScenario_noException():
    logger = AV_Logger.get_logger('test_werlingPlanner_toyScenario_noException')
    route_points = CartesianFrame.add_yaw_and_derivatives(
        RouteFixture.get_route(lng=10, k=1, step=1, lat=3, offset=-.5))

    v0 = 6
    vT = 10
    v_min = 0
    v_max = 10
    a_min = -5
    a_max = 5
    Ts = 1.5

    predictor = RoadFollowingPredictor(logger)

    goal = np.concatenate((route_points[len(route_points) // 2, [CURVE_X, CURVE_Y, CURVE_YAW]], [vT, DEFAULT_ACCELERATION, DEFAULT_CURVATURE]))
    pos1 = np.array([7, -.5])
    yaw1 = 0
    pos2 = np.array([11, 1.5])
    yaw2 = np.pi / 4

    obs = list([
        DynamicObject(obj_id=0, timestamp=950*10e6, x=pos1[0], y=pos1[1], z=0, yaw=yaw1, size=ObjectSize(1.5, 0.5, 0),
                      confidence=1.0, v_x=0, v_y=0, acceleration_lon=0.0, omega_yaw=0.0),
        DynamicObject(obj_id=0, timestamp=950*10e6, x=pos2[0], y=pos2[1], z=0, yaw=yaw2, size=ObjectSize(1.5, 0.5, 0),
                      confidence=1.0, v_x=0, v_y=0, acceleration_lon=0.0, omega_yaw=0.0)
    ])

    ego = EgoState(obj_id=-1, timestamp=1000*10e6, x=0, y=0, z=0, yaw=0, size=ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT),
                   confidence=1.0, v_x=v0, v_y=0, steering_angle=0.0, acceleration_lon=0.0, omega_yaw=0.0)

    state = State(occupancy_state=None, dynamic_objects=obs, ego_state=ego)

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
                                       velocity_limits=np.array([-np.inf, np.inf]),     # TODO: temporary because this is solved in other PR
                                       lon_acceleration_limits=np.array([-np.inf, np.inf]),   # TODO: temporary because this is solved in other PR
                                       lat_acceleration_limits=np.array([-np.inf, np.inf]))   # TODO: temporary because this is solved in other PR

    planner = WerlingPlanner(logger, predictor)

    start_time = time.time()

    samplable, ctrajectories, costs = planner.plan(state=state, reference_route=route_points[:, :2], goal=goal,
                                                   time_horizon=Ts, cost_params=cost_params)

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
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    # plt.show()
    # fig.clear()

@pytest.mark.skip(reason="takes too long.")
@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_werlingPlanner_twoStaticObjScenario_withCostViz():
    """
    The route is set to route_points by calling to RouteFixture.get_route(). Currently it is a straight line.
    The test runs with 16 iterations. In each iteration one or more obstacles move.
    At each iteration the image with costs and calculated trajectories is saved in a file.
    The number of obstacles is determined by the length of obs_poses.
    """
    logger = AV_Logger.get_logger('test_werlingPlanner_twoStaticObjScenario_withCostViz')

    lane_width = 3.6
    num_lanes = 2
    road_width = num_lanes*lane_width
    reference_route_latitude = 3*lane_width/2

    xrange = (0, 60)
    yrange = (-2, 10)
    x = np.arange(xrange[0], xrange[1], 0.1)
    y = np.arange(yrange[0], yrange[1], 0.1)
    width = x.shape[0]
    height = y.shape[0]
    points = np.array([np.transpose([np.tile(x, y.shape[0]), np.repeat(y, x.shape[0])])])

    route_points = CartesianFrame.add_yaw_and_derivatives(
        RouteFixture.get_route(lng=20, k=1, step=1, lat=0, offset=reference_route_latitude))

    v0 = 6
    vT = 10
    T = 6.8

    goal_latitude = reference_route_latitude
    target_lane = int(goal_latitude / lane_width)
    goal = np.concatenate(
        (route_points[-1, [CURVE_X, CURVE_Y, CURVE_YAW]], [vT, DEFAULT_ACCELERATION, DEFAULT_CURVATURE]))
    goal[C_Y] = goal_latitude

    predictor = RoadFollowingPredictor(logger)

    for test_idx in range(16):

        if test_idx < 8:
            obs_poses = np.array([np.array([4, 5.4]), np.array([14, 6.0]), np.array([24, 7.5]), np.array([40, 8.8]),
                                  np.array([54, 0.8 + test_idx*0.2])])
            start_ego_lat = lane_width/2
            obs_yaws = [0, 0, 0, 0, np.pi / 32]
        else:
            obs_poses = np.array([np.array([54, -1.6 + test_idx*0.4])])
            start_ego_lat = 3*lane_width / 2
            obs_yaws = [0]
        obs = []
        for i, pose in enumerate(obs_poses):
            obs.append(DynamicObject(obj_id=i, timestamp=0, x=pose[0], y=pose[1], z=0, yaw=obs_yaws[i],
                                     size=ObjectSize(4, 1.8, 0), confidence=1.0, v_x=0, v_y=0,
                                     acceleration_lon=0.0, omega_yaw=0.0))
        #obs = list([])

        ego = EgoState(obj_id=-1, timestamp=0, x=0.001, y=start_ego_lat, z=0, yaw=0,
                       size=ObjectSize(EGO_LENGTH, EGO_WIDTH, 0),
                       confidence=1.0, v_x=v0, v_y=0, steering_angle=0.0, acceleration_lon=0.0, omega_yaw=0.0)

        state = State(occupancy_state=None, dynamic_objects=obs, ego_state=ego)

        objects_dilation_length = ego.size.length / 2 + LATERAL_SAFETY_MARGIN_FROM_OBJECT
        objects_dilation_width = ego.size.width / 2 + LATERAL_SAFETY_MARGIN_FROM_OBJECT
        right_lane_offset = max(0.0, reference_route_latitude - ego.size.width / 2 - target_lane * lane_width)
        left_lane_offset = (road_width - reference_route_latitude) - ego.size.width / 2 - (num_lanes - target_lane - 1) * lane_width
        right_shoulder_offset = reference_route_latitude - ego.size.width / 2 + SHOULDER_SIGMOID_OFFSET
        left_shoulder_offset = (road_width - reference_route_latitude) - ego.size.width / 2 + SHOULDER_SIGMOID_OFFSET
        right_road_offset = reference_route_latitude - ego.size.width / 2 + ROAD_SHOULDERS_WIDTH
        left_road_offset = (road_width - reference_route_latitude) - ego.size.width / 2 + ROAD_SHOULDERS_WIDTH

        cost_params = TrajectoryCostParams(
            left_lane_cost=SigmoidFunctionParams(DEVIATION_FROM_LANE_COST, ROAD_SIGMOID_K_PARAM, left_lane_offset),
            right_lane_cost=SigmoidFunctionParams(DEVIATION_FROM_LANE_COST, ROAD_SIGMOID_K_PARAM, right_lane_offset),
            left_road_cost=SigmoidFunctionParams(DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM, left_road_offset),
            right_road_cost=SigmoidFunctionParams(DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM, right_road_offset),
            left_shoulder_cost=SigmoidFunctionParams(DEVIATION_TO_SHOULDER_COST, ROAD_SIGMOID_K_PARAM, left_shoulder_offset),
            right_shoulder_cost=SigmoidFunctionParams(DEVIATION_TO_SHOULDER_COST, ROAD_SIGMOID_K_PARAM, right_shoulder_offset),
            obstacle_cost_x=SigmoidFunctionParams(OBSTACLE_SIGMOID_COST, OBSTACLE_SIGMOID_K_PARAM, objects_dilation_length),
            obstacle_cost_y=SigmoidFunctionParams(OBSTACLE_SIGMOID_COST, OBSTACLE_SIGMOID_K_PARAM, objects_dilation_width),
            dist_from_goal_cost=SigmoidFunctionParams(DEVIATION_FROM_GOAL_COST, GOAL_SIGMOID_K_PARAM, GOAL_SIGMOID_OFFSET),
            dist_from_goal_lat_factor=DEVIATION_FROM_GOAL_LAT_FACTOR,
            velocity_limits=VELOCITY_LIMITS,
            lon_acceleration_limits=LON_ACC_LIMITS,
            lat_acceleration_limits=LAT_ACC_LIMITS)

        planner = WerlingPlanner(logger, predictor)

        start_time = time.time()

        samplable, ctrajectories, costs = planner.plan(state=state, reference_route=route_points[:, :2], goal=goal,
                                                        goal_time=T, cost_params=cost_params)

        end_time = time.time() - start_time

        obs_costs = np.zeros(width * height)
        for obj in obs:
            sobj = PlottableSigmoidStaticBoxObstacle(obj, k=OBSTACLE_SIGMOID_K_PARAM,
                                                     margin=np.array([objects_dilation_length, objects_dilation_width]))
            obs_costs += OBSTACLE_SIGMOID_COST * sobj.compute_cost_per_point(points)[0]

        latitudes = points[0][:, 1]
        left_lane_offsets = (latitudes-reference_route_latitude) - left_lane_offset
        right_lane_offsets = -(latitudes-reference_route_latitude) - right_lane_offset
        left_shoulder_offsets = (latitudes-reference_route_latitude) - left_shoulder_offset
        right_shoulder_offsets = -(latitudes-reference_route_latitude) - right_shoulder_offset
        left_road_offsets = (latitudes-reference_route_latitude) - left_road_offset
        right_road_offsets = -(latitudes-reference_route_latitude) - right_road_offset

        road_deviations_costs = \
            Math.clipped_sigmoid(left_lane_offsets, DEVIATION_FROM_LANE_COST, LANE_SIGMOID_K_PARAM) + \
            Math.clipped_sigmoid(right_lane_offsets, DEVIATION_FROM_LANE_COST, LANE_SIGMOID_K_PARAM) + \
            Math.clipped_sigmoid(left_shoulder_offsets, DEVIATION_TO_SHOULDER_COST, SHOULDER_SIGMOID_K_PARAM) + \
            Math.clipped_sigmoid(right_shoulder_offsets, DEVIATION_TO_SHOULDER_COST, SHOULDER_SIGMOID_K_PARAM) + \
            Math.clipped_sigmoid(left_road_offsets, DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM) + \
            Math.clipped_sigmoid(right_road_offsets, DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM)

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(22, 11))
        p1 = fig.add_subplot(211)
        p2 = fig.add_subplot(212)
        time_samples = np.arange(0.0, T, 0.1)
        offsets = np.array([cost_params.obstacle_cost_x.offset, cost_params.obstacle_cost_y.offset])
        plottable_obs = [PlottableSigmoidDynamicBoxObstacle(o, cost_params.obstacle_cost_x.k, offsets, time_samples,
                                                            predictor)
                         for o in state.dynamic_objects]

        x = points[0, :, 0].reshape(height, width)
        y = points[0, :, 1].reshape(height, width)
        z = obs_costs.reshape(height, width) + road_deviations_costs.reshape(height, width)

        for p in list([p1, p2]):
            WerlingVisualizer.plot_obstacles(p, plottable_obs)
            p.plot(np.arange(0, xrange[1]), np.repeat(np.array([0]), xrange[1]), '-k')
            p.plot(np.arange(0, xrange[1]), np.repeat(np.array([road_width]), xrange[1]), '-k')
            p.plot(np.arange(0, xrange[1]), np.repeat(np.array([-1.5]), xrange[1]), '-r')
            p.plot(np.arange(0, xrange[1]), np.repeat(np.array([road_width+1.5]), xrange[1]), '-r')
            p.plot(np.arange(0, xrange[1]), np.repeat(np.array([road_width/2]), xrange[1]), '--k')

            # plot ego's best position for both obstacles
            for obs in state.dynamic_objects:
                min_cost_y = np.argmin(z[:, int((obs.x - xrange[0])/0.1)]) * 0.1 + yrange[0]
                p.plot(np.arange(- ego.size.length / 2, ego.size.length / 2 + 0.01) + obs.x,
                       np.repeat(np.array([min_cost_y+ego.size.width/2]), np.ceil(ego.size.length)+1), '*w')
                p.plot(np.arange(- ego.size.length / 2, ego.size.length / 2 + 0.01) + obs.x,
                       np.repeat(np.array([min_cost_y - ego.size.width / 2]), np.ceil(ego.size.length)+1), '*w')
                p.plot(np.repeat(np.array([-ego.size.length / 2 + obs.x]), np.ceil(ego.size.width)+1),
                       np.arange(min_cost_y-ego.size.width/2, min_cost_y+ego.size.width/2 + 0.01), '*w')
                p.plot(np.repeat(np.array([ego.size.length / 2 + obs.x]), np.ceil(ego.size.width)+1),
                       np.arange(min_cost_y-ego.size.width/2, min_cost_y+ego.size.width/2 + 0.01), '*w')

        z = np.log(1 + z)
        p2.contourf(x, y, z, 100)

        WerlingVisualizer.plot_obstacles(p1, plottable_obs)
        WerlingVisualizer.plot_obstacles(p2, plottable_obs)
        WerlingVisualizer.plot_route(p1, route_points[:, :2])
        WerlingVisualizer.plot_route(p2, route_points[:, :2])

        WerlingVisualizer.plot_best(p2, ctrajectories[0])
        WerlingVisualizer.plot_alternatives(p1, ctrajectories, costs)

        WerlingVisualizer.plot_goal(p2, goal)

        print(costs)

        WerlingVisualizer.plot_route(p1, route_points)

        filename = 'test_costs'+str(test_idx)+'.png'
        fig.savefig(filename)

    fig.show()
    fig.clear()
